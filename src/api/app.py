"""
src/api/app.py
--------------
FastAPI inference service for molecular toxicity prediction.

Routes:
  GET  /health           — liveness check
  POST /predict          — SMILES → 12-task toxicity probabilities
  POST /predict/batch    — list of SMILES → batch predictions

Deployment (M2 requirement):
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 2
  # OR via Docker: docker run -p 8000:8000 qsar-tox21
"""

import time
from datetime import datetime
from pathlib import Path
from typing import List

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.data.dataset import smiles_to_pyg
from src.models.gcn_model import ToxGCN
from src.utils.metrics import TOX21_TASKS

# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT_PATH = Path("checkpoints/best_gcn.pt")
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model once at startup ────────────────────────────────────────────────
_model: ToxGCN | None = None


def _load_model() -> ToxGCN:
    global _model
    if _model is not None:
        return _model

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Run `python train.py` first."
        )

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    hp   = ckpt.get("args", {})

    model = ToxGCN(
        node_feat=34,
        hidden=hp.get("hidden", 128),
        num_tasks=12,
        dropout=0.0,         # inference: disable dropout
    ).to(DEVICE)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _model = model
    return model


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="QSAR Toxicity Predictor",
    description=(
        "Predicts the probability of a chemical compound being toxic "
        "across the 12 Tox21 assay endpoints using a Graph Convolutional Network."
    ),
    version="1.0.0",
)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class SMILESInput(BaseModel):
    smiles: str

    @field_validator("smiles")
    @classmethod
    def smiles_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("SMILES string must not be empty.")
        return v.strip()


class BatchSMILESInput(BaseModel):
    smiles_list: List[str]


class ToxPrediction(BaseModel):
    smiles:             str
    task_probabilities: dict   # {task_name: probability}
    latency_ms:         float
    timestamp_iso:      str


class BatchToxPrediction(BaseModel):
    predictions: List[ToxPrediction]
    mean_latency_ms: float
    timestamp_iso:   str


# ── Helpers ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def _predict_single(smiles: str, model: ToxGCN) -> dict:
    """Convert SMILES → probabilities dict."""
    from torch_geometric.data import Batch

    graph = smiles_to_pyg(smiles)
    if graph is None:
        raise HTTPException(status_code=422,
                            detail=f"Invalid SMILES string: '{smiles}'")

    batch = Batch.from_data_list([graph]).to(DEVICE)
    logits = model(batch)                          # (1, 12)
    probs  = torch.sigmoid(logits).cpu().numpy()[0]  # (12,)

    return {task: round(float(p), 4) for task, p in zip(TOX21_TASKS, probs)}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
def health_check():
    """Liveness check — returns OK and model status."""
    model_loaded = _model is not None
    return {"status": "ok", "model_loaded": model_loaded, "device": str(DEVICE)}


@app.get("/model-info", tags=["info"])
def get_model_info():
    """Returns basic information about the deployed model."""
    try:
        model = _load_model()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"model_name": "ToxGCN", "trainable_parameters": params, "supported_tasks": TOX21_TASKS}
    except FileNotFoundError:
        return {"status": "error", "message": "Model checkpoint not found."}


@app.post("/predict", response_model=ToxPrediction, tags=["inference"])
def predict(payload: SMILESInput):
    """
    Predict Tox21 toxicity probabilities for a single SMILES.

    Example request:
        {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}

    Returns:
        task_probabilities: dict mapping each Tox21 assay to P(toxic).
    """
    model = _load_model()
    t0    = time.perf_counter()
    probs = _predict_single(payload.smiles, model)
    latency_ms = (time.perf_counter() - t0) * 1000

    return ToxPrediction(
        smiles=payload.smiles,
        task_probabilities=probs,
        latency_ms=round(latency_ms, 2),
        timestamp_iso=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/predict/batch", response_model=BatchToxPrediction, tags=["inference"])
def predict_batch(payload: BatchSMILESInput):
    """
    Predict toxicity probabilities for a batch of SMILES strings.
    Useful for high-throughput virtual screening.
    """
    if len(payload.smiles_list) > 100:
        raise HTTPException(status_code=400,
                            detail="Maximum batch size is 100.")

    model = _load_model()
    predictions = []
    latencies   = []

    for smi in payload.smiles_list:
        t0    = time.perf_counter()
        try:
            probs = _predict_single(smi, model)
        except HTTPException:
            probs = {task: None for task in TOX21_TASKS}
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)
        predictions.append(ToxPrediction(
            smiles=smi,
            task_probabilities=probs,
            latency_ms=round(latency_ms, 2),
            timestamp_iso=datetime.utcnow().isoformat() + "Z",
        ))

    return BatchToxPrediction(
        predictions=predictions,
        mean_latency_ms=round(float(np.mean(latencies)), 2),
        timestamp_iso=datetime.utcnow().isoformat() + "Z",
    )
