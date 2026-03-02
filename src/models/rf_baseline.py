"""
src/models/rf_baseline.py
--------------------------
Random Forest baseline trained on ECFP4 Morgan fingerprints.
Provides a strong tabular benchmark against which the GCN is compared.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score


TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def train_rf_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
    checkpoint_path: str = "checkpoints/rf_baseline.pkl",
) -> MultiOutputClassifier:
    """
    Train a MultiOutput Random Forest on Morgan fingerprints.
    Missing labels (NaN) are handled per-task by subsetting rows.

    Args:
        X_train: (N_train, 2048) fingerprint matrix
        y_train: (N_train, 12) label matrix, NaN for missing
        X_val:   (N_val, 2048)
        y_val:   (N_val, 12)

    Returns:
        Fitted MultiOutputClassifier.
    """
    # Train per-task RF; each task trains only on its non-NaN subset
    per_task_models = {}
    val_aurocs = []

    for t, task in enumerate(TOX21_TASKS):
        yt_train = y_train[:, t]
        yt_val   = y_val[:, t]

        train_mask = ~np.isnan(yt_train)
        val_mask   = ~np.isnan(yt_val)

        if train_mask.sum() == 0 or val_mask.sum() == 0:
            per_task_models[task] = None
            continue

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",    # handles class imbalance
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_train[train_mask], yt_train[train_mask].astype(int))
        per_task_models[task] = rf

        # Validate
        yt_pred_proba = rf.predict_proba(X_val[val_mask])[:, 1]
        yt_true_val   = yt_val[val_mask].astype(int)

        if len(np.unique(yt_true_val)) > 1:
            auroc = roc_auc_score(yt_true_val, yt_pred_proba)
            val_aurocs.append(auroc)
            print(f"  [{task:<18}] val AUROC = {auroc:.4f}")

    mean_auroc = np.mean(val_aurocs) if val_aurocs else float("nan")
    print(f"\n[RF baseline] Mean val AUROC = {mean_auroc:.4f}")

    # Save all per-task models
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(per_task_models, checkpoint_path)
    print(f"[RF baseline] Saved to {checkpoint_path}")

    return per_task_models


def predict_rf(
    per_task_models: dict,
    X: np.ndarray,
) -> np.ndarray:
    """
    Run inference with per-task RF models.

    Returns:
        prob_matrix: (N, 12) predicted probabilities, NaN for tasks without model
    """
    N = X.shape[0]
    probs = np.full((N, len(TOX21_TASKS)), np.nan)
    for t, task in enumerate(TOX21_TASKS):
        rf = per_task_models.get(task)
        if rf is None:
            continue
        probs[:, t] = rf.predict_proba(X)[:, 1]
    return probs
