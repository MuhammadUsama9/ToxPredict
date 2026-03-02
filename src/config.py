"""
src/config.py
-------------
Centralized configuration for the Tox21 toxicity prediction project.
"""

from pathlib import Path

# ── Project Paths ────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# ── Tox21 Dataset Task List ──────────────────────────────────────────────────

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

# ── Model Hyperparameters (Defaults) ─────────────────────────────────────────

DEFAULT_HIDDEN_DIM = 128
DEFAULT_DROPOUT = 0.3
NODE_FEAT_DIM = 34
EDGE_FEAT_DIM = 5
NUM_TASKS = 12

# ── Training Defaults ────────────────────────────────────────────────────────

DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_PATIENCE = 10
DEFAULT_SEED = 42
