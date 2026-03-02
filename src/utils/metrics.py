"""
src/utils/metrics.py
--------------------
Per-task and aggregate metric computation for the Tox21 multi-label
classification problem.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
)
from typing import Dict, List, Tuple


TOX21_TASKS: List[str] = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def compute_per_task_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    task_names: List[str] = TOX21_TASKS,
) -> Dict[str, Dict[str, float]]:
    """
    Compute AUROC, AUPRC, and accuracy for each task, skipping tasks
    where only one class is present in y_true (undefined AUROC).

    Args:
        y_true  : (N, T) ground truth labels, NaN for missing
        y_score : (N, T) predicted probabilities
        y_pred  : (N, T) predicted binary labels (threshold 0.5)
        task_names: list of T task names

    Returns:
        dict mapping task_name → {auroc, auprc, accuracy}
    """
    results = {}
    n_tasks = y_true.shape[1]

    for t in range(n_tasks):
        mask = ~np.isnan(y_true[:, t])
        yt = y_true[mask, t].astype(int)
        ys = y_score[mask, t]
        yp = y_pred[mask, t].astype(int)

        if len(np.unique(yt)) < 2:
            # Cannot compute AUROC with a single class present
            results[task_names[t]] = {"auroc": float("nan"),
                                      "auprc": float("nan"),
                                      "accuracy": float("nan")}
            continue

        results[task_names[t]] = {
            "auroc":    round(roc_auc_score(yt, ys), 4),
            "auprc":    round(average_precision_score(yt, ys), 4),
            "accuracy": round(accuracy_score(yt, yp), 4),
        }
    return results


def mean_auroc(metrics: Dict[str, Dict[str, float]]) -> float:
    """Return mean AUROC across tasks, ignoring NaN."""
    vals = [v["auroc"] for v in metrics.values() if not np.isnan(v["auroc"])]
    return round(float(np.mean(vals)), 4) if vals else float("nan")


def worst_task(metrics: Dict[str, Dict[str, float]]) -> Tuple[str, float]:
    """Return (task_name, auroc) for the task with the lowest AUROC."""
    valid = {k: v["auroc"] for k, v in metrics.items()
             if not np.isnan(v["auroc"])}
    worst = min(valid, key=valid.get)
    return worst, valid[worst]


def print_metrics_table(metrics: Dict[str, Dict[str, float]]) -> None:
    header = f"{'Task':<18} {'AUROC':>8} {'AUPRC':>8} {'Acc':>8}"
    print(header)
    print("-" * len(header))
    for task, m in metrics.items():
        print(f"{task:<18} {m['auroc']:>8.4f} {m['auprc']:>8.4f} {m['accuracy']:>8.4f}")
    print("-" * len(header))
    print(f"{'Mean AUROC':<18} {mean_auroc(metrics):>8.4f}")
