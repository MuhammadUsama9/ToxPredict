"""
evaluate.py
-----------
Load the best checkpoint and produce:
  - Per-task AUROC / AUPRC / Accuracy table
  - Confusion matrix for the worst-performing task
  - Error analysis: top-10 misclassified compounds
  - Scatter: predicted probability vs. true label

Usage:
  python evaluate.py --checkpoint checkpoints/best_gcn.pt \
                     --test-csv data/processed/test.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader

from src.data.dataset import MoleculeGraphDataset
from src.models.gcn_model import ToxGCN
from src.utils.metrics import (
    compute_per_task_metrics, mean_auroc, worst_task,
    print_metrics_table, TOX21_TASKS,
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ToxGCN on Tox21 test set")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_gcn.pt")
    p.add_argument("--test-csv",   type=str, default="data/processed/test.csv")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threshold",  type=float, default=0.5)
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_logits, all_labels, all_smiles = [], [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(batch.y.squeeze(1).cpu().numpy())
        all_smiles.extend(batch.smiles)

    return (
        np.vstack(all_logits),
        np.vstack(all_labels),
        all_smiles,
    )


def plot_confusion_matrix(y_true, y_pred, task_name, save_path):
    valid = ~np.isnan(y_true)
    yt = y_true[valid].astype(int)
    yp = y_pred[valid].astype(int)
    cm = confusion_matrix(yt, yp)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred: 0", "Pred: 1"],
                yticklabels=["True: 0", "True: 1"])
    ax.set_title(f"Confusion Matrix — {task_name}")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved to {save_path}")


def error_analysis(y_true, y_probs, y_preds, smiles, task_name, task_idx):
    """
    Find the top-10 most confident misclassifications for the worst task.
    Confidence = |prob - 0.5| (higher = model more wrong).
    """
    mask = ~np.isnan(y_true[:, task_idx])
    yt   = y_true[mask, task_idx].astype(int)
    yp   = y_preds[mask, task_idx].astype(int)
    ypr  = y_probs[mask, task_idx]
    sm   = np.array(smiles)[mask]

    wrong = yt != yp
    if wrong.sum() == 0:
        print("[evaluate] No misclassifications found — perfect task?")
        return

    # Sort by confidence of error (most wrong first)
    conf   = np.abs(ypr - 0.5)
    idx    = np.argsort(conf[wrong])[::-1][:10]
    rows   = []
    wrong_idx = np.where(wrong)[0]
    for i in wrong_idx[idx]:
        rows.append({
            "smiles":    sm[i],
            "true_label": yt[i],
            "pred_label": yp[i],
            "pred_prob":  round(float(ypr[i]), 4),
        })

    df_err = pd.DataFrame(rows)
    out = OUTPUT_DIR / f"error_analysis_{task_name.replace('-','_')}.csv"
    df_err.to_csv(out, index=False)
    print(f"\n[evaluate] Top-10 misclassifications for {task_name}:")
    print(df_err.to_string(index=False))
    print(f"\n  Saved to {out}")
    return df_err


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    hp   = ckpt["args"]
    print(f"[evaluate] Loaded checkpoint: {args.checkpoint}")
    print(f"           Epoch={ckpt['epoch']}  val_auroc={ckpt['val_auroc']:.4f}")

    model = ToxGCN(
        node_feat=34,
        hidden=hp.get("hidden", 128),
        num_tasks=12,
        dropout=hp.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])

    # ── Load test data ─────────────────────────────────────────────────────
    test_ds     = MoleculeGraphDataset(args.test_csv)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ── Inference ─────────────────────────────────────────────────────────
    logits, labels, smiles = run_inference(model, test_loader, device)
    probs = 1 / (1 + np.exp(-logits))          # sigmoid
    preds = (probs >= args.threshold).astype(float)

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics     = compute_per_task_metrics(labels, probs, preds)
    m_auroc     = mean_auroc(metrics)
    worst, wauc = worst_task(metrics)
    worst_idx   = TOX21_TASKS.index(worst)

    print(f"\n{'='*55}")
    print(f"TEST RESULTS  |  mean AUROC = {m_auroc:.4f}")
    print(f"{'='*55}")
    print_metrics_table(metrics)

    # ── Worst-task confusion matrix ────────────────────────────────────────
    plot_confusion_matrix(
        labels[:, worst_idx], preds[:, worst_idx],
        worst, OUTPUT_DIR / f"confusion_{worst.replace('-','_')}.png"
    )

    # ── Error analysis ─────────────────────────────────────────────────────
    error_analysis(labels, probs, preds, smiles, worst, worst_idx)

    # ── Probability distribution plot ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for t, task in enumerate(TOX21_TASKS):
        mask = ~np.isnan(labels[:, t])
        ax.hist(probs[mask, t], bins=50, alpha=0.4, label=task)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Probability Distributions per Task")
    ax.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "prob_distributions.png", dpi=150)
    plt.close(fig)
    print(f"\n[evaluate] Plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
