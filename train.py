"""
train.py
--------
Main training script for the Tox21 GCN toxicity predictor.

Training command (exact, as used for the questionnaire):
  python train.py --lr 1e-3 --epochs 50 --batch-size 64 --hidden 128 \
                  --dropout 0.3 --seed 42 --mlflow-uri ./mlruns

The script:
  - Sets all seeds before any data loading (reproducibility).
  - Builds PyG DataLoaders for train / val splits.
  - Computes per-task positive class weights for imbalance handling.
  - Trains ToxGCN with masked BCE + L2 weight decay.
  - Logs epoch metrics to MLflow (AUROC, loss).
  - Applies early stopping (patience=10, criterion=mean val AUROC).
  - Saves best checkpoint to checkpoints/best_gcn.pt.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import mlflow
import mlflow.pytorch

from src.data.dataset import MoleculeGraphDataset
from src.models.gcn_model import ToxGCN, masked_bce_loss
from src.utils.seed_utils import set_all_seeds
from src.utils.metrics import (
    compute_per_task_metrics, mean_auroc, print_metrics_table
)
from src.config import (
    TOX21_TASKS, DEFAULT_LR, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN_DIM, DEFAULT_DROPOUT, DEFAULT_PATIENCE, DEFAULT_SEED,
    CHECKPOINT_DIR
)


def parse_args():
    p = argparse.ArgumentParser(description="Train ToxGCN on Tox21")
    p.add_argument("--lr",          type=float, default=DEFAULT_LR,
                   help="Learning rate (searched: 1e-4 to 1e-2)")
    p.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--hidden",      type=int,   default=DEFAULT_HIDDEN_DIM,
                   help="GCN hidden channel width (searched: 64, 128, 256)")
    p.add_argument("--dropout",     type=float, default=DEFAULT_DROPOUT,
                   help="Dropout rate (searched: 0.1, 0.3, 0.5)")
    p.add_argument("--weight-decay",type=float, default=1e-4,
                   help="L2 regularisation coefficient")
    p.add_argument("--patience",    type=int,   default=DEFAULT_PATIENCE,
                   help="Early stopping patience (epochs)")
    p.add_argument("--seed",        type=int,   default=DEFAULT_SEED)

    p.add_argument("--train-csv",   type=str,   default="data/processed/train.csv")
    p.add_argument("--val-csv",     type=str,   default="data/processed/val.csv")
    p.add_argument("--mlflow-uri",  type=str,   default="./mlruns")
    p.add_argument("--device",      type=str,   default="auto",
                   help="'cpu', 'cuda', or 'auto'")
    p.add_argument("--force",       action="store_true",
                   help="Force re-training even if checkpoint exists")
    return p.parse_args()



def train_one_epoch(model, loader, optimizer, device, pos_weight):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)            # (B, 12)
        loss   = masked_bce_loss(logits, batch.y.squeeze(1), pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (mean_loss, per_task_metrics_dict, mean_auroc)."""
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch)
        loss   = masked_bce_loss(logits, batch.y.squeeze(1))
        total_loss += loss.item() * batch.num_graphs
        all_logits.append(logits.cpu().numpy())
        all_labels.append(batch.y.squeeze(1).cpu().numpy())

    all_logits = np.vstack(all_logits)    # (N, 12)
    all_labels = np.vstack(all_labels)    # (N, 12)
    all_probs  = 1 / (1 + np.exp(-all_logits))   # sigmoid
    all_preds  = (all_probs >= 0.5).astype(float)

    metrics  = compute_per_task_metrics(all_labels, all_probs, all_preds)
    m_auroc  = mean_auroc(metrics)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, metrics, m_auroc



def compute_pos_weights(train_dataset, num_tasks=12, device="cpu"):
    """
    For each task compute pos_weight = (# negatives) / (# positives).
    This upweights the rare toxic class in the BCE loss.
    """
    all_y = []
    for data in train_dataset:
        all_y.append(data.y.squeeze(0).numpy())
    all_y = np.array(all_y)   # (N, 12)

    pos_weights = []
    for t in range(num_tasks):
        col = all_y[:, t]
        valid = col[~np.isnan(col)]
        n_pos = (valid == 1).sum()
        n_neg = (valid == 0).sum()
        w = n_neg / n_pos if n_pos > 0 else 1.0
        pos_weights.append(w)

    return torch.tensor(pos_weights, dtype=torch.float).to(device)



def main():
    args = parse_args()

    set_all_seeds(args.seed)
    best_checkpoint = CHECKPOINT_DIR / "best_gcn.pt"

    if best_checkpoint.exists() and not args.force:
        print(f"[train] Found existing checkpoint at {best_checkpoint}. Skipping training.")
        print("[train] Run with --force to re-train.")
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[train] Using device: {device}")

    print("[train] Loading datasets ...")
    train_ds = MoleculeGraphDataset(args.train_csv)
    val_ds   = MoleculeGraphDataset(args.val_csv)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    pos_weight = compute_pos_weights(train_ds, device=str(device))
    print(f"[train] Positive class weights (top 3): "
          f"{pos_weight[:3].tolist()}")

    model = ToxGCN(
        node_feat=34,
        hidden=args.hidden,
        num_tasks=12,
        dropout=args.dropout,
    ).to(device)
    print(f"[train] Model parameters: {model.count_parameters():,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay, 
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("Tox21-GCN-Toxicity")

    with mlflow.start_run(run_name=f"gcn_lr{args.lr}_h{args.hidden}_drop{args.dropout}") as run:
        print(f"[train] MLflow run ID: {run.info.run_id}")
        mlflow.log_params(vars(args))

        best_val_auroc = 0.0
        patience_counter = 0
        best_checkpoint = CHECKPOINT_DIR / "best_gcn.pt"

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, pos_weight
            )
            val_loss, val_metrics, val_auroc = evaluate(model, val_loader, device)
            scheduler.step(val_auroc)

            epoch_time = time.time() - t0

            mlflow.log_metrics({
                "train_loss":  train_loss,
                "val_loss":    val_loss,
                "val_auroc":   val_auroc,
            }, step=epoch)

            print(
                f"Epoch [{epoch:03d}/{args.epochs}] "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_auroc={val_auroc:.4f}  "
                f"time={epoch_time:.1f}s"
            )

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_counter = 0
                torch.save({
                    "epoch":      epoch,
                    "state_dict": model.state_dict(),
                    "val_auroc":  val_auroc,
                    "val_loss":   val_loss,
                    "args":       vars(args),
                }, best_checkpoint)
                print(f"  ✓ New best checkpoint saved (val_auroc={val_auroc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"[train] Early stopping at epoch {epoch} "
                          f"(patience={args.patience}).")
                    break

        print(f"\n[train] Best val AUROC: {best_val_auroc:.4f}")
        print(f"[train] Best checkpoint: {best_checkpoint}")

        ckpt = torch.load(best_checkpoint, map_location="cpu")
        print(
            f"\nFINAL VAL LOG | Epoch {ckpt['epoch']:03d} | "
            f"val_loss={ckpt['val_loss']:.4f} | "
            f"val_auroc={ckpt['val_auroc']:.4f} | "
            f"checkpoint=best_gcn.pt"
        )

        mlflow.log_artifact(str(best_checkpoint))
        mlflow.log_metric("best_val_auroc", best_val_auroc)

        # Print per-task breakdown for the best epoch
        print("\n[train] Final per-task validation metrics:")
        print_metrics_table(val_metrics)


if __name__ == "__main__":
    main()
