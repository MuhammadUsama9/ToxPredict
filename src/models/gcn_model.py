"""
src/models/gcn_model.py
------------------------
Graph Convolutional Network (GCN) for multi-label toxicity classification.

Architecture:
  3 × GCNConv layers  →  global mean pooling  →  2-layer MLP head
  Output: raw logits of shape (batch_size, 12) — one per Tox21 assay.
  Sigmoid is applied externally (in the loss and at inference time).

Why GCN over alternatives?
  - Random Forest (chosen baseline): strong on tabular fingerprints but ignores
    3-D graph topology and cannot generalise to unseen scaffolds.
  - SVM with RBF kernel: scales poorly to 2048-dim fingerprints and requires
    careful one-vs-rest wrapping for multi-label tasks.
  - GCN directly processes the molecular graph, making it scaffold-agnostic
    and naturally end-to-end differentiable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class ToxGCN(nn.Module):
    """
    Three-layer Graph Convolutional Network for Tox21 multi-label prediction.

    Args:
        node_feat  : input node feature dimension (34 from atom_features)
        hidden     : hidden channel width (default 128)
        num_tasks  : number of toxicity endpoints to predict (12 for Tox21)
        dropout    : dropout probability applied before the MLP head
    """

    def __init__(
        self,
        node_feat: int = 34,
        hidden: int = 128,
        num_tasks: int = 12,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # ── Graph convolution layers ─────────────────────────────────────────
        # Each GCNConv applies: H' = σ( D̂^{-1/2} Â D̂^{-1/2} H W )
        # where Â = A + I (self-loops added automatically by add_self_loops=True)
        self.conv1 = GCNConv(node_feat, hidden)
        self.conv2 = GCNConv(hidden,    hidden)
        self.conv3 = GCNConv(hidden,    hidden // 2)

        # ── MLP classification head ──────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_tasks),
            # No sigmoid here — we use BCEWithLogitsLoss for numerical stability
        )

        # ── Weight initialisation ────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass through the GCN.

        Args:
            data: PyG Batch object with attributes:
                  .x           — node features (N_total, node_feat)
                  .edge_index  — COO edge index (2, E_total)
                  .batch       — node-to-graph assignment (N_total,)

        Returns:
            logits: (batch_size, num_tasks) raw scores (before sigmoid)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph convolutions with ReLU activations
        x = F.relu(self.conv1(x, edge_index))   # (N, hidden)
        x = F.relu(self.conv2(x, edge_index))   # (N, hidden)
        x = F.relu(self.conv3(x, edge_index))   # (N, hidden//2)

        # Global graph-level readout: mean of all node embeddings per molecule
        x = global_mean_pool(x, batch)          # (B, hidden//2)

        # MLP head → logits
        return self.head(x)                      # (B, num_tasks)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Binary cross-entropy loss with NaN masking.

    Loss function minimised:
        L = -1/|V| Σ_{(i,t) ∈ V} [ y_{it} log σ(ẑ_{it})
                                   + (1-y_{it}) log(1-σ(ẑ_{it})) ]

    where V = { (i,t) : y_{it} is not NaN } are valid label pairs,
    ẑ_{it} is the raw logit, and σ is the sigmoid function.
    Optional pos_weight upweights rare positive class per task.

    Args:
        logits     : (B, T) raw model outputs
        targets    : (B, T) ground-truth labels, NaN for missing
        pos_weight : (T,) per-task positive class weight tensor

    Returns:
        Scalar masked BCE loss.
    """
    mask = ~torch.isnan(targets)
    masked_logits  = logits[mask]
    masked_targets = targets[mask].float()

    if pos_weight is not None:
        # Expand pos_weight to match masked elements
        task_idx = mask.nonzero(as_tuple=True)[1]
        pw = pos_weight[task_idx]
        loss = F.binary_cross_entropy_with_logits(
            masked_logits, masked_targets, pos_weight=pw, reduction="mean"
        )
    else:
        loss = F.binary_cross_entropy_with_logits(
            masked_logits, masked_targets, reduction="mean"
        )
    return loss
