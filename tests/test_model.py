"""
tests/test_model.py
--------------------
Unit tests for ToxGCN and masked_bce_loss.

Checks:
  1. Forward pass produces correct output shape (B, 12).
  2. Loss is finite (no NaN/Inf) under randomised input.
  3. Masked BCE loss ignores NaN labels correctly.
  4. Gradient flows through all layers (no dead parameters).
  5. Checkpoint save / load cycle reproduces identical outputs.

Run with:
  pytest tests/test_model.py -v
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
from torch_geometric.data import Data, Batch

from src.models.gcn_model import ToxGCN, masked_bce_loss


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_random_graph(n_atoms: int = 12, n_bonds: int = 10,
                       num_tasks: int = 12, node_feat: int = 34) -> Data:
    """Create a synthetic PyG Data object for testing."""
    x          = torch.randn(n_atoms, node_feat)
    src        = torch.randint(0, n_atoms, (n_bonds,))
    dst        = torch.randint(0, n_atoms, (n_bonds,))
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ])
    y = torch.rand(1, num_tasks)          # random probabilities as labels
    # Introduce NaN for some tasks (simulates missing labels)
    y[0, [2, 5, 9]] = float("nan")
    return Data(x=x, edge_index=edge_index, y=y, smiles="CC")


def _make_batch(batch_size: int = 4) -> Batch:
    graphs = [_make_random_graph() for _ in range(batch_size)]
    return Batch.from_data_list(graphs)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestToxGCN:

    def setup_method(self):
        self.model = ToxGCN(node_feat=34, hidden=64, num_tasks=12, dropout=0.0)
        self.model.eval()

    def test_output_shape(self):
        batch  = _make_batch(batch_size=4)
        with torch.no_grad():
            logits = self.model(batch)
        assert logits.shape == (4, 12), (
            f"Expected output shape (4, 12), got {logits.shape}"
        )

    def test_single_molecule(self):
        batch  = _make_batch(batch_size=1)
        with torch.no_grad():
            logits = self.model(batch)
        assert logits.shape == (1, 12)

    def test_output_finite(self):
        batch  = _make_batch(batch_size=8)
        with torch.no_grad():
            logits = self.model(batch)
        assert torch.isfinite(logits).all(), "Model output contains NaN or Inf"

    def test_gradient_flows(self):
        """All learnable parameters should receive gradients."""
        self.model.train()
        batch  = _make_batch(batch_size=4)
        logits = self.model(batch)
        labels = batch.y.squeeze(1) if batch.y.dim() > 2 else batch.y.view(-1, 12)
        loss   = masked_bce_loss(logits, labels)
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient for {name}"
                )

    def test_parameter_count(self):
        n_params = self.model.count_parameters()
        # Sanity check: should be in the tens-of-thousands range
        assert n_params > 1_000, f"Suspiciously few parameters: {n_params}"
        assert n_params < 10_000_000, f"Suspiciously many parameters: {n_params}"

    def test_checkpoint_roundtrip(self):
        """Save and reload checkpoint; output must be identical."""
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            logits_before = self.model(batch).clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt.pt"
            torch.save({
                "state_dict": self.model.state_dict(),
                "epoch":      1,
                "val_auroc":  0.75,
                "args":       {"hidden": 64, "dropout": 0.0},
            }, ckpt_path)

            model2 = ToxGCN(node_feat=34, hidden=64, num_tasks=12, dropout=0.0)
            ckpt   = torch.load(ckpt_path, map_location="cpu")
            model2.load_state_dict(ckpt["state_dict"])
            model2.eval()

            with torch.no_grad():
                logits_after = model2(batch).clone()

        assert torch.allclose(logits_before, logits_after, atol=1e-6), (
            "Checkpoint roundtrip produced different outputs"
        )


class TestMaskedBCELoss:

    def test_loss_is_finite(self):
        logits  = torch.randn(8, 12)
        targets = torch.randint(0, 2, (8, 12)).float()
        loss    = masked_bce_loss(logits, targets)
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    def test_nan_labels_ignored(self):
        """Loss should be identical whether NaN entries exist or not."""
        logits  = torch.randn(4, 12)
        targets = torch.randint(0, 2, (4, 12)).float()

        # Version with NaN in task 3
        targets_nan       = targets.clone()
        targets_nan[:, 3] = float("nan")
        loss_nan          = masked_bce_loss(logits, targets_nan)

        # Version without NaN (manually exclude column 3)
        mask     = torch.ones(4, 12, dtype=torch.bool)
        mask[:, 3] = False
        logits_m  = logits[mask]
        targets_m = targets[mask].float()
        loss_full = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_m, targets_m, reduction="mean"
        )

        assert torch.isclose(loss_nan, loss_full, atol=1e-5), (
            f"Masked loss ({loss_nan:.6f}) differs from expected ({loss_full:.6f})"
        )

    def test_all_nan_raises_or_nan(self):
        """If all labels are NaN, loss should be 0 (empty mean) or handled gracefully."""
        logits  = torch.randn(4, 12)
        targets = torch.full((4, 12), float("nan"))
        # Should not raise; behaviour is nan or 0 due to empty mean
        try:
            loss = masked_bce_loss(logits, targets)
            assert torch.isnan(loss) or loss == 0.0
        except Exception as e:
            pytest.fail(f"masked_bce_loss raised unexpectedly: {e}")
