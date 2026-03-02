"""
tests/test_dataset.py
---------------------
Unit tests for MoleculeGraphDataset and MorganFingerprintDataset.

Checks:
  1. smiles_to_pyg returns correct node/edge feature shapes.
  2. Atom features contain no NaN values.
  3. MorganFingerprintDataset returns 2048-bit fingerprints.
  4. Bond features have the expected dimensionality.

Run with:
  pytest tests/test_dataset.py -v
"""

import numpy as np
import pytest
import torch

from src.data.dataset import (
    smiles_to_pyg,
    atom_features,
    bond_features,
    MorganFingerprintDataset,
)
from rdkit import Chem


# ── Test molecules ────────────────────────────────────────────────────────────
VALID_SMILES   = "CC(=O)Oc1ccccc1C(=O)O"    # Aspirin
INVALID_SMILES = "not_a_smiles_!!"
SINGLE_ATOM    = "[Na+]"


class TestAtomFeatures:
    def test_feature_length(self):
        mol  = Chem.MolFromSmiles(VALID_SMILES)
        atom = mol.GetAtomWithIdx(0)
        feat = atom_features(atom)
        # 19 (type) + 6 (degree) + 5 (hybrid) + 1 (arom) + 1 (charge) + 1 (Hs) + 1 (ring) = 34
        assert len(feat) == 34, f"Expected 34 atom features, got {len(feat)}"

    def test_no_nan(self):
        mol  = Chem.MolFromSmiles(VALID_SMILES)
        for atom in mol.GetAtoms():
            feat = atom_features(atom)
            assert not any(np.isnan(feat)), f"NaN found in atom features for atom {atom.GetSymbol()}"

    def test_one_hot_sums(self):
        """Each one-hot block should sum to exactly 1."""
        mol  = Chem.MolFromSmiles(VALID_SMILES)
        atom = mol.GetAtomWithIdx(0)
        feat = atom_features(atom)
        # Atom type one-hot: positions 0–18
        assert sum(feat[:19]) == 1, "Atom type one-hot block should sum to 1"


class TestBondFeatures:
    def test_feature_length(self):
        mol  = Chem.MolFromSmiles(VALID_SMILES)
        bond = mol.GetBondWithIdx(0)
        feat = bond_features(bond)
        # 4 (bond type) + 1 (ring) = 5
        assert len(feat) == 5, f"Expected 5 bond features, got {len(feat)}"


class TestSmilesToPyG:
    def test_valid_smiles_node_shape(self):
        graph = smiles_to_pyg(VALID_SMILES)
        assert graph is not None
        mol   = Chem.MolFromSmiles(VALID_SMILES)
        n_atoms = mol.GetNumAtoms()
        assert graph.x.shape == (n_atoms, 34), (
            f"Expected node matrix ({n_atoms}, 34), got {graph.x.shape}"
        )

    def test_valid_smiles_edge_shape(self):
        graph = smiles_to_pyg(VALID_SMILES)
        mol   = Chem.MolFromSmiles(VALID_SMILES)
        n_bonds = mol.GetNumBonds()
        # Bidirectional: 2 × n_bonds edges
        assert graph.edge_index.shape[1] == 2 * n_bonds, (
            f"Expected {2 * n_bonds} edges, got {graph.edge_index.shape[1]}"
        )

    def test_invalid_smiles_returns_none(self):
        graph = smiles_to_pyg(INVALID_SMILES)
        assert graph is None, "Invalid SMILES should return None"

    def test_single_atom_molecule(self):
        graph = smiles_to_pyg(SINGLE_ATOM)
        assert graph is not None
        assert graph.x.shape[0] == 1         # one atom
        assert graph.edge_index.shape[1] == 0  # no bonds

    def test_labels_shape(self):
        labels = np.array([1, 0, np.nan, 0, 1, 0, 0, 1, 0, 0, 1, 0])
        graph  = smiles_to_pyg(VALID_SMILES, labels=labels)
        assert graph is not None
        assert graph.y.shape == (1, 12), f"Expected y shape (1,12), got {graph.y.shape}"

    def test_node_features_no_nan(self):
        graph = smiles_to_pyg(VALID_SMILES)
        assert not torch.isnan(graph.x).any(), "Node feature matrix contains NaN"


class TestMorganFingerprintDataset:
    """Tests require data/processed/train.csv — skip if not available."""

    @pytest.fixture(autouse=True)
    def check_csv(self, tmp_path):
        """Create a tiny CSV for testing without needing the real dataset."""
        import pandas as pd
        csv = tmp_path / "tiny.csv"
        data = {
            "smiles": [VALID_SMILES, "c1ccccc1", "[Na+]"],
            "NR-AR": [1, 0, np.nan],
        }
        # Pad missing task columns
        for t in ["NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
                  "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]:
            data[t] = [np.nan, np.nan, np.nan]
        pd.DataFrame(data).to_csv(csv, index=False)
        self.csv_path = str(csv)

    def test_fingerprint_shape(self):
        ds = MorganFingerprintDataset(self.csv_path, radius=2, n_bits=2048)
        fp, _ = ds[0]
        assert fp.shape == (2048,), f"Expected (2048,), got {fp.shape}"

    def test_fingerprint_binary(self):
        ds = MorganFingerprintDataset(self.csv_path)
        fp, _ = ds[0]
        assert set(fp.tolist()).issubset({0, 1}), "Fingerprint bits should be 0 or 1"

    def test_dataset_length(self):
        ds = MorganFingerprintDataset(self.csv_path)
        # All 3 SMILES are valid, so len should be 3
        assert len(ds) == 3
