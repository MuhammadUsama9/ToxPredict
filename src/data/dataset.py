"""
src/data/dataset.py
--------------------
Two dataset classes:
  1. MoleculeGraphDataset  — converts SMILES → PyTorch Geometric Data graphs
                             for use with the GCN model.
  2. MorganFingerprintDataset — converts SMILES → ECFP4 (2048-bit) numpy arrays
                                 for use with the Random Forest baseline.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import Dataset as TorchDataset
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm




from src.data.featurizer import atom_features, bond_features


def smiles_to_pyg(smiles: str,
                  labels: Optional[np.ndarray] = None) -> Optional[Data]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.
    Returns None if the molecule is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)  # (N_atoms, 34)

    # Edge features (bidirectional)
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs   += [bf, bf]

    if len(edge_indices) == 0:
        # Single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 5), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attrs,   dtype=torch.float)

    y = None
    if labels is not None:
        y = torch.tensor(labels, dtype=torch.float).unsqueeze(0)  # (1, 12)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                smiles=smiles)


# ── Graph Dataset ─────────────────────────────────────────────────────────────

class MoleculeGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset built from a processed CSV.

    Args:
        csv_path : path to train/val/test CSV (smiles + 12 label cols)
        transform : optional PyG transform
    """

    def __init__(self, csv_path: str, transform=None):
        super().__init__(root=None, transform=transform)
        self.csv_path = Path(csv_path)
        self._data_list = self._build()
        self.data, self.slices = self.collate(self._data_list)

    def _build(self) -> List[Data]:
        df = pd.read_csv(self.csv_path)
        label_cols = [c for c in df.columns if c != "smiles"]
        data_list: List[Data] = []

        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"Building graph dataset ({self.csv_path.name})"):
            smiles = row["smiles"]
            labels = row[label_cols].values.astype(float)  # may have NaN
            graph  = smiles_to_pyg(smiles, labels)
            if graph is not None:
                data_list.append(graph)

        print(f"[dataset] Built {len(data_list)} molecular graphs from {self.csv_path.name}.")
        return data_list

    def len(self) -> int:
        return super().len()

    def get(self, idx: int) -> Data:
        return super().get(idx)


# ── Morgan Fingerprint Dataset (RF baseline) ──────────────────────────────────

class MorganFingerprintDataset(TorchDataset):
    """
    Returns ECFP4 Morgan fingerprints (radius=2, 2048 bits) as numpy arrays.
    Used for the Random Forest baseline.

    Args:
        csv_path : preprocessed CSV with smiles + label columns
        radius   : Morgan radius (default 2 → ECFP4)
        n_bits   : fingerprint length (default 2048)
    """

    def __init__(self, csv_path: str, radius: int = 2, n_bits: int = 2048):
        self.df         = pd.read_csv(csv_path)
        self.label_cols = [c for c in self.df.columns if c != "smiles"]
        self.radius     = radius
        self.n_bits     = n_bits
        self.fps, self.labels = self._featurise()

    def _featurise(self) -> Tuple[np.ndarray, np.ndarray]:
        fps, labels = [], []
        for _, row in self.df.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=self.radius, nBits=self.n_bits
            )
            fps.append(np.array(fp))
            labels.append(row[self.label_cols].values.astype(float))
        return np.array(fps), np.array(labels)

    def __len__(self) -> int:
        return len(self.fps)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.fps[idx], self.labels[idx]
