"""
src/data/featurizer.py
-----------------------
Utility functions for converting RDKit molecules into numerical feature vectors.
Extracted from dataset.py to allow usage in visualization and inference scripts.
"""

from typing import List
from rdkit import Chem

# ── Feature Vocabularies ──────────────────────────────────────────────────────

ATOM_TYPES = [
    "C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Si",
    "B", "Na", "K", "Ca", "Fe", "As", "Al", "Se", "other",
]

HYBRIDISATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def _one_hot(value, choices: list) -> List[int]:
    """Helper for one-hot encoding with a catch-all 'other' category."""
    enc = [int(value == c) for c in choices]
    if sum(enc) == 0:
        enc[-1] = 1   # catch-all "other"
    return enc


def atom_features(atom) -> List[float]:
    """
    34-dimensional atom feature vector:
      - 19 atom type (one-hot)
      - 6  degree (0-5)
      - 5  hybridisation type
      - 1  aromaticity flag
      - 1  formal charge
      - 1  number of Hs
      - 1  in-ring flag
    """
    feats: List[float] = []
    feats += _one_hot(atom.GetSymbol(), ATOM_TYPES)                 # 19
    feats += _one_hot(atom.GetDegree(), list(range(6)))              # 6
    feats += _one_hot(atom.GetHybridization(), HYBRIDISATION)        # 5
    feats.append(float(atom.GetIsAromatic()))                        # 1
    feats.append(float(atom.GetFormalCharge()))                      # 1
    feats.append(float(atom.GetTotalNumHs()))                        # 1
    feats.append(float(atom.IsInRing()))                             # 1
    return feats


def bond_features(bond) -> List[float]:
    """5-dimensional bond feature vector: [one-hot(4 types), in-ring(1)]."""
    feats: List[float] = []
    feats += _one_hot(bond.GetBondType(), BOND_TYPES)   # 4
    feats.append(float(bond.IsInRing()))                 # 1
    return feats
