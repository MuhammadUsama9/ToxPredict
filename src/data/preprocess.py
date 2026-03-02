"""
src/data/preprocess.py
-----------------------
Download the Tox21 dataset via DeepChem, clean it, and produce
stratified scaffold-split CSV files ready for modelling.

Key steps (the ones that required domain decisions):
  1. pIC50 not needed here — labels are binary (active/inactive).
  2. Missing labels kept as NaN → masked in loss function.
  3. Scaffold split avoids chemical-similarity leakage between splits.
  4. Class-imbalance measured and logged so downstream can weight loss.
"""

import os
import random
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

# Seed all randomness before importing deepchem / rdkit
random.seed(42)
np.random.seed(42)

import deepchem as dc
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def download_tox21() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load Tox21 via DeepChem MoleculeNet with a scaffold splitter.
    Returns (train_df, val_df, test_df) as pandas DataFrames.
    """
    featurizer = dc.feat.RDKitDescriptors()          # placeholder; we rebuild later
    tasks, datasets, _ = dc.molnet.load_tox21(
        featurizer="Raw",
        splitter="scaffold",
        transformers=[],
    )
    train_ds, val_ds, test_ds = datasets
    return (
        _dataset_to_df(train_ds, tasks),
        _dataset_to_df(val_ds,   tasks),
        _dataset_to_df(test_ds,  tasks),
    )


def _dataset_to_df(ds: dc.data.Dataset, tasks: list) -> pd.DataFrame:
    """Convert a DeepChem Dataset to a pandas DataFrame with SMILES + labels."""
    smiles = ds.ids                  # RAW featurizer stores SMILES in .ids
    labels = ds.y                    # shape (N, 12)
    df = pd.DataFrame(labels, columns=tasks)
    df.insert(0, "smiles", smiles)
    # DeepChem encodes missing labels as 0 in y but uses w=0 weights;
    # we recover missingness from the weight matrix.
    w = ds.w                         # shape (N, 12); 0 means missing
    for i, task in enumerate(tasks):
        df.loc[w[:, i] == 0, task] = np.nan
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning steps:
      1. Remove rows with invalid SMILES (RDKit cannot parse → None).
      2. Remove duplicate SMILES (keep first occurrence).
      3. Drop rows where ALL 12 labels are NaN (no annotations).
    """
    # Step 1: Validate SMILES
    valid_mask = df["smiles"].apply(
        lambda s: Chem.MolFromSmiles(s) is not None
    )
    n_invalid = (~valid_mask).sum()
    if n_invalid:
        print(f"[clean] Dropped {n_invalid} rows with invalid SMILES.")
    df = df[valid_mask].copy()

    # Step 2: Remove duplicate SMILES
    n_before = len(df)
    df = df.drop_duplicates(subset="smiles", keep="first")
    print(f"[clean] Removed {n_before - len(df)} duplicate SMILES.")

    # Step 3: Drop fully un-annotated rows
    label_cols = [c for c in df.columns if c != "smiles"]
    all_nan = df[label_cols].isna().all(axis=1)
    df = df[~all_nan].reset_index(drop=True)
    print(f"[clean] Final dataset size: {len(df)} compounds.")
    return df


def measure_class_imbalance(df: pd.DataFrame) -> None:
    """Log positive/negative ratio per task to quantify class imbalance."""
    print("\n[imbalance] Positive class ratio per task:")
    label_cols = [c for c in df.columns if c != "smiles"]
    for task in label_cols:
        valid = df[task].dropna()
        pos_ratio = valid.mean()
        print(f"  {task:<18}  pos_ratio={pos_ratio:.3f}  "
              f"(n={len(valid)}, pos={int(valid.sum())})")


def save_splits(train: pd.DataFrame, val: pd.DataFrame,
                test: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir  / "val.csv",   index=False)
    test.to_csv(out_dir / "test.csv",  index=False)
    print(f"\n[preprocess] Saved splits to {out_dir}/")
    print(f"  train={len(train)}, val={len(val)}, test={len(test)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Tox21 dataset")
    parser.add_argument("--out-dir", type=str, default="data/processed",
                        help="Directory to save processed CSV splits")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    required_files = ["train.csv", "val.csv", "test.csv"]
    if all((out_path / f).exists() for f in required_files):
        print(f"[preprocess] Processed data found in {out_path}. Skipping download.")
        return

    print("[preprocess] Downloading Tox21 via DeepChem ...")
    train_df, val_df, test_df = download_tox21()

    print("[preprocess] Cleaning training set ...")
    train_df = clean_df(train_df)
    val_df   = clean_df(val_df)
    test_df  = clean_df(test_df)

    measure_class_imbalance(train_df)
    save_splits(train_df, val_df, test_df, out_path)


if __name__ == "__main__":
    main()
