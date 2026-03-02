"""
src/utils/visualize.py
----------------------
Utility for visualising molecular graphs converted from SMILES.
Uses RDKit for chemical plotting and Matplotlib for layout.
"""

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from src.data.dataset import smiles_to_pyg
from src.config import TOX21_TASKS


def plot_molecule_with_labels(smiles: str, title: str = "Molecular Structure"):
    """Plot the RDKit molecule structure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error: Invalid SMILES {smiles}")
        return

    img = Draw.MolToImage(mol, size=(400, 400), legend=smiles)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()


def debug_print_graph(smiles: str):
    """Print the PyG graph details for debugging featurisation."""
    graph = smiles_to_pyg(smiles)
    if graph is None:
        print("Failed to featurise molecule.")
        return

    print(f"\nGraph Details for: {smiles}")
    print(f"  Nodes (atoms): {graph.x.shape[0]}")
    print(f"  Edges (bonds): {graph.edge_index.shape[1]}")
    print(f"  Node features: {graph.x.shape[1]} dims")
    print(f"  Edge features: {graph.edge_attr.shape[1]} dims")


if __name__ == "__main__":
    # Example: Aspirin
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    debug_print_graph(aspirin)
    # Note: plot_molecule_with_labels requires a display/X11
