import sys
from unittest.mock import MagicMock

# Mock ML dependencies to avoid installing them for simple API tests
sys.modules['torch'] = MagicMock()
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['torch_geometric'] = MagicMock()
sys.modules['torch_geometric.data'] = MagicMock()

import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data

def test_properties_valid_smiles():
    response = client.post("/properties", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    assert response.status_code == 200
    data = response.json()
    assert data["smiles"] == "CC(=O)Oc1ccccc1C(=O)O"
    assert "mol_weight" in data
    assert "log_p" in data
    assert "num_h_donors" in data
    assert "num_h_acceptors" in data
    assert abs(data["mol_weight"] - 180.16) < 1.0  # Aspirin MolWt ~ 180.16

def test_properties_invalid_smiles():
    response = client.post("/properties", json={"smiles": "invalid_smiles"})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
