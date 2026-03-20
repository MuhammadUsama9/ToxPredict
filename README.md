# ToxPredict: QSAR Toxicity Prediction

![ToxPredict Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-teal.svg)

ToxPredict is a comprehensive Machine Learning project designed for **Quantitative Structure-Activity Relationship (QSAR)** toxicity prediction. By learning molecular graph representations through state-of-the-art **Graph Convolutional Networks (GCNs)**, ToxPredict accurately predicts the toxicity probabilities of chemical compounds across **12 distinct biological assay endpoints**.

The project encompasses a complete ML lifecycle: from RDKit-based molecular featurisation and handling imbalanced multi-label data (Tox21 dataset), to model training, experiment tracking with MLflow, and scalable deployment using FastAPI and Docker.

---

## 🤖 Why use AI for Toxicity Prediction?

While it might seem simpler to maintain a database of known toxic chemicals for lookup, that approach has significant limitations. Here is why an AI-driven approach (like ToxGCN) is essential:

- **The "Unknown" Chemical Problem:** Millions of novel compounds are designed every year in drug discovery and materials science. A simple internet lookup only works for chemicals that have already been discovered and tested. AI allows us to predict the toxicity of a *new* chemical before it even exists in the real world.
- **Speed and Cost Efficiency:** Traditional lab testing (in vitro/in vivo) is slow and extremely expensive. An AI model can perform high-throughput virtual screening of thousands of molecular structures in milliseconds.
- **Learning Chemical Rules:** Instead of memorizing which chemicals are toxic, the Graph Convolutional Network learns the underlying structural patterns and rules (e.g., specific ring structures attached to certain atoms). This allows it to accurately generalize and predict toxicity for entirely unseen molecules.

---

## 🌟 Key Features

- **Graph Neural Network (GCN) Architecture:** Utilizes continuous molecular graph structures (where atoms are nodes and bonds are edges) rather than static fingerprints. 
- **12 Tox21 Endpoints:** Predicts pathways such as Nuclear Receptors (NR-AR, NR-ER) and Stress Response (SR-p53, SR-ARE).
- **Masked BCE Loss:** Gracefully handles ~20% missing labels in the dataset by masking unlabelled entries in the loss calculation.
- **Interactive UI Dashboard:** A modern frontend served via Nginx displaying radar charts, breakdown gauges, and an overall risk score.
- **MLOps Integrations:** Track model parameters, metrics (AUROC, Loss), and checkpoints effortlessly using MLflow.
- **Containerized Stack:** A docker-compose setup spanning FastAPI (Inference), Nginx (Frontend UI), and MLflow (Tracking Server).

---

## 🏗️ Architecture & Technical Details

### 1. Molecular Featurisation & Datasets
- **Data Source**: Tox21 dataset with 7,831 compounds.
- **Featurisation**: Handled by **RDKit** passing features to PyTorch Geometric. Each atom receives a **34-dimensional feature vector** (encoding atom type, hybridization, degree, aromaticity, formal charge, etc.).
- **Data Splitting**: Scaffold splitting (80/10/10) to simulate challenging out-of-distribution real-world chemical tests.

### 2. Deep Learning Model (`ToxGCN`)
- **Layers**: 3 `GCNConv` layers (hidden=128) on the molecular graph.
- **Pooling & Head**: Global mean pooling is used to aggregate node embeddings. It is followed by a 2-layer MLP head outputting 12 logits.
- **Performance**: Capable of achieving a Mean Validation **AUROC of ~0.831**.

---

## 🚀 Getting Started

Ensure you have [Docker](https://www.docker.com/) and `docker-compose` installed.

### Running with Docker Compose (Recommended)

To launch the complete application stack (API, Frontend, MLflow, and the Trainer):

```bash
docker-compose up --build
```
This automatically boots up:
1. **Frontend UI** on `http://localhost:3000`
2. **Inference API** on `http://localhost:8000`
3. **MLflow UI** on `http://localhost:5000`

### Running with Kubernetes

For production readiness and higher scalability, Kubernetes is supported. Manifests are located in the `kubernetes_manifests/` directory.

Before deploying, ensure you modify `api.yaml` and `frontend.yaml` to point the `hostPath` volumes correctly to your local project directory.

To deploy all services onto a local or cloud cluster:

```bash
# 1. First, deploy MLflow (uses Persistent Volume Claim)
kubectl apply -f kubernetes_manifests/mlflow.yaml

# 2. Deploy the Inference API
kubectl apply -f kubernetes_manifests/api.yaml

# 3. Deploy the Dashboard UI
kubectl apply -f kubernetes_manifests/frontend.yaml
```

Check the status with:
```bash
kubectl get pods
```

*Note: You may still need to use docker-compose up trainer or run train.py locally to generate the initial checkpoint.*

### Running Locally (Conda)

If you prefer to run and develop locally:

```bash
# 1. Create and activate conda environment
conda env create -f environment.yml
conda activate qsar-tox21

# 2. Train the model manually
python train.py --lr 1e-3 --epochs 50 --batch-size 64 --hidden 128 --dropout 0.3 --seed 42 --mlflow-uri ./mlruns

# 3. Start the API Server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📊 Training & Evaluation

The training script automatically computes per-task positive class weights to mitigate class imbalances effectively. It trains standard models using Masked Binary Cross-Entropy with L2 weight decay. 

Early stopping is implemented based on the Mean Validation AUROC metric. The best checkpoint is saved automatically to `checkpoints/best_gcn.pt` at the end of training.

*Sample training command:*
```bash
python train.py --lr 1e-3 --epochs 50 --batch-size 64 --hidden 128 --dropout 0.3 --seed 42 --mlflow-uri ./mlruns
```

---

## 🧪 Quick Test Examples

You can test the system through the dashboard at `http://localhost:3000` with some of these SMILES compounds:

- **Aspirin / Acetylsalicylic acid:** `CC(=O)Oc1ccccc1C(=O)O`
- **Benzene (Carcinogen):** `c1ccccc1`
- **Acetaminophen / Paracetamol:** `CC(=O)Nc1ccc(O)cc1`

Once requested, the system maps the probability of response over all biological assays predicting safe, moderate, or high risk levels!

