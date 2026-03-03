# ToxPredict: QSAR Toxicity Prediction

ToxPredict is a machine-learning project that utilizes Graph Neural Networks (GNNs) for Quantitative Structure-Activity Relationship (QSAR) toxicity prediction. By learning molecular graph representations, ToxPredict predicts the toxicity of various chemical compounds.

## Project Overview

The aim of ToxPredict is to provide accurate toxicity predictions for molecules using advanced deep learning techniques, specifically Graph Convolutional Networks (GCNs) built with PyTorch Geometric. This project handles multi-label classification tasks commonly found in chemical datasets like Tox21.

## Features

- **Graph Neural Network Model**: Utilizes PyTorch and PyTorch Geometric to represent and learn from molecular structures.
- **RDKit Integration**: Uses RDKit for robust molecular processing and feature extraction.
- **FastAPI Backend**: Provides a scalable and fast web API for making predictions on new molecules.
- **Docker Compose Setup**: Easily reproducible environment using Docker containers for the API and frontend.
- **MLflow Tracking**: Integrated experiment tracking to monitor model training and performance.

## Prerequisites

To run this project locally, ensure you have the following installed:
- [Docker](https://www.docker.com/) and Docker Compose
- Or [Conda](https://docs.conda.io/en/latest/) (for local development)

## Getting Started

### Using Docker (Recommended)

The easiest way to get the project running is via Docker Compose:

```bash
docker-compose up --build
```

This will start the backend API and the frontend application.

### Local Installation (Conda)

If you prefer running without Docker, you can set up a Conda environment:

```bash
conda env create -f environment.yml
conda activate qsar-tox21
```

## Structure

- `src/`: Contains the source code including data processing, model definitions (e.g., `gcn_model.py`), and training scripts.
- `frontend/`: Contains the frontend web interface.
- `tests/`: Unit tests and pytest fixtures to ensure codebase reliability.
- `environment.yml`: Conda dependencies for easy local setup.
- `docker-compose.yml`: Configuration for running the full stack through Docker.

## License

This project is open-source and available for the general public.
