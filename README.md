# NN API: Molecular Solubility Inference Service

A containerised inference service that predicts the aqueous solubility (logS) of molecules from SMILES strings. A PyTorch model is served with **Ray Serve** behind a **FastAPI** ingress, runs on a multi-node Ray cluster with replica autoscaling, and has a **Streamlit** frontend. Load testing uses **Locust**.

## Architecture

```
            ┌────────────────────┐
 browser ──▶│ frontend-app       │  Streamlit, :8501
            └─────────┬──────────┘
                      │ HTTP (backend-app:8000)
            ┌─────────▼──────────┐       ┌────────────────────┐
            │ backend-app        │──────▶│ backend-head       │
            │ Ray worker node    │ joins │ Ray head node      │
            │ Serve + FastAPI    │       │ GCS :8500          │
            │ SolubilityInference│       │ dashboard :8265    │
            │ replicas (auto)    │       │ client :10001      │
            └────────────────────┘       └────────────────────┘
```

- **backend-head**: Ray head node. It has `--num-cpus=0`, so it only coordinates and never runs replicas. It also hosts the Ray dashboard.
- **backend-app**: Ray worker node. It joins the cluster, then `serve deploy`s the application through the head's dashboard API.
- **frontend-app**: Streamlit UI. It starts only after both backend services pass their health checks.

Each service is built from the same `dockerfile`. A `DEPS_GROUP` build arg picks the uv dependency group (`prod-back` or `prod-front`), so the frontend image doesn't pull in torch or Ray.

## Features

- **Ray Serve deployment** with `num_replicas="auto"`: replicas scale with request load, 0.2 CPU per replica
- **FastAPI ingress** with Pydantic request/response schemas and auto-generated OpenAPI docs
- **TorchScript model**, loaded once per replica, CUDA if available, otherwise CPU
- **Single and batch prediction** endpoints; an invalid SMILES returns a per-molecule error instead of failing the whole request
- **Docker Compose orchestration** with health checks and ordered startup
- **Locust load tests** and an endpoint consistency test
- **CI**: GitHub Actions runs Ruff formatting on every push and PR and auto-commits the result
- **API docs** generated with pdoc3 into `html/`

## Project structure

```
.
├── app/
│   ├── main.py                  # Ray Serve deployment + FastAPI routes
│   └── frontend.py              # Streamlit UI
├── models/
│   └── molecule_solubility.py   # Dataset, model, training loop, TorchScript export
├── benchmarks/locust.py         # Locust load test
├── tests/single_batch.py        # Checks single and batch endpoints agree
├── html/                        # pdoc3-generated documentation
├── dockerfile                   # Shared image, dependency group chosen by build arg
├── docker-compose.yml           # Ray head, Ray worker/Serve app, Streamlit
├── .github/workflows/lint.yml   # Ruff CI
└── pyproject.toml / uv.lock     # uv dependency groups: prod-back, prod-front, dev
```

## Model

`models/molecule_solubility.py` trains a regressor on RDKit Morgan fingerprints.

- **Features:** 2048-bit Morgan fingerprints (radius 2). For salts and mixtures, the largest fragment is kept.
- **Network:** MLP, 2048 → 1024 → 512 → 256 → 1, with ReLU and 0.2 dropout
- **Training:** MSE loss, Adam (lr 1e-4), `ReduceLROnPlateau`, 80/20 split, early stopping (patience 10), validation R² tracked each epoch
- **Export:** TorchScript, `model_scripted.pt`

The dataset (`curated-solubility-dataset.csv`, with `SMILES` and `Solubility` columns, e.g. AqSolDB) and the trained weights (`*.pt`) are git-ignored. You need to produce them before running the service.

```bash
uv sync --group dev
cd models
# place curated-solubility-dataset.csv here
uv run python molecule_solubility.py   # writes models/model_scripted.pt
```

## Running

### Docker Compose (full stack)

Make sure `models/model_scripted.pt` exists first; it is copied into the image.

```bash
docker compose up --build
```

| Service           | URL                      |
|-------------------|--------------------------|
| Streamlit UI      | http://localhost:8501    |
| Ray dashboard     | http://localhost:8265    |

The Serve HTTP port (8000) is internal to the Compose network. To reach it from the host, uncomment the `ports` block on `backend-app`.

### Local (without Docker)

```bash
uv sync --group dev
uv run serve run app.main:SolubilityInference_app   # API on http://127.0.0.1:8000
```

Interactive API docs are at `http://127.0.0.1:8000/docs`.

## API

### `POST /predict`

```json
{ "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" }
```

```json
{ "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "solubility": -0.87, "error": null }
```

### `POST /predict_batch`

```json
{ "smiles_list": ["CC(=O)OC1=CC=CC=C1C(=O)O", "not-a-smiles"] }
```

```json
{
  "predictions": [
    { "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "solubility": -1.72, "error": null },
    { "smiles": "not-a-smiles", "solubility": null, "error": "Could not process SMILES string" }
  ]
}
```

The numbers above are illustrative. Real values depend on your trained weights.

## Testing and load testing

Both scripts target `http://127.0.0.1:8000`, so run the service locally with `serve run` or expose port 8000.

```bash
# Batch and single predictions must match for each molecule
uv run python tests/single_batch.py

# Load test; open http://localhost:8089 to set users and spawn rate
uv run locust -f benchmarks/locust.py --host http://127.0.0.1:8000
```

Watch replicas scale up under load in the Ray dashboard's **Serve** tab.

## Documentation

Regenerate the HTML API docs with:

```bash
uv run pdoc --html --force -o html .
```

## License

MIT
