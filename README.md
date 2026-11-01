# Iris Species Classifier: End-to-End MLOps (Experiment 8)

**Arnav Mhatre · E002 · BTI Data Science Sem 11 · Batch E1**

A small ML project taken through the complete MLOps lifecycle:

```
Git  →  GitHub Actions (CI/CD)  →  Docker image  →  Kubernetes  →  Prometheus + Grafana  →  Streamlit
```

| Layer | What is used |
|---|---|
| Model | scikit-learn `RandomForestClassifier` on the Iris dataset (test accuracy ≈ 96.7%) |
| App | Streamlit UI (`app.py`) with live prediction, class probabilities and simulated traffic |
| Metrics | `prometheus_client` endpoint on port **8000** (`/metrics`) |
| CI/CD | GitHub Actions: lint → train → test → Docker build → smoke test → push to GHCR |
| Container | `python:3.12-slim` image, model trained during the build |
| Orchestration | Kubernetes (Docker Desktop): 2 app replicas + Prometheus + Grafana |
| Monitoring | Prometheus scrapes every pod; Grafana dashboard is provisioned automatically |

## Project structure

```
app.py              Streamlit front end
train.py            Trains the model, writes model/iris_model.joblib + model/metrics.json
monitoring.py       Prometheus metric definitions and /metrics server
run.py              Container entry point (starts /metrics, then Streamlit)
tests/              pytest unit tests
Dockerfile          Container image
k8s/                Namespace, app, Prometheus and Grafana manifests
.github/workflows/  GitHub Actions pipeline
```

## Run locally

```bash
pip install -r requirements.txt
python train.py
pytest -v
python run.py          # UI on http://localhost:8501, metrics on http://localhost:8000/metrics
```

## Docker

```bash
docker build -t iris-mlops:1.0 .
docker run -d --name iris-app -p 8501:8501 -p 8000:8000 iris-mlops:1.0
```

## Kubernetes + monitoring

```bash
kubectl apply -f k8s/
kubectl get all -n mlops-exp8
```

| Service | URL |
|---|---|
| Streamlit app | http://localhost:30501 |
| Prometheus | http://localhost:30090 |
| Grafana (dashboard "Iris MLOps - Model Monitoring") | http://localhost:30300 |

Exported metrics: `iris_predictions_total{species,source}`, `iris_prediction_latency_seconds`,
`iris_prediction_confidence`, `iris_input_feature_value{feature}`, `iris_app_sessions_total`,
`iris_model_accuracy`, `iris_model_f1_macro`, `iris_model_info`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to <https://share.streamlit.io> → **Create app** → pick this repo, branch `main`, main file `app.py`.
3. The app trains the model on first start if `model/` is not present.
