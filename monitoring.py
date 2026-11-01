"""Prometheus metrics for the Iris Streamlit app.

Metrics are defined once at module level and served on a separate port
(default 8000) so that Prometheus can scrape /metrics.
"""
import json
import os
import threading
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))

PREDICTIONS = Counter(
    "iris_predictions_total", "Total predictions made by the model", ["species", "source"]
)
LATENCY = Histogram(
    "iris_prediction_latency_seconds",
    "Time taken to run a single prediction",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0),
)
CONFIDENCE = Histogram(
    "iris_prediction_confidence",
    "Probability of the predicted class",
    buckets=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
INPUT_FEATURE = Gauge(
    "iris_input_feature_value", "Last input value received for each feature (cm)", ["feature"]
)
SESSIONS = Counter("iris_app_sessions_total", "Browser sessions opened on the Streamlit app")
MODEL_ACCURACY = Gauge("iris_model_accuracy", "Accuracy of the deployed model on the test set")
MODEL_F1 = Gauge("iris_model_f1_macro", "Macro F1 score of the deployed model on the test set")
MODEL_INFO = Info("iris_model", "Deployed model metadata")

_server_lock = threading.Lock()
_server_started = False


def set_model_metrics(metrics: dict):
    MODEL_ACCURACY.set(metrics["accuracy"])
    MODEL_F1.set(metrics["f1_macro"])
    MODEL_INFO.info({"version": metrics["model_version"], "algorithm": metrics["algorithm"]})


def load_saved_model_metrics():
    path = Path(__file__).parent / "model" / "metrics.json"
    if path.exists():
        set_model_metrics(json.loads(path.read_text()))


def start_metrics_server():
    """Start the /metrics endpoint once per process. Safe to call on every rerun."""
    global _server_started
    with _server_lock:
        if _server_started:
            return
        try:
            start_http_server(METRICS_PORT)
        except OSError:
            # Port already in use (another process owns it) - keep the app running anyway
            pass
        _server_started = True
