import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prometheus_client import generate_latest  # noqa: E402

import monitoring  # noqa: E402
from train import train_model  # noqa: E402


def test_model_accuracy_above_threshold():
    _, metrics, _ = train_model()
    assert metrics["accuracy"] >= 0.9
    assert metrics["f1_macro"] >= 0.9


def test_known_samples_are_classified_correctly():
    model, metrics, _ = train_model()
    names = metrics["target_names"]
    assert names[model.predict([[5.1, 3.5, 1.4, 0.2]])[0]] == "setosa"
    assert names[model.predict([[6.0, 2.9, 4.5, 1.5]])[0]] == "versicolor"
    assert names[model.predict([[6.9, 3.1, 5.8, 2.3]])[0]] == "virginica"


def test_probabilities_sum_to_one():
    model, _, _ = train_model()
    proba = model.predict_proba([[5.8, 2.7, 4.1, 1.0]])[0]
    assert abs(proba.sum() - 1.0) < 1e-9


def test_prometheus_metrics_are_registered():
    _, metrics, _ = train_model()
    monitoring.set_model_metrics(metrics)
    monitoring.PREDICTIONS.labels(species="setosa", source="test").inc()
    output = generate_latest().decode()
    assert "iris_predictions_total" in output
    assert "iris_model_accuracy" in output
