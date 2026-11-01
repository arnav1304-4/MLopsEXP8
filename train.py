"""Train a RandomForest classifier on the Iris dataset and save the model artifacts."""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "iris_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
MODEL_VERSION = "1.0.0"


def train_model(random_state: int = 42):
    """Train the model and return (model, metrics)."""
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=random_state, stratify=iris.target
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=random_state)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = {
        "model_version": MODEL_VERSION,
        "algorithm": "RandomForestClassifier",
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "feature_names": list(iris.feature_names),
        "target_names": list(iris.target_names),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = classification_report(y_test, y_pred, target_names=iris.target_names)
    return model, metrics, report


def save_artifacts(model, metrics):
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    model, metrics, report = train_model()
    save_artifacts(model, metrics)
    print("Training complete")
    print(f"Accuracy : {metrics['accuracy']}")
    print(f"F1 macro : {metrics['f1_macro']}")
    print(report)
    print(f"Model saved to   {MODEL_PATH.relative_to(Path(__file__).parent)}")
    print(f"Metrics saved to {METRICS_PATH.relative_to(Path(__file__).parent)}")
