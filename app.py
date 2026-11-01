"""Streamlit front end for the Iris species classifier (MLOps Experiment 8)."""
import json
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris

import monitoring
from train import METRICS_PATH, MODEL_PATH, save_artifacts, train_model

FEATURE_KEYS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

st.set_page_config(page_title="Iris Classifier | MLOps Exp 8", page_icon="🌸", layout="wide")


@st.cache_resource
def load_model():
    """Load the trained model, or train one if the artifacts are missing (e.g. on Streamlit Cloud)."""
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text())
    model, metrics, _ = train_model()
    save_artifacts(model, metrics)
    return model, metrics


@st.cache_data
def load_dataset():
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns=dict(zip(iris.feature_names, FEATURE_KEYS)))
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df.drop(columns="target")


def predict(model, target_names, features, source):
    start = time.perf_counter()
    proba = model.predict_proba([features])[0]
    monitoring.LATENCY.observe(time.perf_counter() - start)

    idx = int(np.argmax(proba))
    species = target_names[idx]
    monitoring.PREDICTIONS.labels(species=species, source=source).inc()
    monitoring.CONFIDENCE.observe(float(proba[idx]))
    for key, value in zip(FEATURE_KEYS, features):
        monitoring.INPUT_FEATURE.labels(feature=key).set(value)
    return species, proba


model, metrics = load_model()
target_names = metrics["target_names"]
data = load_dataset()

monitoring.start_metrics_server()
monitoring.set_model_metrics(metrics)
if "session_counted" not in st.session_state:
    monitoring.SESSIONS.inc()
    st.session_state.session_counted = True
    st.session_state.history = []

st.title("🌸 Iris Species Classifier")
st.caption(
    "MLOps Experiment 8: Git → GitHub Actions → Docker → Kubernetes → Prometheus & Grafana → Streamlit"
)

# ---- Sidebar: inputs ----
with st.sidebar:
    st.header("Flower measurements (cm)")
    with st.form("predict_form"):
        features = []
        for key in FEATURE_KEYS:
            col = data[key]
            features.append(
                st.slider(
                    key.replace("_", " ").title(),
                    min_value=float(col.min()),
                    max_value=float(col.max()),
                    value=float(round(col.mean(), 1)),
                    step=0.1,
                )
            )
        submitted = st.form_submit_button("Predict species", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Simulate traffic")
    n_requests = st.number_input("Number of requests", min_value=5, max_value=500, value=25, step=5)
    simulate = st.button("Send simulated requests", use_container_width=True)

# ---- Prediction ----
if submitted:
    species, proba = predict(model, target_names, features, source="user")
    st.session_state.history.insert(
        0, {**dict(zip(FEATURE_KEYS, features)), "prediction": species, "confidence": round(proba.max(), 3)}
    )
    st.session_state.last = (species, proba)

left, right = st.columns([1, 1.3])
with left:
    st.subheader("Prediction")
    if "last" in st.session_state:
        species, proba = st.session_state.last
        st.success(f"Predicted species: **Iris {species}**")
        st.metric("Confidence", f"{proba.max() * 100:.1f}%")
    else:
        st.info("Set the measurements in the sidebar and click **Predict species**.")

with right:
    st.subheader("Class probabilities")
    if "last" in st.session_state:
        proba_df = pd.DataFrame({"species": target_names, "probability": st.session_state.last[1]})
        st.bar_chart(proba_df, x="species", y="probability", height=260)
    else:
        st.empty()

if simulate:
    rng = np.random.default_rng()
    sample = data.sample(int(n_requests), replace=True)
    results = []
    for row in sample[FEATURE_KEYS].to_numpy():
        noisy = np.clip(row + rng.normal(0, 0.2, size=4), 0.1, None).round(1)
        results.append(predict(model, target_names, noisy.tolist(), source="simulated")[0])
    counts = pd.Series(results).value_counts().rename_axis("species").reset_index(name="count")
    st.toast(f"Sent {len(results)} simulated prediction requests")
    st.subheader("Simulated batch result")
    st.dataframe(counts, hide_index=True)

# ---- Model card ----
st.divider()
st.subheader("Model card")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Algorithm", "Random Forest")
c2.metric("Test accuracy", f"{metrics['accuracy'] * 100:.1f}%")
c3.metric("F1 (macro)", f"{metrics['f1_macro']:.3f}")
c4.metric("Model version", metrics["model_version"])

if st.session_state.history:
    st.subheader("Predictions in this session")
    st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, use_container_width=True)

with st.expander("Training data (Iris dataset, 150 samples)"):
    st.dataframe(data, hide_index=True, use_container_width=True)
    st.scatter_chart(data, x="petal_length", y="petal_width", color="species", height=300)

st.caption("Arnav Mhatre · E002 · BTI DS Sem 11 · Batch E1 · Prometheus metrics on port 8000 (/metrics)")
