import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

st.set_page_config(page_title="Credit Default Predictor", page_icon="💳", layout="wide")

TARGET = "default"
MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


def evaluate(model, X, y):
    """Predict with a saved pipeline (no retraining) and compute the 6 metrics."""
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_score),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return y_pred, metrics


# ------------------------------------------------------------------ header
st.title("💳 Credit Card Default Risk: Model Explorer")
st.caption(
    "Upload test data, pick a classifier and see how it performs. "
    "Models are pre-trained; nothing is retrained here."
)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
    st.markdown("---")
    st.markdown(
        "**Expected CSV:** the 23 feature columns used in training plus the "
        f"target column `{TARGET}` (0 = no default, 1 = default)."
    )
    st.markdown("Use the `test_data.csv` file from the GitHub repo.")

if uploaded is None:
    st.info("👈 Upload a test CSV in the sidebar to get started.")
    st.stop()

# ------------------------------------------------------------------ data
df = pd.read_csv(uploaded)

if TARGET not in df.columns:
    st.error(f"The uploaded file must contain the target column `{TARGET}`.")
    st.stop()

X = df.drop(columns=[TARGET])
y = df[TARGET]

model = load_model(model_name)

# Check that all training features are present, and use the training order
expected = list(getattr(model, "feature_names_in_", X.columns))
missing = [c for c in expected if c not in X.columns]
if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()
X = X[expected]

with st.expander(f"Preview of uploaded data ({df.shape[0]:,} rows, {df.shape[1]} columns)"):
    st.dataframe(df.head(20), use_container_width=True)

# ------------------------------------------------------------------ results
y_pred, metrics = evaluate(model, X, y)

st.subheader(f"Results for: {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)

with left:
    st.markdown("#### Confusion matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples", cbar=False, ax=ax,
        xticklabels=["No default", "Default"],
        yticklabels=["No default", "Default"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.markdown("#### Classification report")
    report = classification_report(
        y, y_pred, target_names=["No default", "Default"], output_dict=True
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# ------------------------------------------------------------------ comparison
st.markdown("---")
st.subheader("All models on this uploaded data")
rows = []
for name in MODEL_FILES:
    _, m = evaluate(load_model(name), X, y)
    rows.append({"ML Model Name": name, **m})
comparison = pd.DataFrame(rows).set_index("ML Model Name").round(4)
st.dataframe(comparison, use_container_width=True)
st.caption(f"Best MCC on this data: **{comparison['MCC'].idxmax()}**")
