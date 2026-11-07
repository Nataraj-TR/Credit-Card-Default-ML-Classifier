import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    roc_curve,
)

st.set_page_config(page_title="Credit Default Predictor", page_icon="🏦", layout="wide")

TARGET = "default"
MODEL_DIR = "model"
DEMO_FILE = "test_data.csv"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
CLASS_NAMES = ["No default", "Default"]
ACCENT = "#0f766e"
PAGES = ["🏠 Overview", "🔍 Single model", "📈 ROC curves", "🧾 Customer scores", "🧠 What drives defaults"]

st.markdown(
    f"""
    <style>
    .kpi {{border:1px solid rgba(128,128,128,.3); border-left:5px solid {ACCENT};
          border-radius:10px; padding:10px 14px; background:rgba(15,118,110,.07);}}
    .kpi-l {{font-size:.75rem; opacity:.7; text-transform:uppercase; letter-spacing:.05em;}}
    .kpi-v {{font-size:1.55rem; font-weight:700; line-height:1.3;}}
    .kpi-s {{font-size:.8rem; opacity:.7;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi(label, value, sub=""):
    return (f'<div class="kpi"><div class="kpi-l">{label}</div>'
            f'<div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div></div>')


# ------------------------------------------------------------------ helpers
@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


@st.cache_data
def load_demo():
    return pd.read_csv(DEMO_FILE)


def evaluate(model, X, y):
    """Predict with a saved pipeline (no retraining) and compute the 6 metrics."""
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    else:
        score = model.decision_function(X)
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, score),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return y_pred, score, metrics


@st.cache_data(show_spinner="Scoring the five models...")
def evaluate_all(X, y):
    out = {}
    for name in MODEL_FILES:
        y_pred, score, metrics = evaluate(load_model(name), X, y)
        out[name] = {"pred": y_pred, "score": score, "metrics": metrics}
    return out


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🏦 Default Risk Lab")
    st.caption("Credit card default · 5 pre-trained classifiers")

    st.subheader("Data")
    uploaded = st.file_uploader("Upload a test CSV", type=["csv"])
    st.caption(f"Needs the 23 training features and a `{TARGET}` column (0/1). "
               "With no upload, the app shows the bundled test split.")

    st.subheader("Model")
    model_name = st.selectbox("Classifier", list(MODEL_FILES.keys()))

    st.subheader("Page")
    page = st.radio("Go to", PAGES, label_visibility="collapsed")

# ------------------------------------------------------------------ data (never empty)
if uploaded is not None:
    df = pd.read_csv(uploaded)
    source, is_demo = uploaded.name, False
elif os.path.exists(DEMO_FILE):
    df = load_demo()
    source, is_demo = f"{DEMO_FILE} (bundled demo)", True
else:
    st.error("No data available. Upload a test CSV in the sidebar.")
    st.stop()

if TARGET not in df.columns:
    st.error(f"The file must contain the target column `{TARGET}`.")
    st.stop()

reference_model = load_model(model_name)
expected = list(getattr(reference_model, "feature_names_in_",
                        [c for c in df.columns if c != TARGET]))
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()

X = df[expected]
y = df[TARGET].astype(int)
results = evaluate_all(X, y)
comparison = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T
comparison.index.name = "ML Model Name"

n_def = int((y == 1).sum())
st.title(page)
if is_demo:
    st.info("Showing results on the bundled test split. Upload your own CSV in the sidebar to replace it.")
st.markdown(
    f"**Data:** `{source}` · **{len(df):,}** customers · "
    f"**{n_def:,}** defaulters ({n_def / len(df):.1%})"
)

# ------------------------------------------------------------------ page: overview
if page == PAGES[0]:
    cols = st.columns(4)
    for col, metric in zip(cols, ["Accuracy", "AUC", "Recall", "MCC"]):
        best = comparison[metric].idxmax()
        col.markdown(kpi(f"Best {metric}", best, f"{comparison.loc[best, metric]:.4f}"),
                     unsafe_allow_html=True)

    st.markdown("### Scoreboard")
    st.dataframe(comparison.style.format("{:.4f}").highlight_max(axis=0, color="#bfe5dc"))
    st.caption("Highlighted = best model for that metric. Precision, Recall and F1 refer to the Default class.")

    st.markdown("### All metrics side by side")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    comparison.plot(kind="bar", ax=ax, colormap="YlGnBu", edgecolor="white", width=0.8)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15)
    ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
    st.pyplot(fig)

# ------------------------------------------------------------------ page: single model
elif page == PAGES[1]:
    st.subheader(model_name)
    metrics = results[model_name]["metrics"]
    y_pred = results[model_name]["pred"]

    for col, (k, v) in zip(st.columns(6), metrics.items()):
        col.markdown(kpi(k, f"{v:.4f}"), unsafe_allow_html=True)

    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Confusion matrix")
        per_class = st.toggle("Show recall per class instead of counts")
        cm = confusion_matrix(y, y_pred, normalize="true" if per_class else None)
        fig, ax = plt.subplots(figsize=(4.8, 3.9))
        sns.heatmap(cm, annot=True, fmt=".2f" if per_class else "d", cmap="YlGnBu",
                    cbar=False, ax=ax, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("#### Classification report")
        report = classification_report(y, y_pred, target_names=CLASS_NAMES, output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3))

# ------------------------------------------------------------------ page: ROC
elif page == PAGES[2]:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y, r["score"])
        lw = 3 if name == model_name else 1.4
        ax.plot(fpr, tpr, linewidth=lw, label=f"{name} (AUC {r['metrics']['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    st.caption(f"The thick line is the model selected in the sidebar ({model_name}).")

# ------------------------------------------------------------------ page: customer scores
elif page == PAGES[3]:
    out = df.copy()
    out["predicted"] = results[model_name]["pred"]
    out["risk_score"] = results[model_name]["score"].round(4)
    out["correct"] = out["predicted"] == out[TARGET]

    st.caption(f"Model: **{model_name}**. `risk_score` is the predicted default probability "
               "(for the SVM it is the decision-function value; higher = riskier).")
    c1, c2 = st.columns(2)
    top_n = c1.slider("Customers to show (riskiest first)", 10, 500, 50, step=10)
    wrong_only = c2.checkbox("Only misclassified customers")

    view = out[~out["correct"]] if wrong_only else out
    view = view.sort_values("risk_score", ascending=False).head(top_n)
    st.dataframe(view)
    st.download_button("⬇ Download all scored rows (CSV)",
                       out.to_csv(index=False).encode("utf-8"),
                       file_name="scored_customers.csv", mime="text/csv")

# ------------------------------------------------------------------ page: drivers
else:
    clf = reference_model.named_steps.get("clf", reference_model) if hasattr(reference_model, "named_steps") else reference_model
    if hasattr(clf, "feature_importances_"):
        values = pd.Series(clf.feature_importances_, index=expected)
        title = "Decision Tree feature importance"
    elif hasattr(clf, "coef_"):
        values = pd.Series(np.abs(clf.coef_[0]), index=expected)
        title = "Logistic Regression |coefficient| (features were standardised)"
    else:
        st.info(f"{model_name} has no built-in feature ranking. "
                "Select **Decision Tree** or **Logistic Regression** in the sidebar.")
        st.stop()

    top = values.sort_values(ascending=False).head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(top.index, top.values, color=ACCENT)
    ax.set_title(title, fontsize=10)
    st.pyplot(fig)
    st.caption("Repayment-status columns (PAY_*) usually dominate: a customer who is already "
               "late on payments is the strongest warning sign.")
