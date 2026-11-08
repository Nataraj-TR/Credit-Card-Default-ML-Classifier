
"""
run this first in a cell:
    !pip install xlrd openpyxl kagglehub tabulate
     %pip install --upgrade --user numpy scipy scikit-learn
"""

import io
import os
import urllib.request
import zipfile

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)

RANDOM_STATE = 42
TARGET = "default"

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
KAGGLE_SLUG = "uciml/default-of-credit-card-clients-dataset"
LOCAL_FILE = "UCI_Credit_Card.csv"


# ---------------------------------------------------------------- 1. load
def from_uci():
    """Download the zip from the UCI repository and read the Excel sheet inside."""
    req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = [n for n in z.namelist() if n.lower().endswith((".xls", ".xlsx"))][0]
        
        return pd.read_excel(io.BytesIO(z.read(name)), header=1)


def from_kaggle():
    """Download the Kaggle mirror of the same dataset."""
    import kagglehub
    folder = kagglehub.dataset_download(KAGGLE_SLUG)
    return pd.read_csv(os.path.join(folder, "UCI_Credit_Card.csv"))


def load_dataset():
    for label, loader in (("UCI repository", from_uci), ("Kaggle", from_kaggle),
                          ("local CSV", lambda: pd.read_csv(LOCAL_FILE))):
        try:
            data = loader()
            print(f"Loaded dataset from: {label}")
            return data
        except Exception as err:  
            print(f"Could not load from {label}: {err}")
    raise RuntimeError("Dataset could not be loaded from any source.")


df = load_dataset()
print("Raw shape:", df.shape)

# ---------------------------------------------------------------- 2. clean
df.columns = [str(c).strip() for c in df.columns]

# Drop ID column if present 
df = df.drop(columns=[c for c in df.columns if c.lower() == "id"])

# Rename target column 
target_candidates = [c for c in df.columns if "default" in c.lower()]
df = df.rename(columns={target_candidates[0]: TARGET})

# Some sources call the first repayment column PAY_1; the saved models use PAY_0
if "PAY_1" in df.columns and "PAY_0" not in df.columns:
    df = df.rename(columns={"PAY_1": "PAY_0"})


df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

print("Clean shape:", df.shape)
print("Class balance:\n", df[TARGET].value_counts(normalize=True).round(3))

# ---------------------------------------------------------------- 3. split
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Save test data for the Streamlit app 
pd.concat([X_test, y_test], axis=1).to_csv("test_data.csv", index=False)

# ---------------------------------------------------------------- 4. models
# Scaling matters for LR, KNN and SVM (distance / gradient based).
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]),
    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=15)),
    ]),
    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB()),
    ]),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
      
        ("clf", SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE)),
    ]),
}

# ---------------------------------------------------------------- 5. train + evaluate
os.makedirs("model", exist_ok=True)
rows = []

for name, pipe in models.items():
    print(f"Training {name} ...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Scores for AUC
    if hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(X_test)[:, 1]
    else:
        y_score = pipe.decision_function(X_test)

    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    })

    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, os.path.join("model", filename))

# ---------------------------------------------------------------- 6. results
results = pd.DataFrame(rows).round(4)
results.to_csv("metrics_results.csv", index=False)
print("\n=== Comparison table (copy into README) ===")
print(results.to_string(index=False))
try:
    print("\nMarkdown version:\n")
    print(results.to_markdown(index=False))
except ImportError:
    print("(install tabulate for the markdown table: pip install tabulate)")
