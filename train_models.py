import os
import sys

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

# ---------------------------------------------------------------- 1. load
csv_path = "UCI_Credit_Card.csv"
df = pd.read_csv(csv_path)
print("Raw shape:", df.shape)

# ---------------------------------------------------------------- 2. clean
# Drop ID column 
df = df.drop(columns=[c for c in df.columns if c.lower() == "id"])

# Rename target column
target_candidates = [c for c in df.columns if "default" in c.lower()]
df = df.rename(columns={target_candidates[0]: TARGET})


df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

print("Clean shape:", df.shape)
print("Class balance:\n", df[TARGET].value_counts(normalize=True).round(3))

# ---------------------------------------------------------------- 3. split
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Save test data for the Streamlit app (features + target)
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
        # No probability=True (slow); AUC is computed from decision_function instead
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
print("\nMarkdown version:\n")
print(results.to_markdown(index=False) if hasattr(results, "to_markdown") else "")
