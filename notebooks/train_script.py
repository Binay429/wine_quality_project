"""
Wine Quality Prediction - Model Training Script
=================================================
Loads WineQT.csv, performs EDA, trains and compares Logistic Regression,
Random Forest and Gradient Boosting classifiers, and serializes the best
performing model (plus the scaler and feature order) for use by the
Django application (predictor/ml_model.py).
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "WineQT.csv"
MODEL_DIR = BASE_DIR / "model"
FIG_DIR = Path(__file__).resolve().parent / "figures"
MODEL_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET_COLUMN = "quality"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# Id is an identifier, not a model input
X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()

# ---------------------------------------------------------------------------
# 2. Exploratory data analysis
# ---------------------------------------------------------------------------
quality_counts = y.value_counts().sort_index()
quality_pct = (quality_counts / len(y) * 100).round(2)
print("\nQuality distribution:\n", pd.DataFrame({"count": quality_counts, "%": quality_pct}))

plt.figure(figsize=(7, 4.5))
sns.countplot(x=y, order=sorted(y.unique()), color="#722F37")
plt.title("Wine Quality Class Distribution")
plt.xlabel("Quality score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIG_DIR / "quality_distribution.png", dpi=110)
plt.close()

plt.figure(figsize=(9, 7))
corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=110)
plt.close()

print("\nDescriptive statistics:\n", X.describe().T[["mean", "std", "min", "max"]])

# ---------------------------------------------------------------------------
# 3. Preprocessing - stratified 80/20 split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classes = sorted(y.unique())

# ---------------------------------------------------------------------------
# 4. Train models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

results = {}
fitted = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

    y_test_bin = label_binarize(y_test, classes=model.classes_)
    try:
        auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
    except ValueError:
        auc = float("nan")

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_roc_auc": auc,
    }
    fitted[name] = model

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=classes)
    disp.ax_.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"confusion_{name.replace(' ', '_').lower()}.png", dpi=110)
    plt.close()

results_df = pd.DataFrame(results).T.round(3)
print("\nModel comparison:\n", results_df)

# ---------------------------------------------------------------------------
# 5. Select best model (highest macro F1) and persist artifacts
# ---------------------------------------------------------------------------
best_name = results_df["macro_f1"].astype(float).idxmax()
best_model = fitted[best_name]
print(f"\nSelected model: {best_name}")

# Feature importance (tree-based models only) for reference
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False)
    plt.figure(figsize=(7, 5))
    sns.barplot(x=importances.values, y=importances.index, color="#722F37")
    plt.title(f"Feature Importance — {best_name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance.png", dpi=110)
    plt.close()

with open(MODEL_DIR / "wine_quality_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open(MODEL_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

metadata = {
    "model_name": best_name,
    "uses_scaler": best_name == "Logistic Regression",
    "feature_order": FEATURE_COLUMNS,
    "classes": [int(c) for c in classes],
    "metrics": {k: {m: round(float(v), 4) for m, v in d.items()} for k, d in results.items()},
    "random_state": RANDOM_STATE,
    "test_size": 0.2,
}
with open(MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

results_df.to_csv(FIG_DIR / "model_comparison.csv")

print("\nSaved model, scaler and metadata to", MODEL_DIR)
