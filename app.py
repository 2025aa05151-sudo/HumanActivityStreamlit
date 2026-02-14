import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Human Activity Recognition")
st.write("Multiclass Classification Dashboard")

# -----------------------------
# Model Configuration
# -----------------------------
MODEL_CONFIG = {
    "Logistic Regression": {
        "model_path": "models/logistic.pkl",
        "needs_scaling": True
    },
    "KNN": {
        "model_path": "models/knn.pkl",
        "needs_scaling": True
    },
    "Decision Tree": {
        "model_path": "models/decision_tree.pkl",
        "needs_scaling": False
    },
    "Naive Bayes": {
        "model_path": "models/naive_bayes.pkl",
        "needs_scaling": False
    },
    "Random Forest": {
        "model_path": "models/random_forest.pkl",
        "needs_scaling": False
    },
    "XGBoost": {
        "model_path": "models/xgboost.pkl",
        "needs_scaling": False
    }
}

# -----------------------------
# Load Data
# -----------------------------
X_test = pd.read_csv("data/X_test_selected.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# -----------------------------
# Model Selection
# -----------------------------
selected_model_name = st.selectbox(
    "Choose Model",
    list(MODEL_CONFIG.keys())
)

config = MODEL_CONFIG[selected_model_name]

# Load model
model = joblib.load(config["model_path"])

# Load scaler
if config["needs_scaling"]:
    scaler = joblib.load("models/scaler.pkl")
    X_input = scaler.transform(X_test)
else:
    X_input = X_test

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_input)
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_input)
else:
    y_prob = None


classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)

# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
mcc = matthews_corrcoef(y_test, y_pred)
auc = roc_auc_score(
    y_test_bin,
    y_prob,
    multi_class="ovr",
    average="macro"
)

# -----------------------------
# Display Metrics
# -----------------------------
st.subheader(f"Evaluation Metrics — {selected_model_name}")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("Precision", f"{precision:.4f}")
col3.metric("Recall", f"{recall:.4f}")
col4.metric("F1 Score", f"{f1:.4f}")
col5.metric("AUC", f"{auc:.4f}")
col6.metric("MCC", f"{mcc:.4f}")

# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("Confusion Matrix")

labels = np.unique(y_test)
cm = confusion_matrix(y_test, y_pred, labels=labels)

st.write("Raw Confusion Matrix:")
st.write(cm)

st.write("Unique classes in y_test:", np.unique(y_test))
st.write("Unique classes in y_pred:", np.unique(y_pred))


# fig, ax = plt.subplots()
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
# ax.set_xlabel("Predicted")
# ax.set_ylabel("Actual")

# st.pyplot(fig)