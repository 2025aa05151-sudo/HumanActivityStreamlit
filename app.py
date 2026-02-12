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
        "file": "models/logistic.pkl",
        "needs_scaling": True
    },
    "Decision Tree": {
        "file": "models/decision_tree.pkl",
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
model = joblib.load(config["file"])

# Load scaler (always available but used conditionally)
scaler = joblib.load("models/scaler.pkl")

# Apply scaling if needed
if config["needs_scaling"]:
    X_input = scaler.transform(X_test)
else:
    X_input = X_test

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_input)
y_prob = model.predict_proba(X_input)

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

st.write(f"**Accuracy:** {accuracy:.4f}")
st.write(f"**Precision (Macro):** {precision:.4f}")
st.write(f"**Recall (Macro):** {recall:.4f}")
st.write(f"**F1 Score (Macro):** {f1:.4f}")
st.write(f"**AUC (OvR):** {auc:.4f}")
st.write(f"**MCC:** {mcc:.4f}")

# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)