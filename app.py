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
    "Logistic Regression": "models/logistic.pkl",
    "KNN": "models/knn.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

# -----------------------------
# Load Data
# -----------------------------
X_test = pd.read_csv("data/X_test_selected.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# -----------------------------
# Class Label Mapping
# -----------------------------
CLASS_NAMES = {
    1: "Walking",
    2: "Walking Upstairs",
    3: "Walking Downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying"
}

# -----------------------------
# Model Leaderboard
# -----------------------------
st.subheader("Model Comparison")

with open("models/all_metrics.json") as f:
    all_metrics = json.load(f)

df_metrics = pd.DataFrame(all_metrics).T
df_metrics = df_metrics.sort_values("Accuracy", ascending=False)

st.dataframe(df_metrics.style.format("{:.4f}"))

# -----------------------------
# Model Selection
# -----------------------------
selected_model_name = st.selectbox(
    "Choose Model",
    list(MODEL_CONFIG.keys())
)

model_path = MODEL_CONFIG[selected_model_name]
loaded = joblib.load(model_path)

if selected_model_name == "XGBoost":
    model, le = loaded
else:
    model = loaded

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

if selected_model_name == "XGBoost":
    y_pred = le.inverse_transform(y_pred)

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)
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

auc = None
if y_prob is not None:
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

if auc is not None:
    col5.metric("AUC", f"{auc:.4f}")
else:
    col5.metric("AUC", "N/A")

col6.metric("MCC", f"{mcc:.4f}")


# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("Confusion Matrix")

labels = np.unique(y_test)
label_names = [CLASS_NAMES[l] for l in labels]

cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names,
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

st.pyplot(fig)

# -----------------------------
# Prediction Distribution
# -----------------------------
st.subheader("Prediction Distribution")

pred_counts = pd.Series(y_pred).value_counts().sort_index()

pred_counts.index = [CLASS_NAMES[i] for i in pred_counts.index]

fig2, ax2 = plt.subplots(figsize=(8, 5))

pred_counts.plot(kind="bar", ax=ax2)

ax2.set_xlabel("Predicted Activity")
ax2.set_ylabel("Count")

plt.xticks(rotation=45, ha="right")

st.pyplot(fig2)

# -----------------------------
# Feature Importance
# -----------------------------
if selected_model_name in ["Decision Tree", "Random Forest", "XGBoost"]:
    st.subheader("Feature Importance")

    if selected_model_name == "XGBoost":
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_

    feature_names = X_test.columns

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(15)

    fig3, ax3 = plt.subplots()
    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        ax=ax3
    )

    st.pyplot(fig3)
