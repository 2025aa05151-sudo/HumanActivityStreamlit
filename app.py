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
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# App Header
# =========================================================
st.title("Human Activity Recognition")
st.write("Multiclass Classification Dashboard")


# =========================================================
# Model Configuration
# =========================================================
MODEL_CONFIG = {
    "Logistic Regression": "models/logistic.pkl",
    "KNN": "models/knn.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}


# =========================================================
# Load Data
# =========================================================
X_test = pd.read_csv("data/X_test_selected.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()


# =========================================================
# Class Labels
# =========================================================
CLASS_NAMES = {
    1: "Walking",
    2: "Walking Upstairs",
    3: "Walking Downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying"
}


# =========================================================
# GLOBAL MODEL SELECTION
# =========================================================
selected_model_name = st.selectbox(
    "Choose Model",
    list(MODEL_CONFIG.keys()),
    key="model_selector"
)

model_path = MODEL_CONFIG[selected_model_name]
loaded = joblib.load(model_path)

# Handle XGBoost encoder
if selected_model_name == "XGBoost":
    model, le = loaded
else:
    model = loaded

# =========================================================
# GLOBAL PREDICTIONS
# =========================================================
y_pred = model.predict(X_test)

if selected_model_name == "XGBoost":
    y_pred = le.inverse_transform(y_pred)

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)
else:
    y_prob = None

labels = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=labels)


# =========================================================
# Tabs Layout
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Evaluation",
    "Explainability"
])


# =========================================================
# TAB 1 — Overview
# =========================================================
with tab1:

    st.subheader("Model Comparison")

    with open("models/all_metrics.json") as f:
        all_metrics = json.load(f)

    df_metrics = pd.DataFrame(all_metrics).T
    df_metrics = df_metrics.sort_values("Accuracy", ascending=False)

    best_model = df_metrics.index[0]
    st.success(f"Best Model (by Accuracy): {best_model}")

    st.dataframe(df_metrics.style.format("{:.4f}"))


# =========================================================
# TAB 2 — Evaluation
# =========================================================
with tab2:

    # -----------------------------
    # Metrics
    # -----------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    mcc = matthews_corrcoef(y_test, y_pred)

    auc_score = None
    if y_prob is not None:
        auc_score = roc_auc_score(
            y_test_bin,
            y_prob,
            multi_class="ovr",
            average="macro"
        )

    st.markdown(f"## Evaluation — {selected_model_name}")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")
    col4.metric("F1 Score", f"{f1:.4f}")
    col5.metric("AUC", f"{auc_score:.4f}" if auc_score else "N/A")
    col6.metric("MCC", f"{mcc:.4f}")

    st.markdown("---")

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    st.subheader("Confusion Matrix")

    label_names = [CLASS_NAMES[l] for l in labels]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax_cm
    )

    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    st.pyplot(fig_cm)

    # -----------------------------
    # Most Confused Pair
    # -----------------------------
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)

    max_confusion = np.unravel_index(
        np.argmax(cm_no_diag),
        cm_no_diag.shape
    )

    actual_idx, pred_idx = max_confusion

    st.info(
        f"Most confused: **{CLASS_NAMES[labels[actual_idx]]}** → "
        f"**{CLASS_NAMES[labels[pred_idx]]}** "
        f"({cm_no_diag[actual_idx, pred_idx]} times)"
    )

    st.markdown("---")

    # -----------------------------
    # Per-Class Metrics
    # -----------------------------
    st.subheader("Per-Class Metrics")

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).T
    report_df = report_df.loc[
        ~report_df.index.isin(["accuracy", "macro avg", "weighted avg"])
    ]

    report_df.index = [
        CLASS_NAMES[int(i)] for i in report_df.index
    ]

    st.dataframe(report_df.style.format("{:.4f}"))

    # -----------------------------
    # ROC Curves
    # -----------------------------
    if y_prob is not None:

        st.subheader("ROC Curves (One-vs-Rest)")

        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

        for i, label in enumerate(labels):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)

            ax_roc.plot(
                fpr,
                tpr,
                label=f"{CLASS_NAMES[label]} (AUC = {roc_auc:.3f})"
            )

        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()

        st.pyplot(fig_roc)

    # -----------------------------
    # Prediction Distribution
    # -----------------------------
    st.subheader("Prediction Distribution")

    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    pred_counts.index = [CLASS_NAMES[i] for i in pred_counts.index]

    fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
    pred_counts.plot(kind="bar", ax=ax_dist)

    ax_dist.set_xlabel("Predicted Activity")
    ax_dist.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")

    st.pyplot(fig_dist)


# =========================================================
# TAB 3 — Explainability
# =========================================================
with tab3:

    if selected_model_name in ["Decision Tree", "Random Forest", "XGBoost"]:

        st.subheader("Feature Importance")

        importances = model.feature_importances_
        feature_names = X_test.columns

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(15)

        fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
        sns.barplot(
            data=importance_df,
            x="Importance",
            y="Feature",
            ax=ax_imp
        )

        st.pyplot(fig_imp)

    else:
        st.info("Explainability not available for this model.")
