import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap

from sklearn.metrics import (
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
# Load Stored Metrics
# =========================================================
with open("models/all_metrics.json") as f:
    all_metrics = json.load(f)

df_metrics = pd.DataFrame(all_metrics).T


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

if selected_model_name == "XGBoost":
    model, le = loaded
else:
    model = loaded


# =========================================================
# GLOBAL PREDICTIONS (FOR TEST SET)
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
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Evaluation",
    "Explainability",
    "Upload & Predict"
])


# =========================================================
# TAB 1 — Overview
# =========================================================
with tab1:

    st.subheader("Model Comparison")

    df_sorted = df_metrics.sort_values("Test Accuracy", ascending=False)

    best_model = df_sorted.index[0]
    st.success(f"Best Model (by Test Accuracy): {best_model}")

    st.dataframe(df_sorted.style.format("{:.4f}"))


# =========================================================
# TAB 2 — Evaluation
# =========================================================
with tab2:

    st.markdown(f"## Evaluation — {selected_model_name}")

    evaluation_mode = st.radio(
        "Select Evaluation Mode",
        ["Test Set Performance", "Cross-Validation Performance"],
        horizontal=True
    )

    selected_metrics = df_metrics.loc[selected_model_name]

    st.markdown("---")

    # =====================================================
    # TEST SET PERFORMANCE
    # =====================================================
    if evaluation_mode == "Test Set Performance":

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        col1.metric("Accuracy", f"{selected_metrics['Test Accuracy']:.4f}")
        col2.metric("Precision", f"{selected_metrics['Precision']:.4f}")
        col3.metric("Recall", f"{selected_metrics['Recall']:.4f}")
        col4.metric("F1 Score", f"{selected_metrics['F1 Score']:.4f}")
        col5.metric("AUC", f"{selected_metrics['AUC']:.4f}")
        col6.metric("MCC", f"{selected_metrics['MCC']:.4f}")

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

    # =====================================================
    # CROSS VALIDATION PERFORMANCE
    # =====================================================
    else:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("CV Accuracy (Mean)", f"{selected_metrics['CV Accuracy Mean']:.4f}")
        col2.metric("CV Accuracy (Std)", f"{selected_metrics['CV Accuracy Std']:.4f}")
        col3.metric("CV F1 (Mean)", f"{selected_metrics['CV F1 Mean']:.4f}")
        col4.metric("CV F1 (Std)", f"{selected_metrics['CV F1 Std']:.4f}")

        st.info(
            "Cross-validation metrics reflect model stability across 5 stratified folds."
        )


# =========================================================
# TAB 3 — Explainability (UNCHANGED)
# =========================================================
with tab3:

    if selected_model_name in ["Decision Tree", "Random Forest", "XGBoost"]:

        st.subheader("Feature Importance")

        if hasattr(model, "named_steps"):
            estimator = model.named_steps["model"]
        else:
            estimator = model

        importances = estimator.feature_importances_
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


# =========================================================
# TAB 4 — Upload & Predict (UNCHANGED LOGIC)
# =========================================================
with tab4:

    st.subheader("Upload Test Dataset (Features Only)")

    st.markdown("### Generate Sample Test File")

    sample_size = st.slider(
        "Select number of test samples to generate",
        min_value=10,
        max_value=len(X_test),
        value=100,
        step=10
    )

    sampled_test = X_test.sample(n=sample_size, random_state=42)

    csv_sample = sampled_test.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Sample Test CSV",
        data=csv_sample,
        file_name=f"test_upload_{sample_size}.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("### Or Upload Your Own Test File")

    uploaded_file = st.file_uploader(
        "Upload CSV file containing test features",
        type="csv"
    )

    if uploaded_file is not None:

        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.write("Uploaded Data Preview:")
            st.dataframe(uploaded_df.head())

            expected_columns = X_test.columns.tolist()

            if set(uploaded_df.columns) != set(expected_columns):
                st.error("Uploaded CSV columns do not match training features.")
                st.write("Expected columns:")
                st.write(expected_columns)
            else:
                uploaded_df = uploaded_df[expected_columns]

                predictions = model.predict(uploaded_df)

                if selected_model_name == "XGBoost":
                    predictions = le.inverse_transform(predictions)

                prediction_labels = [CLASS_NAMES[p] for p in predictions]

                result_df = uploaded_df.copy()
                result_df["Predicted Activity"] = prediction_labels

                st.success("Prediction Completed")
                st.dataframe(result_df.head())

        except Exception as e:
            st.error(f"Error processing file: {e}")
