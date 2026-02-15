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
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
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
# Load Internal Test Data
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
    "Choose Model", list(MODEL_CONFIG.keys()), key="model_selector"
)
model_path = MODEL_CONFIG[selected_model_name]
loaded = joblib.load(model_path)

if selected_model_name == "XGBoost":
    model, le = loaded
else:
    model = loaded

# Precompute internal test predictions
y_pred_internal = model.predict(X_test)
if selected_model_name == "XGBoost":
    y_pred_internal = le.inverse_transform(y_pred_internal)

if hasattr(model, "predict_proba"):
    y_prob_internal = model.predict_proba(X_test)
else:
    y_prob_internal = None

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Evaluation", "Explainability", "Upload & Predict"
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
        ["Internal Test Set", "Uploaded Test Set"],
        horizontal=True
    )

    st.markdown("---")

    # -------------------------------
    # INTERNAL TEST METRICS
    # -------------------------------
    if evaluation_mode == "Internal Test Set":
        st.subheader("Internal Test Set Metrics")

        metrics = df_metrics.loc[selected_model_name]
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{metrics['Test Accuracy']:.4f}")
        col2.metric("Precision", f"{metrics['Precision']:.4f}")
        col3.metric("Recall", f"{metrics['Recall']:.4f}")
        col4.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
        col5.metric("AUC", f"{metrics['AUC']:.4f}")
        col6.metric("MCC", f"{metrics['MCC']:.4f}")

        st.markdown("---")
        st.subheader("Confusion Matrix (Internal Test)")

        labels = np.unique(y_test)
        label_names = [CLASS_NAMES[l] for l in labels]
        cm = confusion_matrix(y_test, y_pred_internal, labels=labels)

        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names, ax=ax_cm
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

# -------------------------------
# UPLOADED TEST METRICS
# -------------------------------
    else:
        st.subheader("Upload Your Test Dataset for Evaluation")

        # =====================================
        # CONFIGURABLE SAMPLE DOWNLOAD SECTION
        # =====================================
        st.markdown("### Download Compatible Test Dataset")

        max_rows = len(X_test)

        sample_size = st.slider(
            "Select number of rows for sample dataset",
            min_value=10,
            max_value=max_rows,
            value=min(200, max_rows),
            step=10
        )

        # Sample data
        sampled_indices = X_test.sample(
            n=sample_size,
            random_state=42
        ).index

        sample_df = X_test.loc[sampled_indices].copy()
        sample_df["target"] = y_test[sampled_indices]

        csv_download = sample_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label=f"Download Sample Test Dataset ({sample_size} rows)",
            data=csv_download,
            file_name=f"sample_test_dataset_{sample_size}.csv",
            mime="text/csv",
        )

        st.markdown("---")

        # =====================================
        # FILE UPLOADER
        # =====================================
        uploaded_eval_file = st.file_uploader(
            "Upload CSV containing top 30 features + target",
            type="csv"
        )

        if uploaded_eval_file is not None:
            df_upload_eval = pd.read_csv(uploaded_eval_file)

            if 'target' not in df_upload_eval.columns:
                st.error("Uploaded file must contain a 'target' column.")
            else:
                X_upload = df_upload_eval.drop(columns=['target'])
                y_upload = df_upload_eval['target'].values

                # Strict column order check
                if list(X_upload.columns) != list(X_test.columns):
                    st.error(
                        "Uploaded CSV columns must match training features EXACTLY.\n"
                        "Please download the sample file above and use that format."
                    )
                else:
                    # Predict
                    y_pred_upload = model.predict(X_upload)

                    if selected_model_name == "XGBoost":
                        y_pred_upload = le.inverse_transform(y_pred_upload)

                    # Probabilities
                    if hasattr(model, "predict_proba"):
                        y_prob_upload = model.predict_proba(X_upload)
                    else:
                        y_prob_upload = None

                    # Compute Metrics
                    acc = accuracy_score(y_upload, y_pred_upload)
                    prec = precision_score(y_upload, y_pred_upload, average='weighted')
                    rec = recall_score(y_upload, y_pred_upload, average='weighted')
                    f1 = f1_score(y_upload, y_pred_upload, average='weighted')
                    mcc = matthews_corrcoef(y_upload, y_pred_upload)

                    # Safe Multiclass AUC Calculation
                    if y_prob_upload is not None:
                        try:
                            auc_val = roc_auc_score(
                                y_upload,
                                y_prob_upload,
                                multi_class="ovr",
                                average="weighted"
                            )
                        except ValueError:
                            # Happens when not all classes present
                            auc_val = None
                    else:
                        auc_val = None

                    # Display Metrics
                    st.markdown("### Uploaded Test Metrics")

                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    col1.metric("Accuracy", f"{acc:.4f}")
                    col2.metric("Precision", f"{prec:.4f}")
                    col3.metric("Recall", f"{rec:.4f}")
                    col4.metric("F1 Score", f"{f1:.4f}")
                    col5.metric("AUC", f"{auc_val:.4f}" if auc_val else "N/A (Insufficient class diversity)")
                    col6.metric("MCC", f"{mcc:.4f}")

                    st.markdown("---")

                    # Confusion Matrix
                    st.subheader("Confusion Matrix (Uploaded Test)")
                    cm_upload = confusion_matrix(y_upload, y_pred_upload)

                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        cm_upload,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        ax=ax2
                    )
                    ax2.set_xlabel("Predicted")
                    ax2.set_ylabel("Actual")
                    st.pyplot(fig2)


# =========================================================
# TAB 3 — Explainability
# =========================================================
with tab3:
    st.subheader("Feature Importance")
    if selected_model_name in ["Decision Tree", "Random Forest", "XGBoost"]:
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
        sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax_imp)
        st.pyplot(fig_imp)
    else:
        st.info("Explainability not available for this model.")

# =========================================================
# TAB 4 — Upload & Predict (unchanged logic)
# =========================================================
with tab4:
    st.subheader("Upload Test Dataset (Features Only)")
    uploaded_predict_file = st.file_uploader(
        "Upload CSV file containing test features",
        type="csv"
    )

    if uploaded_predict_file is not None:
        df_features = pd.read_csv(uploaded_predict_file)
        if set(df_features.columns) != set(X_test.columns):
            st.error("Uploaded CSV columns do not match training features.")
        else:
            preds = model.predict(df_features)
            if selected_model_name == "XGBoost":
                preds = le.inverse_transform(preds)
            pred_labels = [CLASS_NAMES[p] for p in preds]
            df_out = df_features.copy()
            df_out["Predicted Activity"] = pred_labels
            st.dataframe(df_out.head())
