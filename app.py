import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Human Activity Recognition", layout="wide")
st.title("Human Activity Recognition - Dashboard")

# =========================================================
# MODEL CONFIG
# =========================================================
MODEL_CONFIG = {
    "Logistic Regression": "models/logistic.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

# =========================================================
# LOAD INTERNAL TEST DATA
# =========================================================
X_test = pd.read_csv("data/X_test_selected.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# =========================================================
# LOAD METRICS
# =========================================================
with open("models/all_metrics.json") as f:
    all_metrics = json.load(f)
df_metrics = pd.DataFrame(all_metrics).T

# =========================================================
# MODEL SELECTION
# =========================================================
selected_model = st.selectbox(
    "Select Model",
    list(MODEL_CONFIG.keys())
)
loaded = joblib.load(MODEL_CONFIG[selected_model])
if selected_model == "XGBoost":
    model, le = loaded
else:
    model = loaded

st.markdown("---")

# =========================================================
# EVALUATION MODE
# =========================================================
mode = st.radio(
    "Evaluation Mode",
    ["Internal Test Set", "Upload Test CSV"],
    horizontal=True
)

st.markdown("---")

# =========================================================
# INTERNAL TEST EVALUATION
# =========================================================
if mode == "Internal Test Set":
    st.subheader(f"{selected_model} — Internal Test Metrics")

    m = df_metrics.loc[selected_model]
    cols = st.columns(6)
    cols[0].metric("Accuracy", f"{m['Test Accuracy']:.4f}")
    cols[1].metric("Precision", f"{m['Precision']:.4f}")
    cols[2].metric("Recall", f"{m['Recall']:.4f}")
    cols[3].metric("F1 Score", f"{m['F1 Score']:.4f}")
    cols[4].metric("AUC", f"{m['AUC']:.4f}")
    cols[5].metric("MCC", f"{m['MCC']:.4f}")

    st.markdown("---")
    st.subheader("Confusion Matrix")

    y_pred = model.predict(X_test)
    if selected_model == "XGBoost":
        y_pred = le.inverse_transform(y_pred)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# =========================================================
# UPLOAD TEST EVALUATION
# =========================================================
else:
    st.subheader("Upload Compatible Test CSV")

    max_rows = len(X_test)

    sample_size = st.slider(
        "Select sample size to download",
        min_value=10,
        max_value=max_rows,
        value=min(200, max_rows),
        step=10
    )

    # Download sample with target
    sampled = X_test.sample(n=sample_size, random_state=42).copy()
    sampled["target"] = y_test[sampled.index]
    csv = sampled.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=f"Download Sample Test Data ({sample_size} rows)",
        data=csv,
        file_name=f"test_dataset_{sample_size}.csv",
        mime="text/csv"
    )

    st.markdown("---")
    uploaded = st.file_uploader(
        "Upload test CSV (features + target)",
        type="csv"
    )

    if uploaded:
        df = pd.read_csv(uploaded)

        if "target" not in df.columns:
            st.error("File must contain a 'target' column.")
        else:
            X_up = df.drop(columns=["target"])
            y_up = df["target"].values

            if list(X_up.columns) != list(X_test.columns):
                st.error("Columns must match training exactly.")
            else:
                y_pred = model.predict(X_up)
                if selected_model == "XGBoost":
                    y_pred = le.inverse_transform(y_pred)

                acc = accuracy_score(y_up, y_pred)
                prec = precision_score(y_up, y_pred, average="weighted")
                rec = recall_score(y_up, y_pred, average="weighted")
                f1v = f1_score(y_up, y_pred, average="weighted")
                mcc = matthews_corrcoef(y_up, y_pred)

                if hasattr(model, "predict_proba"):
                    try:
                        auc_val = roc_auc_score(
                            y_up,
                            model.predict_proba(X_up),
                            multi_class="ovr",
                            average="weighted"
                        )
                    except ValueError:
                        auc_val = None
                else:
                    auc_val = None

                st.subheader("Uploaded Test Metrics")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Accuracy", f"{acc:.4f}")
                c2.metric("Precision", f"{prec:.4f}")
                c3.metric("Recall", f"{rec:.4f}")
                c4.metric("F1 Score", f"{f1v:.4f}")
                c5.metric("AUC", f"{auc_val:.4f}" if auc_val else "N/A")
                c6.metric("MCC", f"{mcc:.4f}")

                st.markdown("---")
                st.subheader("Confusion Matrix")
                cm2 = confusion_matrix(y_up, y_pred)
                fig2, ax2 = plt.subplots(figsize=(7, 5))
                sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues", ax=ax2)
                ax2.set_xlabel("Predicted")
                ax2.set_ylabel("Actual")
                st.pyplot(fig2)
