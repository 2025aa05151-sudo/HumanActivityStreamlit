import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="HAR ML App", layout="wide")

st.title("Human Activity Recognition (HAR)")
st.markdown("Multiclass Classification using Smartphone Sensor Data")

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()

st.sidebar.header("Dataset Overview")
st.sidebar.write("Training Samples:", X_train.shape[0])
st.sidebar.write("Original Features:", X_train.shape[1])

# -------------------------------
# Feature Selection
# -------------------------------
st.header("Feature Selection (Random Forest Importance)")

@st.cache_resource
def compute_feature_importance(X, y):
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y.values.ravel())
    return rf.feature_importances_

importances = compute_feature_importance(X_train, y_train)

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

top_30 = importance_df.head(30)

st.subheader("Feature Reduction Summary")
st.write("Original Feature Count:", X_train.shape[1])
st.write("Selected Top 30 Features")

# -------------------------------
# Plot Top 10 Features
# -------------------------------
st.subheader("Top 10 Important Features")

top_10 = importance_df.head(10)

fig, ax = plt.subplots()
ax.barh(top_10["Feature"], top_10["Importance"])
ax.invert_yaxis()
ax.set_xlabel("Importance Score")
ax.set_title("Top 10 Feature Importances")

st.pyplot(fig)

st.markdown("---")
st.caption("M.Tech AIML - Assignment 2 | Feature Selection Phase")