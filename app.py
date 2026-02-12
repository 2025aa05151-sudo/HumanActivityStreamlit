import streamlit as st

st.set_page_config(
    page_title="Human Activity Recognition App",
    page_icon="🏃",
    layout="wide"
)

st.title("Human Activity Recognition (HAR) - ML App")

st.markdown("""
This application will demonstrate multiple machine learning models 
for multiclass Human Activity Recognition using smartphone sensor data.
""")

st.sidebar.header("⚙ Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select a Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

st.sidebar.write("Selected Model:", model_choice)

st.subheader("Model Overview")

st.info(
    "Model training and evaluation metrics will be displayed here "
)

if st.button("Run Model (Coming Soon)"):
    st.success(f"{model_choice} selected! Model execution will be implemented.")

st.markdown("---")
st.caption("M.Tech AIML - Machine Learning Assignment 2 | HAR Multiclass Classification")
