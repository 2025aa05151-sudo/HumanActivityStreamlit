import json
import pandas as pd

# -----------------------------
# Load Metrics
# -----------------------------
with open("models/all_metrics.json") as f:
    all_metrics = json.load(f)

df = pd.DataFrame(all_metrics).T

# -----------------------------
# Extract Required Columns
# -----------------------------
required_columns = [
    "Test Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
]

df_required = df[required_columns].copy()

# Rename columns exactly as required
df_required.rename(columns={
    "Test Accuracy": "Accuracy",
    "F1 Score": "F1"
}, inplace=True)

# Add ML Model Name as proper column
df_required.insert(0, "ML Model Name", df_required.index)

# Reset index
df_required.reset_index(drop=True, inplace=True)

# -----------------------------
# Generate Observations Table
# -----------------------------
observations = []

for _, row in df_required.iterrows():

    model_name = row["ML Model Name"]
    acc = row["Accuracy"]
    f1 = row["F1"]
    mcc = row["MCC"]

    observation_text = (
        f"Accuracy={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}. "
    )

    if "Random Forest" in model_name or "XGBoost" in model_name:
        observation_text += "Ensemble model shows strong and stable performance."
    elif "Decision Tree" in model_name:
        observation_text += "Tree model provides good interpretability with moderate generalization."
    elif "Logistic Regression" in model_name:
        observation_text += "Linear model performs competitively on structured features."
    elif "KNN" in model_name:
        observation_text += "Distance-based model depends heavily on feature scaling."
    elif "Naive Bayes" in model_name:
        observation_text += "Probabilistic model assumes feature independence."

    observations.append({
        "ML Model Name": model_name,
        "Observation about model performance": observation_text
    })

df_observations = pd.DataFrame(observations)

# -----------------------------
# Build README Content
# -----------------------------
md = f"""
# Machine Learning Assignment 2

## a) Problem Statement

The objective of this project is to implement multiple classification models 
to predict human activities using sensor-based data and compare their performance.

---

## b) Dataset Description

The dataset used is a multi-class classification dataset containing  561 features and 10299 instances. The goal is to classify human activity 
into six categories.
Since it has a lot of features(561), have trimmd them down and using top 30 only.(using random forests)
Dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

---

## c) Model Comparison Table

{df_required.to_markdown(index=False, floatfmt=".4f")}

---

## Observations on Model Performance

{df_observations.to_markdown(index=False)}

---

All six required models were implemented:
Logistic Regression, Decision Tree, KNN, Naive Bayes,
Random Forest (Ensemble), and XGBoost (Ensemble).

"""

# -----------------------------
# Save File
# -----------------------------
with open("README.md", "w") as f:
    f.write(md)

print("README.md generated successfully.")
