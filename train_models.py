import pandas as pd
import numpy as np
import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
from joblib import dump



os.makedirs("models", exist_ok=True)

X_train = pd.read_csv("data/X_train_selected.csv")
X_test = pd.read_csv("data/X_test_selected.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dump(scaler, "models/scaler.pkl")

# -----------------------------
# Train Logistic Regression
# -----------------------------
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)
dump(model, "models/logistic.pkl")

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
mcc = matthews_corrcoef(y_test, y_pred)

# Multiclass AUC (OvR)
classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)

auc = roc_auc_score(
    y_test_bin,
    y_prob,
    multi_class="ovr",
    average="macro"
)

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "AUC": auc,
    "MCC": mcc
}

# Save metrics
with open("models/logistic_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Logistic Regression trained and saved.")
print("Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")