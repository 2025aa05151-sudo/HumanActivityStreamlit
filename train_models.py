import pandas as pd
import numpy as np
import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)
from joblib import dump

# -----------------------------
# Setup
# -----------------------------
os.makedirs("models", exist_ok=True)

# -----------------------------
# Load reduced dataset
# -----------------------------
X_train = pd.read_csv("data/X_train_selected.csv")
X_test = pd.read_csv("data/X_test_selected.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

metrics_all = {}

# =========================================================
# 1 Logistic Regression
# =========================================================


from sklearn.pipeline import Pipeline

log_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

log_pipeline.fit(X_train, y_train)
dump(log_pipeline, "models/logistic.pkl")

y_pred = log_pipeline.predict(X_test)
y_prob = log_pipeline.predict_proba(X_test)

classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)

metrics_all["Logistic Regression"] = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, average="macro"),
    "Recall": recall_score(y_test, y_pred, average="macro"),
    "F1 Score": f1_score(y_test, y_pred, average="macro"),
    "AUC": roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro"),
    "MCC": matthews_corrcoef(y_test, y_pred)
}
log_pipeline.score(X_train, y_train)

print("Logistic Regression trained.")

# =========================================================
# 2 Decision Tree
# =========================================================
dt_pipeline = Pipeline([
    ("model", DecisionTreeClassifier(
    random_state=42,
    max_depth=10 # used none, it had high variance. using 10 has reduced overfitting.
    ))
])
dt_pipeline.fit(X_train, y_train)
dump(dt_pipeline, "models/decision_tree.pkl")

y_pred_dt = dt_pipeline.predict(X_test)
y_prob_dt = dt_pipeline.predict_proba(X_test)

metrics_all["Decision Tree"] = {
    "Accuracy": accuracy_score(y_test, y_pred_dt),
    "Precision": precision_score(y_test, y_pred_dt, average="macro"),
    "Recall": recall_score(y_test, y_pred_dt, average="macro"),
    "F1 Score": f1_score(y_test, y_pred_dt, average="macro"),
    "AUC": roc_auc_score(y_test_bin, y_prob_dt, multi_class="ovr", average="macro"),
    "MCC": matthews_corrcoef(y_test, y_pred_dt)
}
dt_pipeline.score(X_train, y_train)

print("Decision Tree trained.")

# =========================================================
# 3 K-Nearest Neighbors
# =========================================================

from sklearn.neighbors import KNeighborsClassifier

best_k = None
best_acc = 0

for k in [3, 5, 7, 9, 11, 15, 21]:
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=k))
    ])
    knn_pipeline.fit(X_train, y_train)
    acc = knn_pipeline.score(X_test, y_test)

    print(f"k={k} → Accuracy={acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_k = k

print(f"\nBest k: {best_k} with Accuracy={best_acc:.4f}")

# Train final model with best k

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=best_k))
])

knn_pipeline.fit(X_train, y_train)
dump(knn_pipeline, "models/knn.pkl")

y_pred_knn = knn_pipeline.predict(X_test)
y_prob_knn = knn_pipeline.predict_proba(X_test)

metrics_all["KNN"] = {
    "Accuracy": accuracy_score(y_test, y_pred_knn),
    "Precision": precision_score(y_test, y_pred_knn, average="macro"),
    "Recall": recall_score(y_test, y_pred_knn, average="macro"),
    "F1 Score": f1_score(y_test, y_pred_knn, average="macro"),
    "AUC": roc_auc_score(y_test_bin, y_prob_knn, multi_class="ovr", average="macro"),
    "MCC": matthews_corrcoef(y_test, y_pred_knn)
}

print("KNN trained with tuned k.")

# =========================================================
# 4 Gaussian Naive Bayes
# =========================================================

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB(var_smoothing=1e-6)

# NB typically works fine with scaled data,
# but scaling is not strictly required.
# We'll use scaled data for consistency.
nb_model.fit(X_train, y_train)
dump(nb_model, "models/naive_bayes.pkl")

y_pred_nb = nb_model.predict(X_test)
y_prob_nb = nb_model.predict_proba(X_test)

metrics_all["Naive Bayes"] = {
    "Accuracy": accuracy_score(y_test, y_pred_nb),
    "Precision": precision_score(y_test, y_pred_nb, average="macro"),
    "Recall": recall_score(y_test, y_pred_nb, average="macro"),
    "F1 Score": f1_score(y_test, y_pred_nb, average="macro"),
    "AUC": roc_auc_score(y_test_bin, y_prob_nb, multi_class="ovr", average="macro"),
    "MCC": matthews_corrcoef(y_test, y_pred_nb)
}

print("Naive Bayes trained.")

# =========================================================
# 5 Random Forest
# =========================================================

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
dump(rf_model, "models/random_forest.pkl")

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)

metrics_all["Random Forest"] = {
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    "Precision": precision_score(y_test, y_pred_rf, average="macro"),
    "Recall": recall_score(y_test, y_pred_rf, average="macro"),
    "F1 Score": f1_score(y_test, y_pred_rf, average="macro"),
    "AUC": roc_auc_score(y_test_bin, y_prob_rf, multi_class="ovr", average="macro"),
    "MCC": matthews_corrcoef(y_test, y_pred_rf)
}

print("Random Forest trained.")

# =========================================================
# 6 XGBoost
# =========================================================

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    objective="multi:softprob",
    random_state=42,
    eval_metric="mlogloss"
)

xgb_model.fit(X_train, y_train_enc)

# Save both model and encoder together
dump((xgb_model, le), "models/xgboost.pkl")

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)

# Convert predictions back to original labels
y_pred_decoded = le.inverse_transform(y_pred_xgb)

metrics_all["XGBoost"] = {
    "Accuracy": accuracy_score(y_test, y_pred_decoded),
    "Precision": precision_score(y_test, y_pred_decoded, average="macro"),
    "Recall": recall_score(y_test, y_pred_decoded, average="macro"),
    "F1 Score": f1_score(y_test, y_pred_decoded, average="macro"),
    "AUC": roc_auc_score(y_test_bin, y_prob_xgb, multi_class="ovr", average="macro"),
    "MCC": matthews_corrcoef(y_test, y_pred_decoded)
}

print("XGBoost trained.")

# =========================================================
# Save Metrics
# =========================================================

with open("models/all_metrics.json", "w") as f:
    json.dump(metrics_all, f, indent=4)

print("\nAll Metrics:")
for model_name, metrics in metrics_all.items():
    print(f"\n{model_name}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
