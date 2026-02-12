import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

os.makedirs("models", exist_ok=True)

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
X_test = pd.read_csv("data/X_test.csv")

# -----------------------------
# Train Random Forest for importance
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# -----------------------------
# Extract importance
# -----------------------------
importances = rf.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# -----------------------------
# Select Top 30
# -----------------------------
top_30_features = importance_df.head(30)["Feature"].tolist()

# Save feature list
pd.Series(top_30_features).to_csv(
    "models/top_30_features.csv",
    index=False
)

# -----------------------------
# Create reduced datasets
# -----------------------------
X_train_selected = X_train[top_30_features]
X_test_selected = X_test[top_30_features]

X_train_selected.to_csv("data/X_train_selected.csv", index=False)
X_test_selected.to_csv("data/X_test_selected.csv", index=False)

print("Top 30 features selected.")
print("Reduced datasets saved.")
