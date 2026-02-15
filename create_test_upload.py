import pandas as pd
import argparse

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--size", type=int, default=100,
    help="Number of test samples to export"
)
args = parser.parse_args()
sample_size = args.size

# Load test features and labels
X_test = pd.read_csv("data/X_test_selected.csv")
y_test = pd.read_csv("data/y_test.csv")

# Ensure sample_size is not too big
if sample_size > len(X_test):
    sample_size = len(X_test)

sampled = X_test.sample(n=sample_size, random_state=42)
sampled["target"] = y_test.loc[sampled.index].values

# Save file
output_name = f"test_upload_with_labels_{sample_size}.csv"
sampled.to_csv(output_name, index=False)

print(f"{output_name} created successfully.")
