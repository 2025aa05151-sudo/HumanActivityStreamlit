import pandas as pd
import argparse

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--size",
    type=int,
    default=100,
    help="Number of test samples to export"
)

args = parser.parse_args()
sample_size = args.size

# -----------------------------
# Load test features
# -----------------------------
X_test = pd.read_csv("data/X_test_selected.csv")

if sample_size > len(X_test):
    sample_size = len(X_test)

# Sample without replacement
sampled = X_test.sample(n=sample_size, random_state=42)

# Save file
output_name = f"test_upload_{sample_size}.csv"
sampled.to_csv(output_name, index=False)

print(f"{output_name} created successfully.")
