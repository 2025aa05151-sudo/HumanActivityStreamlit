import pandas as pd
import os

base_path = "UCI HAR Dataset"

features = pd.read_csv(
    os.path.join(base_path, "features.txt"),
    sep=r"\s+",
    header=None,
    names=["index", "feature"]
)

feature_names = features["feature"].tolist()

def make_unique(names):
    seen = {}
    unique_names = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            unique_names.append(name)
        else:
            seen[name] += 1
            new_name = f"{name}_{seen[name]}"
            unique_names.append(new_name)
    return unique_names

feature_names = make_unique(feature_names)

X_train = pd.read_csv(
    os.path.join(base_path, "train", "X_train.txt"),
    sep=r"\s+",
    header=None
)
X_train.columns = feature_names

y_train = pd.read_csv(
    os.path.join(base_path, "train", "y_train.txt"),
    header=None,
    names=["Activity"]
)

X_test = pd.read_csv(
    os.path.join(base_path, "test", "X_test.txt"),
    sep=r"\s+",
    header=None
)
X_test.columns = feature_names

y_test = pd.read_csv(
    os.path.join(base_path, "test", "y_test.txt"),
    header=None,
    names=["Activity"]
)

os.makedirs("data", exist_ok=True)

X_train.to_csv("data/X_train.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("Data preparation complete. CSV files saved in /data folder.")