"""
vectorize.py

Creates a copy of the dataset with labels in vector format.

Author: Aidan Murray
Date: 2025-09-26
"""

import ast
from pathlib import Path
import pandas as pd

def vectorize(series):
    labels = series.to_list()
    vectors = []
    for item in labels:
        item = ast.literal_eval(item)
        vector = [1 if label in item else 0 for label in allowed_labels]
        vectors.append(vector)
    return vectors

BASE_PATH = Path("../../data")
LABEL_PATH = BASE_PATH / "prompt/allowed_labels.txt"
DATASETS = ("test.csv", "train_full.csv", "train_partial.csv", "validation.csv")

with open(LABEL_PATH, "r") as f:
    allowed_labels = f.read()
allowed_labels = ast.literal_eval(allowed_labels)

for dataset in DATASETS:
    path = BASE_PATH / dataset
    df = pd.read_csv(path)
    labels = df["label.name"]
    vectors = vectorize(labels)
    df["label.vector"] = vectors
    df.to_csv(path)