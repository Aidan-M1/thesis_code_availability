"""
histogram.py

Creates a histogram of the true labels and predicted labels from the final tests.

Author: Aidan Murray
Date: 2025-09-26
"""

import pandas as pd
import ast
import matplotlib.pyplot as plt
from collections import Counter

# --- Validation labels ---
df = pd.read_csv("../../data/ecoregions/test.csv")
df = df.sample(n=1000, random_state=42)
df["parsed"] = df["label.name"].apply(ast.literal_eval)
val_values = [val for tup in df["parsed"] for val in tup]
val_counts = Counter(val_values)
val_counts = dict(sorted(val_counts.items(), key=lambda x: x[1], reverse=True))

# --- Predicted labels ---
with open("../../data/output_final/predicted_labels.txt") as f:
    pred_values = [item for line in f for item in ast.literal_eval(line)]
pred_counts = Counter(pred_values)
pred_counts = dict(sorted(pred_counts.items(), key=lambda x: x[1], reverse=True))

# --- Plot two subplots ---
fig, axes = plt.subplots(1, 2, figsize=(16, 16), sharey=True)  # side by side, same y-axis

# Validation plot
axes[0].barh(list(val_counts.keys()), list(val_counts.values()), color="skyblue", edgecolor="black")
axes[0].set_title("Test Label Frequencies")
axes[0].set_xlabel("Frequency")
axes[0].set_ylabel("Labels")

# Predicted plot
axes[1].barh(list(pred_counts.keys()), list(pred_counts.values()), color="salmon", edgecolor="black")
axes[1].set_title("Predicted Label Frequencies")
axes[1].set_xlabel("Frequency")

plt.tight_layout()
plt.savefig("final_histogram.png", dpi=300)