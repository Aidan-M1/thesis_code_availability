"""
model_lineplot.py

Creates a lineplot of the results of the baseline models against number
of images used in training.

Author: Aidan Murray
Date: 2025-09-26
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df_f1 = pd.read_csv("model_performance(F1_scores).csv")
df_std = pd.read_csv("model_performance(STD).csv")

df_f1 = df_f1[['Num Images', '3B Base model', '7B Base model']]

plt.figure(figsize=(8, 6))
sns.lineplot(data=df_f1, x="Num Images", y="3B Base model", label="3B", color="skyblue")
sns.lineplot(data=df_f1, x="Num Images", y="7B Base model", label="7B", color="salmon")

plt.xlabel("Number of Images", fontsize=12, fontweight="bold")
plt.ylabel("F1 Score", fontsize=12, fontweight="bold")
plt.title("F1 Score per number of images\nused in training", fontsize=15, fontweight="bold")
plt.legend(title="Model Size", loc="upper left")

plt.xticks(np.arange(0, 6501, 1300))
plt.xlim(0, 7000)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("model_regression.png", dpi=300)