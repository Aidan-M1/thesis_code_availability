"""
prompt_barplots.py

Creates a barplot of the results of model predictions for each prompt.

Author: Aidan Murray
Date: 2025-09-26
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("prompt_tests.csv", index_col=0)
df.columns = df.columns.astype(str)

bar_color = "skyblue"
bar_color_2 = "salmon"

ste_f1 = df.loc["SD F1 Score"] / (300 ** 0.5)
ste_time = df.loc["SD execution times (s)"] / (300 ** 0.5)

fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes = axes.flatten()

axes[0].bar(df.columns, df.loc["Mean F1 Score"], yerr=ste_f1, color=bar_color, edgecolor="black")
axes[0].set_title("Mean F1 Score", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Mean F1 Score", fontweight="bold")
axes[0].set_xticklabels(df.columns, rotation=45, ha="right")

axes[1].bar(df.columns, df.loc["Mean Execution Times (s)"], yerr=ste_time, color=bar_color, edgecolor="black")
axes[1].set_title("Mean Execution Time", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Mean Execution\nTime (s)", fontweight="bold")
axes[1].set_xticklabels(df.columns, rotation=45, ha="right")

axes[2].bar(df.columns, df.loc["Failed Parses"], color=bar_color_2, edgecolor="black")
axes[2].set_title("Number of\nFailed Parses", fontsize=12, fontweight="bold")
axes[2].set_ylabel("Count", fontweight="bold")
axes[2].set_xticklabels(df.columns, rotation=45, ha="right")

axes[3].bar(df.columns, df.loc["Timeouts"], color=bar_color_2, edgecolor="black")
axes[3].set_title("Number of\nTimeouts", fontsize=12, fontweight="bold")
axes[3].set_ylabel("Count", fontweight="bold")
axes[3].set_xticklabels(df.columns, rotation=45, ha="right")

fig.suptitle("Bar Plots of Different Metrics\nFor Each Prompt", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
plt.savefig("prompts.png", dpi=300)
