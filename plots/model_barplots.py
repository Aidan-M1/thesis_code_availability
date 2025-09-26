"""
model_barplots.py

Creates barplots of the results generated from each model.

Author: Aidan Murray
Date: 2025-09-26
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df_f1 = pd.read_csv("model_performance(F1_scores).csv").set_index('Num Images')
df_std = pd.read_csv("model_performance(STD).csv").set_index('Num Images')  / (300 ** 0.5)

# Select row for 6500 images
row_mean = df_f1.loc[6500].copy()
row_se = df_std.loc[6500].copy()

row_mean["3B Untrained"] = df_f1.loc[0, "3B Baseline"]
row_mean["7B Untrained"] = df_f1.loc[0, "7B Baseline"]

row_se["3B Untrained"] = df_std.loc[0, "3B Baseline"]
row_se["7B Untrained"] = df_std.loc[0, "3B Baseline"]

# Convert to tidy (long) format
df_plot = pd.DataFrame({
    "Model": row_mean.index,
    "F1": row_mean.values,
    "SE": row_se.values
}).dropna()

order = ["Untrained", "Baseline", "720p", 
         "Numerical Context", "Hierarchical (R/P/E)", "Ecoregions only"]

# Add a "Family" column (3B or 7B) based on the model name
df_plot["Family"] = df_plot["Model"].apply(lambda x: "3B" if x.startswith("3B") else "7B")

# Remove "3B " or "7B " prefix for pairing
df_plot["Category"] = df_plot["Model"].str.replace(r"^[37]B ", "", regex=True)



# --- Plot ---
plt.figure(figsize=(12, 7))
sns.barplot(
    data=df_plot,
    x="Category", y="F1", hue="Family",
    palette={"3B": "skyblue", "7B": "salmon"},
    edgecolor="black",
    order=order,
)

# Now add error bars manually
for i, row in df_plot.iterrows():
    x_pos = order.index(row["Category"])
    offset = -0.2 if row["Family"] == "3B" else 0.2  # adjust left/right
    plt.errorbar(
        x=x_pos + offset,
        y=row["F1"],
        yerr=row["SE"],
        fmt="none",
        ecolor="black",
        capsize=4,
        lw=1
    )

plt.xlabel("Model Type", fontsize=12, fontweight="bold", labelpad=20)
plt.ylabel("F1 Score", fontsize=12, fontweight="bold")
plt.title("F1 Score of Different\nTraining Approaches (6500 Images)", fontsize=15, fontweight="bold")
plt.legend(title="Model Size", loc="upper left")
plt.tight_layout()
plt.savefig("model_comparisons_grouped.png", dpi=300)