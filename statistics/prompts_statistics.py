"""
prompt_statistics.py

Performs a friedman test, parwise wilcoxon test with benjamini-hochberg
correction, and produces a heatmap of p-values for each prompt.

Author: Aidan Murray
Date: 2025-09-26
"""

import pandas as pd
import scipy.stats as stats
from itertools import combinations
import numpy as np
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

df = pd.read_csv("../data/output/prompt_test_1/prompt_evals.csv", index_col=0)

friedman = stats.friedmanchisquare(df['Prompt 0'], df['Prompt 1'], df['Prompt 2'], df['Prompt 3'], df['Prompt 4'])
print(friedman)

rename_map = {
    "Prompt 0": "Basic",
    "Prompt 1": "Emphasis",
    "Prompt 2": "Context",
    "Prompt 3": "Few-Shot",
    "Prompt 4": "CoT"
}
df = df.rename(columns=rename_map)

prompts = df.columns
results = []

for prompt1, prompt2 in combinations(prompts, 2):
    stat, p = stats.wilcoxon(df[prompt1], df[prompt2])
    results.append((prompt1, prompt2, p))

groups = df.columns
matrix = pd.DataFrame(1.0, index=groups, columns=groups)

pvals = [p for _, _, p in results]
reject, p_adj, _, _ = multipletests(pvals, method="fdr_bh")

for (cat1, cat2, _), p_corr in zip(results, p_adj):
    matrix.loc[cat1, cat2] = p_corr
    matrix.loc[cat2, cat1] = p_corr

cmap = ListedColormap(["tomato", "skyblue"]) 
norm = BoundaryNorm([0, 0.05, 1], ncolors=cmap.N, clip=True)

tri = matrix.copy()
tri.values[np.triu_indices_from(tri)] = np.nan
tri = tri.dropna(axis=0, how="all")
tri = tri.dropna(axis=1, how="all")

plt.figure(figsize=(8,6))
sns.heatmap(tri, 
            annot=True, 
            cmap=cmap, 
            norm=norm, 
            vmin=0, vmax=1,
            linewidths=0.5,      
            linecolor="white",    
            cbar_kws={"label": "Adjusted p-value"})
plt.yticks(rotation=0, ha="right")
plt.title("Pairwise adjusted p-values\nfor F1 Scores on the untrained model\nfor each prompt")
plt.tight_layout()
plt.savefig("p_vals.png")