"""
model_7B_stats.py

Performs a friedman test, pairwise wilcoxon tests with
Benjamini-Hochberg corrections, and produces a heatmap of p-values
for each of the different 7B fine-tunes

Author: Aidan Murray
Date: 2025-09-26
"""

import pandas as pd
from pathlib import Path
from scipy import stats
import numpy as np
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from statsmodels.stats.multitest import multipletests

WD = Path("./7B")
FILENAME = "evals.csv"

# gather evals from each folder into one dataframe
df = pd.DataFrame()
for item in WD.iterdir():
    if not item.is_dir():
        continue
    eval_path = item / FILENAME
    evals = pd.read_csv(eval_path).set_index("ID")
    df[item.name] = evals['F1 Score']
df.dropna(inplace=True)

stat, p = stats.friedmanchisquare(*[df[col] for col in df.columns])
print("Friedman test statistic:", stat)
print("p-value:", p)
print()

categories = df.columns
results = []

for cat1, cat2 in combinations(categories, 2):
    if cat1 == cat2:
        continue
    stat, p = stats.wilcoxon(df[cat1], df[cat2])
    results.append((cat1, cat2, p))

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
order = [
    "Untrained", "Baseline", "720p", 
    "Numerical Contex", "Hierarchical", "Ecoregions Only"
    ]

tri = tri.reindex(index=order, columns=order)
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
plt.title("Pairwise adjusted p-values\nfor F1 Scores on the\n7B models")
plt.tight_layout()
plt.savefig("p_vals.png")