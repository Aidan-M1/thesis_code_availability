"""
table.py

Creates a table from per_class_metrics.

Author: Aidan Murray
Date: 2025-09-26
"""

import pandas as pd
import dataframe_image as dfi
import math

df = pd.read_csv("per_class_f1.csv", index_col=0)
df = df.set_index("Label").sort_values(by="Support", ascending=False).round(3)

rows_per_page = math.ceil(len(df) / 2)

# Loop through chunks
for i in range(2):
    start = i * rows_per_page
    end = (i + 1) * rows_per_page
    chunk = df.iloc[start:end]

    styled = (
        chunk.style
        .set_caption(f"Per-Class Metrics (Part {i+1})")
        .set_table_styles(
            [
                {"selector": "caption",
                 "props": [("font-size", "16px"), ("text-align", "center")]},
            ]
        )
        .format(precision=3)
        .background_gradient(subset=["Precision", "Recall", "F1-Score"], cmap="Blues")
    )
    dfi.export(styled, f"table_part_{i+1}.png", max_rows=-1, max_cols=-1)