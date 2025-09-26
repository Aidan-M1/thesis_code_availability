"""
csv_combine.py

Combines all csv files from squidle+ to one dataset.

Author: Aidan Murray
Date: 2025-09-26
"""

import os
import pandas as pd

folder_path = '../datasets'

csv_files = [
    file for file in os.listdir(folder_path) if file.endswith('.csv')
    ]

dataframes = []

# Loop through the list of CSV files and read each one
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)

combined_df.to_csv('../combined.csv', index=False)