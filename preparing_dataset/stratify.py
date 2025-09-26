"""
stratify.py

Stratifies the dataset with iterative stratification into train,
validation, and test data.

Author: Aidan Murray
Date: 2025-09-26
"""

import numpy as np
import pandas as pd
from pathlib import Path
from skmultilearn.model_selection.iterative_stratification \
    import iterative_train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

TEST_SIZE = 0.2
PATH = Path("../../data")

df = pd.read_csv(PATH / "combined.csv")
df = df.dropna(subset=['label.name'])
df = df.sample(frac=1, ignore_index=True, random_state=42)

media = df[[
    'point.media.id',
    'point.media.path_best',
    'point.media.deployment.campaign.name',
    'point.media.timestamp_start',
    'point.pose.lat',
    'point.pose.dep',
    'point.pose.lon',
    ]].drop_duplicates()

labels_per_media = df.groupby('point.media.id')['label.name']\
    .unique().reset_index()
media = media.merge(labels_per_media, on='point.media.id')

mlb = MultiLabelBinarizer()
X, y = media.iloc[:,0:-1].to_numpy(), mlb.fit_transform(media['label.name'])

print("Stratifying train / test split ...")
X_train, y_train, X_test, y_test = iterative_train_test_split(
    X, y, TEST_SIZE, random_state=42
    )

print("Stratifying train / validation split ...")
X_train_partial, y_train_partial, X_val, y_val = iterative_train_test_split(
    X_train, y_train, TEST_SIZE, random_state=42
    )

def rebuild_dataframe(
        X_split, y_split, column_names, label_column_name='label.name'
        ):
    df = pd.DataFrame(X_split, columns=column_names)
    decoded_labels = mlb.inverse_transform(y_split)
    df[label_column_name] = decoded_labels
    return df

columns_without_labels = media.drop(columns=['label.name']).columns

rebuild_dataframe(X_train, y_train, columns_without_labels)\
    .to_csv(PATH / "train_full.csv", index=False)

rebuild_dataframe(X_test, y_test, columns_without_labels)\
    .to_csv(PATH / "test.csv", index=False)

rebuild_dataframe(X_train_partial, y_train_partial, columns_without_labels)\
    .to_csv(PATH / "train_partial.csv", index=False)

rebuild_dataframe(X_val, y_val, columns_without_labels)\
    .to_csv(PATH / "validation.csv", index=False)