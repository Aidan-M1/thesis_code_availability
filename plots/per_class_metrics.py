"""
per_class_metrics.py

Creates a csv file with per class precision, recall, F1, and support
from the results of model predictions.

Author: Aidan Murray
Date: 2025-09-26
"""

from pathlib import Path
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, precision_recall_fscore_support

DATA_FOLDER = Path("../../data")
TEST_PATH = DATA_FOLDER / "ecoregions" / "test.csv"
PRED_PATH = DATA_FOLDER / "output_final" / "predicted_labels.txt"
EVALS = DATA_FOLDER / "output_final" / "evals.csv"

with open(PRED_PATH) as f:
    y_pred = [literal_eval(line) for line in f]

df = pd.read_csv(EVALS).set_index("ID")
df['y_pred'] = y_pred

df_test = pd.read_csv(TEST_PATH).sample(n=1200, random_state=42).set_index("point.media.id")
df = df.join(df_test[['label.name']], how="inner")
df['y_true'] = df['label.name'].apply(literal_eval)
df = df.drop(columns=['label.name'])

mlb = MultiLabelBinarizer()
mlb.fit(df["y_true"].tolist() + df["y_pred"].tolist())

y_true_bin = mlb.transform(df["y_true"])
y_pred_bin = mlb.transform(df["y_pred"])

print(classification_report(
    y_true_bin,
    y_pred_bin,
    target_names=mlb.classes_,
    zero_division=0
))

precision, recall, f1, support = precision_recall_fscore_support(
    y_true_bin,
    y_pred_bin,
    average=None
)

per_class_df = pd.DataFrame({
    "Label": mlb.classes_,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "Support": support
})

print(per_class_df.sort_values("Support", ascending=False).head())

per_class_df.to_csv("per_class_f1.csv")