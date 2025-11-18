"""Reusable evaluation utilities."""

from __future__ import annotations

import pandas as pd


def per_class_accuracy(frame: pd.DataFrame, label_col: str, pred_col: str) -> pd.DataFrame:
    """Return per-class accuracy / error counts.

    Parameters
    ----------
    frame:
        DataFrame containing ground-truth and prediction columns.
    label_col / pred_col:
        Column names with categorical labels.
    """

    df = frame[[label_col, pred_col]].copy()
    df["correct"] = df[label_col] == df[pred_col]

    counts = df.groupby(label_col).agg(total=(label_col, "count"))
    correct = (
        df[df["correct"]]
        .groupby(label_col)
        .agg(true_count=(label_col, "count"))
    )
    incorrect = (
        df[~df["correct"]]
        .groupby(label_col)
        .agg(false_count=(label_col, "count"))
    )

    merged = counts.join(correct, how="left").join(incorrect, how="left")
    merged = merged.fillna(0)
    merged["true_percent"] = (merged["true_count"] / merged["total"]) * 100
    merged["false_percent"] = (merged["false_count"] / merged["total"]) * 100
    return merged.reset_index().rename(columns={label_col: "label"})


def misclassification_pairs(frame: pd.DataFrame, label_col: str, pred_col: str) -> pd.DataFrame:
    """Return counts of (label, pred) pairs for misclassified samples."""

    df = frame[[label_col, pred_col]].copy()
    df = df[df[label_col] != df[pred_col]]
    if df.empty:
        return pd.DataFrame(columns=["label", "pred", "count"])
    df["count"] = 1
    agg = df.groupby([label_col, pred_col]).agg({"count": "count"}).reset_index()
    return agg.rename(columns={label_col: "label", pred_col: "pred"})
