"""Dataset splitting and balancing utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def build_train_val_test_split(
    dataframe: pd.DataFrame,
    *,
    label_column: str = "label",
    seed: int = 42,
    test_size: float = 0.20,
    val_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a corpus into train, validation and test sets (3-way stratified).

    First separates *test_size* for the test set, then splits the remainder
    into train and validation using *val_size* (relative to the remainder).
    """
    train_val, test = train_test_split(
        dataframe,
        test_size=test_size,
        stratify=dataframe[label_column],
        random_state=seed,
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val[label_column],
        random_state=seed,
    )
    return train, val, test


def build_balanced_training_frame(
    dataframe: pd.DataFrame,
    *,
    label_column: str = "label",
    seed: int = 42,
) -> pd.DataFrame:
    """Downsample the majority class to match the minority class count."""
    counts = dataframe[label_column].value_counts()
    minority_count = counts.min()
    frames = []
    for label in counts.index:
        subset = dataframe[dataframe[label_column] == label]
        frames.append(subset.sample(n=minority_count, random_state=seed))
    return pd.concat(frames).sample(frac=1, random_state=seed)
