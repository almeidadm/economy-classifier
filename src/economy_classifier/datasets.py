"""Dataset splitting and balancing utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

MULTICLASS_TOP7: tuple[str, ...] = (
    "poder",
    "colunas",
    "mercado",
    "esporte",
    "mundo",
    "cotidiano",
    "ilustrada",
)
OTHER_LABEL: str = "outros"


def build_train_val_test_split(
    dataframe: pd.DataFrame,
    *,
    label_column: str = "label",
    seed: int = 2026,
    test_size: float = 0.10,
    val_size: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a corpus into train/val/test (3-way stratified, 80/10/10 default).

    Both ``test_size`` and ``val_size`` are fractions of the **total** corpus
    (not of the remainder), so defaults of 0.10/0.10 yield 80/10/10.
    The test set is split first; validation is then carved from the remainder.
    """
    if test_size + val_size >= 1.0:
        raise ValueError(
            f"test_size + val_size must be < 1.0, got {test_size} + {val_size}"
        )

    train_val, test = train_test_split(
        dataframe,
        test_size=test_size,
        stratify=dataframe[label_column],
        random_state=seed,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val[label_column],
        random_state=seed,
    )
    return train, val, test


def build_balanced_training_frame(
    dataframe: pd.DataFrame,
    *,
    label_column: str = "label",
    seed: int = 2026,
) -> pd.DataFrame:
    """Downsample the majority class to match the minority class count.

    Legacy: kept for reproducibility of the original 64/16/20 + balanced runs.
    The current pipeline (80/10/10 + 10-fold CV) does not balance training.
    """
    counts = dataframe[label_column].value_counts()
    minority_count = counts.min()
    frames = []
    for label in counts.index:
        subset = dataframe[dataframe[label_column] == label]
        frames.append(subset.sample(n=minority_count, random_state=seed))
    return pd.concat(frames).sample(frac=1, random_state=seed)


def build_cv_folds(
    dataframe: pd.DataFrame,
    *,
    label_column: str = "label",
    n_folds: int = 5,
    seed: int = 2026,
) -> list[dict[str, list]]:
    """Build *n_folds* stratified CV folds over *dataframe* (preserves index labels).

    Each fold is a dict with two keys:
      - ``train_indices``: list of original index labels for that fold's train set
      - ``val_indices``: list of original index labels for that fold's val set

    Designed for the 90% pool (train ∪ val) when the test set is a separate
    held-out split. The returned structure is JSON-serializable.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    indices = np.array(dataframe.index.tolist())
    labels = dataframe[label_column].to_numpy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds: list[dict[str, list]] = []
    for train_pos, val_pos in skf.split(indices, labels):
        folds.append({
            "train_indices": indices[train_pos].tolist(),
            "val_indices": indices[val_pos].tolist(),
        })
    return folds


def map_to_multiclass(
    categories: pd.Series,
    *,
    top_classes: Iterable[str] = MULTICLASS_TOP7,
    other_label: str = OTHER_LABEL,
) -> pd.Series:
    """Map a Folha `category` series to a 7+other multiclass label."""
    keep = set(top_classes)
    mapped = categories.where(categories.isin(keep), other_label)
    return mapped.astype("string")


def attach_multiclass_label(
    split_df: pd.DataFrame,
    *,
    category_column: str = "category",
    multiclass_column: str = "label_multi",
    top_classes: Iterable[str] = MULTICLASS_TOP7,
    other_label: str = OTHER_LABEL,
) -> pd.DataFrame:
    """Return a copy of *split_df* with a multiclass label column attached."""
    if category_column not in split_df.columns:
        raise KeyError(
            f"Column '{category_column}' not found in split. "
            f"Available: {list(split_df.columns)}"
        )
    out = split_df.copy()
    out[multiclass_column] = map_to_multiclass(
        out[category_column],
        top_classes=top_classes,
        other_label=other_label,
    )
    return out
