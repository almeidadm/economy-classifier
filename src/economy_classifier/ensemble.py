"""Ensemble strategies: voting, stacking and agreement analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score


def majority_vote(
    predictions: dict[str, pd.Series],
    *,
    threshold: int = 4,
) -> pd.DataFrame:
    """Binary majority vote across classifiers.

    A sample is positive when at least *threshold* classifiers predict 1.
    """
    votes = pd.DataFrame(predictions)
    vote_sum = votes.sum(axis=1)
    y_pred = (vote_sum >= threshold).astype(int)
    return pd.DataFrame({"y_pred": y_pred})


def weighted_vote(
    scores: dict[str, pd.Series],
    weights: dict[str, float],
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Weighted average of continuous scores with F1-based weights."""
    score_df = pd.DataFrame(scores)
    w = np.array([weights[m] for m in score_df.columns])
    w_sum = w.sum()
    if w_sum == 0:
        w_normalized = np.ones_like(w) / len(w)
    else:
        w_normalized = w / w_sum

    y_score = score_df.values @ w_normalized
    y_pred = (y_score >= threshold).astype(int)

    return pd.DataFrame({
        "y_pred": y_pred,
        "y_score": np.round(y_score, 4),
    })


def train_stacking_classifier(
    val_scores: dict[str, pd.Series],
    val_true: pd.Series,
    *,
    seed: int = 42,
) -> LogisticRegression:
    """Train a meta-classifier on validation-set scores (no leakage)."""
    X = pd.DataFrame(val_scores).values
    y = np.asarray(val_true)
    clf = LogisticRegression(random_state=seed, solver="lbfgs", max_iter=1000)
    clf.fit(X, y)
    return clf


def predict_stacking(
    model: LogisticRegression,
    test_scores: dict[str, pd.Series],
) -> pd.DataFrame:
    """Predict using the stacking meta-classifier."""
    X = pd.DataFrame(test_scores).values
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]
    return pd.DataFrame({
        "y_pred": y_pred.tolist(),
        "y_score": np.round(y_score, 4).tolist(),
    })


def compute_agreement_matrix(
    predictions: dict[str, pd.Series],
) -> pd.DataFrame:
    """NxN matrix of Cohen's Kappa between every pair of classifiers."""
    methods = list(predictions.keys())
    n = len(methods)
    matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            kappa = cohen_kappa_score(predictions[methods[i]], predictions[methods[j]])
            matrix[i, j] = round(kappa, 4)
            matrix[j, i] = round(kappa, 4)

    return pd.DataFrame(matrix, index=methods, columns=methods)


def compute_contingency_table(
    predictions: dict[str, pd.Series],
    y_true: pd.Series,
) -> pd.DataFrame:
    """Agreement level (1..N methods) vs actual class contingency table."""
    votes = pd.DataFrame(predictions)
    agreement_level = votes.sum(axis=1)
    y_true = pd.Series(np.asarray(y_true), name="y_true")
    agreement_level.name = "agreement"

    table = pd.crosstab(agreement_level, y_true, margins=True)
    return table
