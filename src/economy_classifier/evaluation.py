"""Evaluation metrics, McNemar test and AUC-ROC."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute precision, recall, F1 and accuracy, rounded to 4 decimals."""
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }


def compute_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Compute ROC AUC from continuous scores."""
    return float(roc_auc_score(y_true, y_score))


def compute_mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict[str, float | bool]:
    """McNemar test comparing two classifiers on the same data.

    Returns dict with chi2, p_value, significant_at_005.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    # Discordant cells: b wrong & a right, a wrong & b right
    b_wrong_a_right = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_right = int(np.sum(~correct_a & correct_b))

    n_discordant = b_wrong_a_right + a_wrong_b_right

    if n_discordant == 0:
        return {"chi2": 0.0, "p_value": 1.0, "significant_at_005": False}

    chi2_stat = (b_wrong_a_right - a_wrong_b_right) ** 2 / n_discordant
    p_value = float(1 - chi2_dist.cdf(chi2_stat, df=1))

    return {
        "chi2": round(float(chi2_stat), 4),
        "p_value": round(p_value, 6),
        "significant_at_005": p_value < 0.05,
    }
