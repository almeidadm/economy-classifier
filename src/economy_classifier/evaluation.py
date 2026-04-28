"""Evaluation metrics, McNemar test and AUC-ROC."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
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


def compute_brier_score(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
) -> float:
    """Brier score for binary predictions: mean squared error between score and label.

    ``brier = mean((y_score - y_true)**2)``. Lower is better (0.0 = perfect).

    Meaningful only when ``y_score`` is a calibrated probability (TF-IDF +
    CalibratedClassifierCV, BERT softmax). For LLMs with deterministic
    ``y_score in {0.0, 1.0}``, Brier collapses to ``1 - accuracy``; report it
    but interpret with the caveat that it does not measure calibration there.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_score_arr = np.asarray(y_score, dtype=float)
    if len(y_true_arr) == 0:
        return 0.0
    return round(float(np.mean((y_score_arr - y_true_arr) ** 2)), 4)


def compute_ece(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (binary) with equal-width binning.

    ``ECE = sum_b (n_b / n) * |acc_b - conf_b|`` where ``acc_b`` is the
    positive fraction in bin ``b`` and ``conf_b`` is the mean predicted score.
    Lower is better (0.0 = perfectly calibrated).

    Meaningful only when ``y_score`` covers a continuous range. For LLMs with
    deterministic scores, almost all mass lands in two bins (0.0 and 1.0) —
    the ECE then approximates miscalibration relative to a hard 0/1 forecaster
    and is mostly redundant with ``1 - accuracy``.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_score_arr = np.asarray(y_score, dtype=float)
    n = len(y_true_arr)
    if n == 0:
        return 0.0
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize against interior edges so scores in [edges[b], edges[b+1]) map to b,
    # with the right boundary inclusive for the final bin.
    bin_idx = np.clip(np.digitize(y_score_arr, edges[1:-1]), 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        acc_b = float(y_true_arr[mask].mean())
        conf_b = float(y_score_arr[mask].mean())
        ece += (n_b / n) * abs(acc_b - conf_b)
    return round(float(ece), 4)


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


def compute_multiclass_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    labels: list[str] | None = None,
) -> dict[str, float | dict[str, float]]:
    """Macro-F1, weighted-F1, accuracy and per-class F1 (rounded to 4 decimals)."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if labels is None:
        labels = sorted(set(y_true_arr.tolist()) | set(y_pred_arr.tolist()))

    per_class = f1_score(
        y_true_arr, y_pred_arr,
        labels=labels, average=None, zero_division=0,
    )
    return {
        "macro_f1": round(float(f1_score(
            y_true_arr, y_pred_arr,
            labels=labels, average="macro", zero_division=0,
        )), 4),
        "weighted_f1": round(float(f1_score(
            y_true_arr, y_pred_arr,
            labels=labels, average="weighted", zero_division=0,
        )), 4),
        "accuracy": round(float(accuracy_score(y_true_arr, y_pred_arr)), 4),
        "per_class_f1": {
            label: round(float(score), 4)
            for label, score in zip(labels, per_class, strict=True)
        },
    }


def compute_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    labels: list[str],
    normalize: str | None = "true",
) -> pd.DataFrame:
    """Confusion matrix as DataFrame (index=true, columns=pred)."""
    matrix = confusion_matrix(
        np.asarray(y_true), np.asarray(y_pred),
        labels=labels, normalize=normalize,
    )
    return pd.DataFrame(matrix, index=labels, columns=labels)


# ---------------------------------------------------------------------------
# Cost / cross-validation aggregation helpers (Fase 2 — comparacao justa)
# ---------------------------------------------------------------------------


def summarize_cv_metrics(fold_metrics: list[dict]) -> dict:
    """Aggregate per-fold metric dicts into ``{key}_mean`` and ``{key}_std``.

    Nested dicts (e.g. ``per_class_f1``) are aggregated recursively.
    Non-numeric values are kept from the first fold (for things like label lists).
    """
    if not fold_metrics:
        return {}

    keys = fold_metrics[0].keys()
    summary: dict[str, object] = {}
    for key in keys:
        values = [fold[key] for fold in fold_metrics]
        if isinstance(values[0], dict):
            summary[key] = summarize_cv_metrics(values)
        elif isinstance(values[0], (int, float)) and not isinstance(values[0], bool):
            arr = np.array(values, dtype=float)
            summary[f"{key}_mean"] = round(float(arr.mean()), 4)
            summary[f"{key}_std"] = round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4)
        else:
            summary[key] = values[0]
    return summary


def compute_cost_metrics(
    *,
    train_seconds: float | list[float],
    inference_seconds: float | list[float],
    n_inference_samples: int,
    model_size_mb: float | None = None,
    n_parameters: int | None = None,
    hardware: str = "CPU",
) -> dict:
    """Build a standardized cost dict. Lists are aggregated as mean/std (CV)."""
    def _stats(value: float | list[float]) -> tuple[float, float]:
        if isinstance(value, list):
            arr = np.array(value, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        else:
            mean, std = float(value), 0.0
        return round(mean, 4), round(std, 4)

    train_mean, train_std = _stats(train_seconds)
    inf_mean, inf_std = _stats(inference_seconds)
    throughput = (
        round(n_inference_samples / inf_mean, 2) if inf_mean > 0 else None
    )

    return {
        "train_seconds_mean": train_mean,
        "train_seconds_std": train_std,
        "inference_seconds_mean": inf_mean,
        "inference_seconds_std": inf_std,
        "throughput_samples_per_second": throughput,
        "model_size_mb": (
            round(float(model_size_mb), 3) if model_size_mb is not None else None
        ),
        "n_parameters": int(n_parameters) if n_parameters is not None else None,
        "hardware": hardware,
    }
