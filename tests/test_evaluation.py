"""Tests for economy_classifier.evaluation — metrics, McNemar, AUC-ROC."""

import numpy as np
import pandas as pd
from economy_classifier.evaluation import (
    compute_binary_metrics,
    compute_mcnemar_test,
    compute_roc_auc,
)


def test_metrics_known_values(known_predictions):
    y_true, y_pred, _ = known_predictions
    m = compute_binary_metrics(y_true, y_pred)
    assert abs(m["precision"] - 2 / 3) < 1e-3
    assert abs(m["recall"] - 2 / 3) < 1e-3
    assert abs(m["f1"] - 2 / 3) < 1e-3
    assert abs(m["accuracy"] - 0.75) < 1e-3


def test_metrics_all_correct():
    y = pd.Series([1, 0, 1, 0])
    m = compute_binary_metrics(y, y)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["accuracy"] == 1.0


def test_metrics_all_wrong():
    y_true = pd.Series([1, 1, 0, 0])
    y_pred = pd.Series([0, 0, 1, 1])
    m = compute_binary_metrics(y_true, y_pred)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_metrics_no_positives_predicted():
    y_true = pd.Series([1, 1, 0, 0])
    y_pred = pd.Series([0, 0, 0, 0])
    m = compute_binary_metrics(y_true, y_pred)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_mcnemar_identical_predictions():
    y_true = pd.Series([1, 0, 1, 0, 1, 0])
    y_pred = pd.Series([1, 0, 0, 0, 1, 1])
    result = compute_mcnemar_test(y_true, y_pred, y_pred)
    assert result["p_value"] == 1.0


def test_mcnemar_opposite_predictions():
    y_true = pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_pred_a = pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # perfect
    y_pred_b = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # all wrong
    result = compute_mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert result["p_value"] < 0.05


def test_mcnemar_returns_expected_keys():
    y_true = pd.Series([1, 0, 1, 0])
    y_pred = pd.Series([1, 0, 0, 0])
    result = compute_mcnemar_test(y_true, y_pred, y_pred)
    assert {"chi2", "p_value", "significant_at_005"}.issubset(result.keys())
    assert isinstance(result["significant_at_005"], bool)


def test_roc_auc_perfect_scores():
    y_true = pd.Series([1, 1, 0, 0])
    y_score = pd.Series([0.9, 0.8, 0.1, 0.2])
    assert compute_roc_auc(y_true, y_score) == 1.0


def test_roc_auc_random_scores():
    rng = np.random.RandomState(42)
    y_true = pd.Series([1] * 500 + [0] * 500)
    y_score = pd.Series(rng.uniform(0, 1, 1000))
    auc = compute_roc_auc(y_true, y_score)
    assert 0.4 <= auc <= 0.6


def test_roc_auc_requires_continuous_scores():
    y_true = pd.Series([1, 0, 1, 0])
    y_score = pd.Series([0.95, 0.12, 0.88, 0.05])
    auc = compute_roc_auc(y_true, y_score)
    assert isinstance(auc, float)


def test_compute_binary_metrics_returns_rounded_floats(known_predictions):
    y_true, y_pred, _ = known_predictions
    m = compute_binary_metrics(y_true, y_pred)
    for key in ["precision", "recall", "f1", "accuracy"]:
        val = m[key]
        assert isinstance(val, float)
        # At most 4 decimal places
        assert val == round(val, 4)
