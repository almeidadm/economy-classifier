"""Tests for economy_classifier.evaluation — metrics, McNemar, AUC-ROC."""

import numpy as np
import pandas as pd
from economy_classifier.evaluation import (
    compute_binary_metrics,
    compute_confusion_matrix,
    compute_cost_metrics,
    compute_mcnemar_test,
    compute_multiclass_metrics,
    compute_roc_auc,
    summarize_cv_metrics,
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


def test_multiclass_metrics_perfect():
    y = pd.Series(["a", "b", "c", "a", "b"])
    m = compute_multiclass_metrics(y, y)
    assert m["macro_f1"] == 1.0
    assert m["weighted_f1"] == 1.0
    assert m["accuracy"] == 1.0
    assert all(v == 1.0 for v in m["per_class_f1"].values())


def test_multiclass_metrics_per_class_keys():
    y_true = pd.Series(["a", "a", "b", "b", "c"])
    y_pred = pd.Series(["a", "b", "b", "b", "c"])
    m = compute_multiclass_metrics(y_true, y_pred, labels=["a", "b", "c"])
    assert set(m["per_class_f1"].keys()) == {"a", "b", "c"}


def test_summarize_cv_metrics_flat():
    folds = [
        {"f1": 0.80, "accuracy": 0.85},
        {"f1": 0.82, "accuracy": 0.86},
        {"f1": 0.78, "accuracy": 0.84},
    ]
    summary = summarize_cv_metrics(folds)
    assert summary["f1_mean"] == 0.8
    assert summary["accuracy_mean"] == 0.85
    assert summary["f1_std"] > 0
    assert summary["accuracy_std"] > 0


def test_summarize_cv_metrics_nested():
    folds = [
        {"macro_f1": 0.7, "per_class_f1": {"a": 0.8, "b": 0.6}},
        {"macro_f1": 0.8, "per_class_f1": {"a": 0.9, "b": 0.7}},
    ]
    summary = summarize_cv_metrics(folds)
    assert summary["macro_f1_mean"] == 0.75
    assert summary["per_class_f1"]["a_mean"] == 0.85
    assert summary["per_class_f1"]["b_mean"] == 0.65


def test_summarize_cv_metrics_single_fold_zero_std():
    summary = summarize_cv_metrics([{"f1": 0.9}])
    assert summary["f1_mean"] == 0.9
    assert summary["f1_std"] == 0.0


def test_compute_cost_metrics_scalar():
    cost = compute_cost_metrics(
        train_seconds=10.0, inference_seconds=2.0,
        n_inference_samples=1000, model_size_mb=5.5,
        n_parameters=1_000_000, hardware="CPU",
    )
    assert cost["train_seconds_mean"] == 10.0
    assert cost["train_seconds_std"] == 0.0
    assert cost["throughput_samples_per_second"] == 500.0
    assert cost["model_size_mb"] == 5.5
    assert cost["n_parameters"] == 1_000_000
    assert cost["hardware"] == "CPU"


def test_compute_cost_metrics_lists():
    cost = compute_cost_metrics(
        train_seconds=[10.0, 11.0, 9.0], inference_seconds=[2.0, 2.0, 2.0],
        n_inference_samples=1000,
    )
    assert cost["train_seconds_mean"] == 10.0
    assert cost["train_seconds_std"] > 0
    assert cost["inference_seconds_std"] == 0.0
    assert cost["throughput_samples_per_second"] == 500.0


def test_confusion_matrix_shape_and_normalization():
    y_true = pd.Series(["a", "a", "b", "b", "c"])
    y_pred = pd.Series(["a", "b", "b", "b", "c"])
    cm = compute_confusion_matrix(
        y_true, y_pred, labels=["a", "b", "c"], normalize="true",
    )
    assert cm.shape == (3, 3)
    # rows sum to 1 when normalize="true" (rows where class exists)
    for row in ["a", "b", "c"]:
        assert abs(cm.loc[row].sum() - 1.0) < 1e-6
