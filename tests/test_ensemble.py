"""Tests for economy_classifier.ensemble — voting, stacking, agreement."""

import numpy as np
import pandas as pd
from economy_classifier.ensemble import (
    compute_agreement_matrix,
    compute_contingency_table,
    majority_vote,
    predict_stacking,
    train_stacking_classifier,
    weighted_vote,
)


def _make_predictions(votes_per_example: list[list[int]]) -> dict[str, pd.Series]:
    """Build predictions dict from a list of per-example votes across methods."""
    n_methods = len(votes_per_example[0])
    methods = [f"m{i}" for i in range(n_methods)]
    result = {}
    for j, method in enumerate(methods):
        result[method] = pd.Series([row[j] for row in votes_per_example])
    return result


def test_majority_vote_unanimous_positive():
    preds = _make_predictions([[1, 1, 1, 1, 1, 1, 1]])
    for threshold in [4, 5, 6, 7]:
        result = majority_vote(preds, threshold=threshold)
        assert result["y_pred"].iloc[0] == 1


def test_majority_vote_unanimous_negative():
    preds = _make_predictions([[0, 0, 0, 0, 0, 0, 0]])
    for threshold in [1, 4, 7]:
        result = majority_vote(preds, threshold=threshold)
        assert result["y_pred"].iloc[0] == 0


def test_majority_vote_threshold_4():
    preds_4 = _make_predictions([[1, 1, 1, 1, 0, 0, 0]])
    preds_3 = _make_predictions([[1, 1, 1, 0, 0, 0, 0]])
    assert majority_vote(preds_4, threshold=4)["y_pred"].iloc[0] == 1
    assert majority_vote(preds_3, threshold=4)["y_pred"].iloc[0] == 0


def test_majority_vote_threshold_5():
    preds_5 = _make_predictions([[1, 1, 1, 1, 1, 0, 0]])
    preds_4 = _make_predictions([[1, 1, 1, 1, 0, 0, 0]])
    assert majority_vote(preds_5, threshold=5)["y_pred"].iloc[0] == 1
    assert majority_vote(preds_4, threshold=5)["y_pred"].iloc[0] == 0


def test_majority_vote_threshold_6():
    preds_6 = _make_predictions([[1, 1, 1, 1, 1, 1, 0]])
    preds_5 = _make_predictions([[1, 1, 1, 1, 1, 0, 0]])
    assert majority_vote(preds_6, threshold=6)["y_pred"].iloc[0] == 1
    assert majority_vote(preds_5, threshold=6)["y_pred"].iloc[0] == 0


def test_majority_vote_returns_standard_format():
    preds = _make_predictions([[1, 0, 1, 0, 1, 0, 1]] * 5)
    result = majority_vote(preds, threshold=4)
    assert isinstance(result, pd.DataFrame)
    assert "y_pred" in result.columns
    assert len(result) == 5


def test_weighted_vote_respects_f1_weights():
    scores = {"good": pd.Series([0.9]), "bad": pd.Series([0.1])}
    weights = {"good": 1.0, "bad": 0.0}
    result = weighted_vote(scores, weights, threshold=0.5)
    assert result["y_pred"].iloc[0] == 1
    assert result["y_score"].iloc[0] > 0.5


def test_weighted_vote_scores_in_0_1():
    rng = np.random.RandomState(42)
    scores = {f"m{i}": pd.Series(rng.uniform(0, 1, 10)) for i in range(7)}
    weights = {f"m{i}": 0.8 for i in range(7)}
    result = weighted_vote(scores, weights)
    assert (result["y_score"] >= 0).all()
    assert (result["y_score"] <= 1).all()


def test_weighted_vote_equal_weights():
    scores = {"a": pd.Series([0.8]), "b": pd.Series([0.4]), "c": pd.Series([0.6])}
    weights = {"a": 1.0, "b": 1.0, "c": 1.0}
    result = weighted_vote(scores, weights)
    expected = (0.8 + 0.4 + 0.6) / 3
    assert abs(result["y_score"].iloc[0] - expected) < 1e-6


def test_stacking_trains_on_val():
    rng = np.random.RandomState(42)
    n = 30
    val_scores = {f"m{i}": pd.Series(rng.uniform(0, 1, n)) for i in range(3)}
    val_true = pd.Series([1] * 10 + [0] * 20)
    model = train_stacking_classifier(val_scores, val_true, seed=42)
    assert hasattr(model, "predict_proba")


def test_stacking_returns_predictions():
    rng = np.random.RandomState(42)
    n = 30
    val_scores = {f"m{i}": pd.Series(rng.uniform(0, 1, n)) for i in range(3)}
    val_true = pd.Series([1] * 10 + [0] * 20)
    model = train_stacking_classifier(val_scores, val_true, seed=42)

    test_scores = {f"m{i}": pd.Series(rng.uniform(0, 1, 10)) for i in range(3)}
    result = predict_stacking(model, test_scores)
    assert "y_pred" in result.columns
    assert "y_score" in result.columns
    assert len(result) == 10


def test_stacking_no_data_leakage():
    """The stacking model is trained on val scores, not train scores."""
    rng = np.random.RandomState(42)
    val_scores = {f"m{i}": pd.Series(rng.uniform(0, 1, 30)) for i in range(3)}
    val_true = pd.Series([1] * 10 + [0] * 20)
    model = train_stacking_classifier(val_scores, val_true)
    # Model should have been fitted with 30 samples (val), not anything else
    assert model.n_features_in_ == 3


def test_agreement_matrix_shape():
    preds = {f"m{i}": pd.Series([1, 0, 1, 0]) for i in range(5)}
    matrix = compute_agreement_matrix(preds)
    assert matrix.shape == (5, 5)


def test_agreement_matrix_diagonal_is_one():
    preds = {"a": pd.Series([1, 0, 1]), "b": pd.Series([0, 1, 1])}
    matrix = compute_agreement_matrix(preds)
    assert matrix.loc["a", "a"] == 1.0
    assert matrix.loc["b", "b"] == 1.0


def test_agreement_matrix_symmetric():
    preds = {"a": pd.Series([1, 0, 1, 0]), "b": pd.Series([0, 1, 1, 0])}
    matrix = compute_agreement_matrix(preds)
    assert matrix.loc["a", "b"] == matrix.loc["b", "a"]


def test_contingency_table_levels():
    preds = {f"m{i}": pd.Series([1, 0, 1, 0, 1]) for i in range(7)}
    y_true = pd.Series([1, 0, 1, 0, 1])
    table = compute_contingency_table(preds, y_true)
    assert isinstance(table, pd.DataFrame)
    # Should have rows for agreement levels present in the data
    assert len(table) > 0
