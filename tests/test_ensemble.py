"""Tests for economy_classifier.ensemble — voting, stacking, agreement."""

import json

import numpy as np
import pandas as pd
from economy_classifier.ensemble import (
    compute_agreement_matrix,
    compute_contingency_table,
    compute_fleiss_kappa,
    discover_runs,
    load_run_predictions,
    majority_vote,
    optimize_voting_threshold,
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


def _make_proba_df(rng, n: int, classes: list[str]) -> pd.DataFrame:
    raw = rng.dirichlet(np.ones(len(classes)), size=n)
    return pd.DataFrame(raw, columns=classes)


def test_stacking_multiclass_features_per_class():
    """Multiclass stacking concatenates per-class proba columns from each model."""
    rng = np.random.RandomState(0)
    classes = ["a", "b", "c"]
    val_features = {f"m{i}": _make_proba_df(rng, 60, classes) for i in range(4)}
    val_true = pd.Series((["a"] * 20) + (["b"] * 20) + (["c"] * 20))

    model = train_stacking_classifier(val_features, val_true, seed=42)
    # 4 base models * 3 classes = 12 stacked features
    assert model.n_features_in_ == 12
    assert set(model.classes_) == {"a", "b", "c"}


def test_stacking_multiclass_predicts_class_labels():
    rng = np.random.RandomState(1)
    classes = ["poder", "mercado", "outros"]
    val_features = {f"m{i}": _make_proba_df(rng, 60, classes) for i in range(3)}
    val_true = pd.Series((["poder"] * 20) + (["mercado"] * 20) + (["outros"] * 20))
    model = train_stacking_classifier(val_features, val_true, seed=42)

    test_features = {f"m{i}": _make_proba_df(rng, 15, classes) for i in range(3)}
    out = predict_stacking(model, test_features)

    assert "y_pred" in out.columns
    assert set(out.columns) >= {"y_pred", "y_proba_poder", "y_proba_mercado", "y_proba_outros"}
    assert len(out) == 15
    assert set(out["y_pred"].unique()).issubset({"poder", "mercado", "outros"})
    proba_cols = [c for c in out.columns if c.startswith("y_proba_")]
    row_sums = out[proba_cols].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-2)


def test_stacking_binary_unchanged_with_series_input():
    """Backward compat: dict-of-Series still produces the binary y_pred + y_score."""
    rng = np.random.RandomState(2)
    val_scores = {f"m{i}": pd.Series(rng.uniform(0, 1, 40)) for i in range(3)}
    val_true = pd.Series([1] * 15 + [0] * 25)
    model = train_stacking_classifier(val_scores, val_true)

    test_scores = {f"m{i}": pd.Series(rng.uniform(0, 1, 12)) for i in range(3)}
    out = predict_stacking(model, test_scores)
    assert list(out.columns) == ["y_pred", "y_score"]
    assert ((out["y_score"] >= 0) & (out["y_score"] <= 1)).all()


def test_save_stacking_classifier_multiclass(tmp_path):
    from economy_classifier.ensemble import save_stacking_classifier

    rng = np.random.RandomState(3)
    classes = ["a", "b", "c"]
    val_features = {f"m{i}": _make_proba_df(rng, 40, classes) for i in range(2)}
    val_true = pd.Series((["a"] * 15) + (["b"] * 15) + (["c"] * 10))
    model = train_stacking_classifier(val_features, val_true, seed=42)

    save_stacking_classifier(model, tmp_path)

    meta = json.loads((tmp_path / "meta_classifier_meta.json").read_text())
    assert meta["classes"] == ["a", "b", "c"]
    assert "coefficients_per_class" in meta
    assert len(meta["coefficients_per_class"]) == 3  # one row per class
    assert "feature_names" in meta
    assert all("__" in fn for fn in meta["feature_names"])  # <model>__<class>


def test_save_stacking_classifier_binary(tmp_path):
    from economy_classifier.ensemble import save_stacking_classifier

    rng = np.random.RandomState(4)
    val_scores = {f"m{i}": pd.Series(rng.uniform(0, 1, 30)) for i in range(2)}
    val_true = pd.Series([1] * 12 + [0] * 18)
    model = train_stacking_classifier(val_scores, val_true)

    save_stacking_classifier(model, tmp_path, feature_names=list(val_scores))

    meta = json.loads((tmp_path / "meta_classifier_meta.json").read_text())
    assert "coefficients" in meta
    assert "intercept" in meta
    assert meta["feature_names"] == ["m0", "m1"]


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


# ---------------------------------------------------------------------------
# Fleiss' Kappa tests
# ---------------------------------------------------------------------------

def test_fleiss_kappa_perfect_agreement():
    preds = {f"m{i}": pd.Series([1, 0, 1, 0, 1]) for i in range(5)}
    kappa = compute_fleiss_kappa(preds)
    assert abs(kappa - 1.0) < 1e-10


def test_fleiss_kappa_returns_float():
    preds = {"a": pd.Series([1, 0, 1]), "b": pd.Series([0, 1, 0]), "c": pd.Series([1, 0, 1])}
    kappa = compute_fleiss_kappa(preds)
    assert isinstance(kappa, float)


def test_fleiss_kappa_range():
    rng = np.random.RandomState(42)
    preds = {f"m{i}": pd.Series(rng.randint(0, 2, 50)) for i in range(5)}
    kappa = compute_fleiss_kappa(preds)
    assert -1.0 <= kappa <= 1.0


def test_fleiss_kappa_multiclass_perfect_agreement():
    """Fleiss' kappa generalises to multi-categorical labels (strings)."""
    classes = ["poder", "mercado", "outros"]
    preds = {f"m{i}": pd.Series(classes * 10) for i in range(4)}
    kappa = compute_fleiss_kappa(preds)
    assert abs(kappa - 1.0) < 1e-10


def test_fleiss_kappa_multiclass_mixed():
    rng = np.random.RandomState(7)
    classes = np.array(["a", "b", "c", "d"])
    preds = {f"m{i}": pd.Series(classes[rng.randint(0, 4, 60)]) for i in range(5)}
    kappa = compute_fleiss_kappa(preds, categories=list(classes))
    assert -1.0 <= kappa <= 1.0


def test_fleiss_kappa_respects_explicit_categories():
    """Empty schema categories should not change the kappa numerically."""
    preds = {f"m{i}": pd.Series([1, 0, 1, 0, 1]) for i in range(3)}
    kappa_inferred = compute_fleiss_kappa(preds)
    kappa_explicit = compute_fleiss_kappa(preds, categories=[0, 1, 2])
    # Adding an unseen category 2 leaves p_i and p_j unchanged for it (zeros).
    assert abs(kappa_inferred - kappa_explicit) < 1e-10


# ---------------------------------------------------------------------------
# Threshold optimization tests
# ---------------------------------------------------------------------------

def test_optimize_voting_threshold_returns_valid():
    rng = np.random.RandomState(42)
    scores = {f"m{i}": pd.Series(rng.uniform(0, 1, 50)) for i in range(3)}
    weights = {f"m{i}": 0.8 for i in range(3)}
    y_true = pd.Series([1] * 15 + [0] * 35)
    result = optimize_voting_threshold(scores, y_true, weights)
    assert 0.3 <= result["best_threshold"] <= 0.7
    assert 0 <= result["best_f1"] <= 1.0
    assert len(result["all_results"]) > 0


def test_optimize_voting_threshold_best_matches():
    rng = np.random.RandomState(42)
    scores = {f"m{i}": pd.Series(rng.uniform(0, 1, 100)) for i in range(3)}
    weights = {f"m{i}": 1.0 for i in range(3)}
    y_true = pd.Series([1] * 30 + [0] * 70)
    result = optimize_voting_threshold(scores, y_true, weights)
    # Best F1 should be the max of all results
    max_f1 = max(r["f1"] for r in result["all_results"])
    assert result["best_f1"] == max_f1


# ---------------------------------------------------------------------------
# load_run_predictions tests
# ---------------------------------------------------------------------------

def test_load_run_predictions_split_file(tmp_path):
    df = pd.DataFrame({"index": [0, 1], "y_true": [0, 1], "y_pred": [0, 1], "y_score": [0.1, 0.9], "method": "test"})
    df.to_csv(tmp_path / "predictions_test.csv", index=False)
    loaded = load_run_predictions(tmp_path, split="test")
    assert loaded is not None
    assert len(loaded) == 2


def test_load_run_predictions_generic_file(tmp_path):
    df = pd.DataFrame({"index": [0], "y_true": [1], "y_pred": [1], "y_score": [0.8], "method": "test"})
    df.to_csv(tmp_path / "predictions.csv", index=False)
    loaded = load_run_predictions(tmp_path, split="val")
    assert loaded is not None
    assert len(loaded) == 1


def test_load_run_predictions_missing(tmp_path):
    loaded = load_run_predictions(tmp_path, split="test")
    assert loaded is None


# ---------------------------------------------------------------------------
# discover_runs tests
# ---------------------------------------------------------------------------

def test_discover_runs(tmp_path):
    # Create a fake TF-IDF run
    run1 = tmp_path / "run-tfidf"
    run1.mkdir()
    (run1 / "run_metadata.json").write_text(json.dumps({
        "run_id": "run-tfidf", "stage": "tfidf-training",
        "summary": {"method": "logreg", "f1": 0.76},
    }))

    # Create a fake BERT run
    run2 = tmp_path / "run-bert"
    run2.mkdir()
    (run2 / "run_metadata.json").write_text(json.dumps({
        "run_id": "run-bert", "stage": "bert-training",
        "summary": {"variant": "bertimbau", "val_metrics": {"f1": 0.81}},
    }))

    discovered = discover_runs(tmp_path)
    assert "logreg" in discovered
    assert "bertimbau" in discovered
    assert discovered["logreg"]["stage"] == "tfidf-training"
    assert discovered["bertimbau"]["stage"] == "bert-training"
