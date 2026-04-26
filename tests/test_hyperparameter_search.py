"""Tests for economy_classifier.hyperparameter_search."""

from __future__ import annotations

import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from scipy.stats import loguniform

from economy_classifier.hyperparameter_search import (
    SearchResult,
    _build_search_pipeline,
    _describe_bert_space,
    _describe_tfidf_space,
    _sample_bert_params,
    _summarize_sklearn_cv_results,
    _to_jsonable,
    build_bert_search_space,
    build_tfidf_search_space,
    random_search_bert,
    random_search_tfidf,
    tfidf_best_params_to_kwargs,
)


# ---------------------------------------------------------------------------
# Search-space builders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("classifier", ["logreg", "linearsvc", "multinomialnb"])
def test_build_tfidf_search_space_binary_has_expected_prefix(classifier):
    space = build_tfidf_search_space(classifier, multiclass=False)
    # All vectorizer keys present
    for key in (
        "tfidf__ngram_range", "tfidf__min_df", "tfidf__max_df",
        "tfidf__max_features", "tfidf__sublinear_tf",
    ):
        assert key in space
    # Classifier params use clf__ prefix (no OvR wrapping in binary)
    clf_keys = [k for k in space if k.startswith("clf__")]
    assert clf_keys
    assert all(not k.startswith("clf__estimator__") for k in clf_keys)


@pytest.mark.parametrize("classifier", ["logreg", "linearsvc", "multinomialnb"])
def test_build_tfidf_search_space_multiclass_native_uses_clf_prefix(classifier):
    space = build_tfidf_search_space(classifier, multiclass=True, strategy="native")
    clf_keys = [k for k in space if k.startswith("clf__")]
    assert clf_keys
    assert all(not k.startswith("clf__estimator__") for k in clf_keys)


@pytest.mark.parametrize("classifier", ["logreg", "linearsvc", "multinomialnb"])
def test_build_tfidf_search_space_multiclass_ovr_uses_estimator_prefix(classifier):
    space = build_tfidf_search_space(classifier, multiclass=True, strategy="ovr")
    clf_keys = [k for k in space if k.startswith("clf__")]
    assert clf_keys
    assert all(k.startswith("clf__estimator__") for k in clf_keys)


def test_build_tfidf_search_space_logreg_has_C_and_class_weight():
    space = build_tfidf_search_space("logreg", multiclass=False)
    assert "clf__C" in space
    assert "clf__class_weight" in space
    assert space["clf__class_weight"] == [None, "balanced"]


def test_build_tfidf_search_space_nb_has_alpha_and_fit_prior():
    space = build_tfidf_search_space("multinomialnb", multiclass=False)
    assert "clf__alpha" in space
    assert "clf__fit_prior" in space
    assert space["clf__fit_prior"] == [True, False]


def test_build_tfidf_search_space_invalid_classifier():
    with pytest.raises(ValueError, match="classifier"):
        build_tfidf_search_space("randomforest")


def test_build_tfidf_search_space_invalid_strategy():
    with pytest.raises(ValueError, match="strategy"):
        build_tfidf_search_space("logreg", strategy="bogus")


def test_build_bert_search_space_has_expected_keys():
    space = build_bert_search_space()
    for key in (
        "learning_rate", "per_device_train_batch_size", "num_train_epochs",
        "weight_decay", "warmup_ratio", "gradient_accumulation_steps",
    ):
        assert key in space
        assert space[key][0] in {"loguniform", "uniform", "int", "choice"}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sample_bert_params_is_deterministic_with_seed():
    space = build_bert_search_space()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    p1 = _sample_bert_params(space, rng1)
    p2 = _sample_bert_params(space, rng2)
    assert p1 == p2


def test_sample_bert_params_respects_distribution_kinds():
    space = build_bert_search_space()
    rng = np.random.default_rng(0)
    sampled = _sample_bert_params(space, rng)
    assert isinstance(sampled["learning_rate"], float)
    assert 1e-5 <= sampled["learning_rate"] <= 5e-5
    assert sampled["per_device_train_batch_size"] in (8, 16, 32)
    assert isinstance(sampled["num_train_epochs"], int)
    assert 2 <= sampled["num_train_epochs"] <= 5
    assert isinstance(sampled["weight_decay"], float)
    assert 0.0 <= sampled["warmup_ratio"] <= 0.2
    assert sampled["gradient_accumulation_steps"] in (1, 2, 4)


def test_sample_bert_params_unknown_kind_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Unknown sampling kind"):
        _sample_bert_params({"x": ("nonsense", 1, 2)}, rng)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("classifier", ["logreg", "linearsvc", "multinomialnb"])
def test_build_search_pipeline_binary(classifier):
    pipeline = _build_search_pipeline(classifier, multiclass=False)
    assert "tfidf" in pipeline.named_steps
    assert "clf" in pipeline.named_steps
    # Binary search pipeline never wraps LinearSVC in CalibratedClassifierCV
    if classifier == "linearsvc":
        from sklearn.calibration import CalibratedClassifierCV
        assert not isinstance(pipeline.named_steps["clf"], CalibratedClassifierCV)


def test_build_search_pipeline_multiclass_ovr_wraps_in_onevsrest():
    from sklearn.multiclass import OneVsRestClassifier

    pipeline = _build_search_pipeline("logreg", multiclass=True, strategy="ovr")
    assert isinstance(pipeline.named_steps["clf"], OneVsRestClassifier)


def test_build_search_pipeline_invalid_classifier():
    with pytest.raises(ValueError):
        _build_search_pipeline("xgboost")


# ---------------------------------------------------------------------------
# Best-params translation
# ---------------------------------------------------------------------------


def test_tfidf_best_params_to_kwargs_strips_pipeline_prefixes():
    best = {
        "tfidf__ngram_range": (1, 2),
        "tfidf__min_df": 5,
        "clf__C": 0.5,
        "clf__class_weight": "balanced",
    }
    out = tfidf_best_params_to_kwargs(best)
    assert out == {
        "ngram_range": (1, 2),
        "min_df": 5,
        "C": 0.5,
        "class_weight": "balanced",
    }


def test_tfidf_best_params_to_kwargs_strips_estimator_prefix():
    best = {"clf__estimator__C": 1.5, "tfidf__sublinear_tf": True}
    out = tfidf_best_params_to_kwargs(best)
    assert out == {"C": 1.5, "sublinear_tf": True}


def test_tfidf_best_params_to_kwargs_coerces_ngram_list_to_tuple():
    out = tfidf_best_params_to_kwargs({"tfidf__ngram_range": [1, 3]})
    assert out["ngram_range"] == (1, 3)
    assert isinstance(out["ngram_range"], tuple)


def test_tfidf_best_params_to_kwargs_keeps_unknown_keys():
    out = tfidf_best_params_to_kwargs({"completely_other": "x"})
    assert out == {"completely_other": "x"}


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def test_describe_tfidf_space_is_json_serializable():
    space = build_tfidf_search_space("logreg")
    described = _describe_tfidf_space(space)
    json.dumps(described)  # must not raise
    assert described["clf__C"]["type"] == "loguniform"
    assert "args" in described["clf__C"]
    assert described["tfidf__ngram_range"]["type"] == "choice"


def test_describe_bert_space_is_json_serializable():
    space = build_bert_search_space()
    described = _describe_bert_space(space)
    json.dumps(described)
    assert described["learning_rate"]["type"] == "loguniform"
    assert described["per_device_train_batch_size"]["type"] == "choice"
    assert described["num_train_epochs"]["type"] == "int"


def test_to_jsonable_handles_numpy_scalars():
    out = _to_jsonable({"a": np.int64(3), "b": np.float64(1.5), "c": [np.int32(1)]})
    assert out == {"a": 3, "b": 1.5, "c": [1]}
    json.dumps(out)


def test_to_jsonable_falls_back_to_repr_for_non_serializable():
    class Custom:
        def __repr__(self):
            return "Custom()"

    out = _to_jsonable({"x": Custom()})
    assert out == {"x": "Custom()"}


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------


def test_search_result_to_dict_and_card_payload():
    result = SearchResult(
        best_params={"C": 0.5},
        best_score=0.81234,
        n_trials=10,
        search_space={"C": {"type": "loguniform", "args": [0.001, 100]}},
        search_seconds=12.345,
        scoring="f1",
        trials=[{"trial": 0, "params": {"C": 0.5}, "score": 0.81}],
    )
    full = result.to_dict()
    card = result.card_payload()

    assert full["best_score"] == 0.8123
    assert full["search_seconds"] == 12.35
    assert full["trials"]
    assert "trials" not in card  # card payload omits per-trial detail
    assert card["best_params"] == {"C": 0.5}
    json.dumps(full)
    json.dumps(card)


# ---------------------------------------------------------------------------
# Integration: random_search_tfidf on a tiny dataset
# ---------------------------------------------------------------------------


def _tiny_corpus(n_per_class: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pos = ["mercado acao bolsa juros taxa cambio dividas economia ipca selic"] * n_per_class
    neg = ["futebol campeonato gol jogador estadio torcida tecnico craque time"] * n_per_class
    # Sprinkle slight variation so vectorizer doesn't degenerate
    pos = [f"{t} {rng.integers(0, 1000)}" for t in pos]
    neg = [f"{t} {rng.integers(0, 1000)}" for t in neg]
    texts = pos + neg
    labels = [1] * n_per_class + [0] * n_per_class
    return pd.DataFrame({"text": texts, "label": labels})


def test_random_search_tfidf_runs_on_tiny_dataset():
    df = _tiny_corpus()
    result = random_search_tfidf(
        df,
        classifier="logreg",
        n_iter=3,
        cv_n_splits=2,
        n_jobs=1,
        seed=42,
        verbose=0,
    )
    assert isinstance(result, SearchResult)
    assert result.n_trials == 3
    assert len(result.trials) == 3
    assert 0.0 <= result.best_score <= 1.0
    assert result.scoring == "f1"
    assert "tfidf__ngram_range" in result.search_space


def test_random_search_tfidf_multiclass_uses_macro_f1():
    rng = np.random.default_rng(0)
    texts = (
        ["mercado bolsa acao"] * 10
        + ["futebol gol time"] * 10
        + ["filme cinema arte"] * 10
    )
    labels = ["mercado"] * 10 + ["esporte"] * 10 + ["ilustrada"] * 10
    df = pd.DataFrame({
        "text": [f"{t} {rng.integers(0, 1000)}" for t in texts],
        "label_multi": labels,
    })
    result = random_search_tfidf(
        df,
        classifier="logreg",
        label_column="label_multi",
        multiclass=True,
        strategy="native",
        n_iter=2,
        cv_n_splits=2,
        n_jobs=1,
        seed=42,
        verbose=0,
    )
    assert result.scoring == "f1_macro"


def test_summarize_sklearn_cv_results_shape():
    fake = {
        "params": [{"clf__C": 0.1}, {"clf__C": 1.0}],
        "mean_test_score": np.array([0.7, 0.8]),
        "std_test_score": np.array([0.01, 0.02]),
        "mean_fit_time": np.array([0.5, 0.6]),
        "rank_test_score": np.array([2, 1]),
    }
    rows = _summarize_sklearn_cv_results(fake)
    assert len(rows) == 2
    assert rows[0]["mean_test_score"] == 0.7
    assert rows[1]["rank_test_score"] == 1


# ---------------------------------------------------------------------------
# random_search_bert: validate the orchestration without launching real BERT
# ---------------------------------------------------------------------------


def test_random_search_bert_validates_label_set_for_multiclass(tmp_path):
    df = pd.DataFrame({"text": ["a"], "label_multi": ["x"]})
    with pytest.raises(ValueError, match="label_set"):
        random_search_bert(
            df, df,
            model_name="dummy",
            work_dir=tmp_path,
            multiclass=True,
            label_set=(),
            n_iter=1,
        )


def test_random_search_bert_orchestrates_trials_and_picks_best(tmp_path):
    """Mock train_bert_classifier to verify scoring/selection logic without GPU."""
    train = pd.DataFrame({"text": ["a", "b", "c", "d"], "label": [0, 1, 0, 1]})
    val = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})

    scores_iter = iter([0.6, 0.9, 0.7])

    def fake_train_bert(train_df, validation_df, *, run_dir, config):
        score = next(scores_iter)
        return {
            "metrics": {"eval_f1": score, "eval_accuracy": score},
            "model_dir": str(run_dir / "model"),
            "predictions": pd.DataFrame(),
            "timing": {"train_seconds": 1.0},
        }

    with mock.patch(
        "economy_classifier.bert.train_bert_classifier",
        side_effect=fake_train_bert,
    ):
        result = random_search_bert(
            train, val,
            model_name="neuralmind/bert-base-portuguese-cased",
            work_dir=tmp_path,
            n_iter=3,
            seed=0,
        )

    assert result.n_trials == 3
    assert result.best_score == 0.9
    assert len(result.trials) == 3
    json.dumps(result.to_dict())


def test_random_search_bert_records_trial_failures(tmp_path):
    train = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    val = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})

    call_count = {"n": 0}

    def fake_train_bert(train_df, validation_df, *, run_dir, config):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated CUDA OOM")
        return {
            "metrics": {"eval_f1": 0.75},
            "model_dir": str(run_dir / "model"),
            "predictions": pd.DataFrame(),
            "timing": {"train_seconds": 1.0},
        }

    with mock.patch(
        "economy_classifier.bert.train_bert_classifier",
        side_effect=fake_train_bert,
    ):
        result = random_search_bert(
            train, val,
            model_name="dummy",
            work_dir=tmp_path,
            n_iter=2,
            seed=0,
        )

    failed = [t for t in result.trials if "error" in t]
    succeeded = [t for t in result.trials if "error" not in t]
    assert len(failed) == 1
    assert len(succeeded) == 1
    assert result.best_score == 0.75


def test_random_search_bert_all_trials_failing_raises(tmp_path):
    train = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    val = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})

    def always_fail(train_df, validation_df, *, run_dir, config):
        raise RuntimeError("everything is on fire")

    with mock.patch(
        "economy_classifier.bert.train_bert_classifier",
        side_effect=always_fail,
    ):
        with pytest.raises(RuntimeError, match="trials failed"):
            random_search_bert(
                train, val,
                model_name="dummy",
                work_dir=tmp_path,
                n_iter=2,
                seed=0,
            )
