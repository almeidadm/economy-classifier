"""Tests for economy_classifier.tfidf — TF-IDF pipeline (M1, M2, M3)."""

import pandas as pd
import pytest

from economy_classifier.datasets import (
    build_balanced_training_frame,
    build_train_val_test_split,
)
from economy_classifier.tfidf import (
    TfidfTrainingConfig,
    load_tfidf_pipeline,
    predict_texts,
    train_tfidf_classifier,
)


def _small_config(classifier: str = "logreg") -> TfidfTrainingConfig:
    """Config suitable for synthetic data (small vocabulary)."""
    return TfidfTrainingConfig(
        classifier=classifier,
        max_features=100,
        min_df=1,
    )


@pytest.fixture
def trained_logreg(synthetic_corpus, tmp_path):
    train, val, _ = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)
    return train_tfidf_classifier(
        balanced, val, run_dir=tmp_path, config=_small_config("logreg"),
    )


def test_train_logreg_produces_metrics(synthetic_corpus, tmp_path):
    train, val, _ = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)
    result = train_tfidf_classifier(
        balanced, val, run_dir=tmp_path, config=_small_config("logreg"),
    )
    assert "metrics" in result
    assert "model_dir" in result
    assert "predictions" in result


def test_train_linearsvc_produces_metrics(synthetic_corpus, tmp_path):
    train, val, _ = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)
    result = train_tfidf_classifier(
        balanced, val, run_dir=tmp_path, config=_small_config("linearsvc"),
    )
    assert "metrics" in result
    assert "model_dir" in result


def test_train_multinomialnb_produces_metrics(synthetic_corpus, tmp_path):
    train, val, _ = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)
    result = train_tfidf_classifier(
        balanced, val, run_dir=tmp_path, config=_small_config("multinomialnb"),
    )
    assert "metrics" in result
    assert "model_dir" in result


def test_predictions_format_standard_csv(trained_logreg):
    preds = trained_logreg["predictions"]
    assert isinstance(preds, pd.DataFrame)
    assert list(preds.columns) == ["index", "y_true", "y_pred", "y_score", "method"]


def test_y_score_in_0_1_range(trained_logreg):
    preds = trained_logreg["predictions"]
    assert (preds["y_score"] >= 0).all()
    assert (preds["y_score"] <= 1).all()


def test_y_pred_is_binary(trained_logreg):
    preds = trained_logreg["predictions"]
    assert set(preds["y_pred"].unique()).issubset({0, 1})


def test_pipeline_roundtrip(synthetic_corpus, tmp_path):
    train, val, _ = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)
    result = train_tfidf_classifier(
        balanced, val, run_dir=tmp_path, config=_small_config("logreg"),
    )
    model_dir = result["model_dir"]
    load_tfidf_pipeline(model_dir)  # verify loads without error
    texts = val["text"].tolist()[:5]
    preds_a = predict_texts(texts, model_dir=model_dir, method="logreg")
    preds_b = predict_texts(texts, model_dir=model_dir, method="logreg")
    pd.testing.assert_frame_equal(preds_a, preds_b)


def test_logreg_has_predict_proba(synthetic_corpus, tmp_path):
    train, val, _ = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)
    result = train_tfidf_classifier(
        balanced, val, run_dir=tmp_path, config=_small_config("logreg"),
    )
    pipeline = load_tfidf_pipeline(result["model_dir"])
    assert hasattr(pipeline.named_steps["clf"], "predict_proba")


def test_linearsvc_score_calibrated(synthetic_corpus, tmp_path):
    train, val, _ = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)
    result = train_tfidf_classifier(
        balanced, val, run_dir=tmp_path, config=_small_config("linearsvc"),
    )
    preds = result["predictions"]
    assert (preds["y_score"] >= 0).all()
    assert (preds["y_score"] <= 1).all()


def test_multinomialnb_has_predict_proba(synthetic_corpus, tmp_path):
    train, val, _ = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)
    result = train_tfidf_classifier(
        balanced, val, run_dir=tmp_path, config=_small_config("multinomialnb"),
    )
    pipeline = load_tfidf_pipeline(result["model_dir"])
    assert hasattr(pipeline.named_steps["clf"], "predict_proba")


def test_config_defaults_match_requirements():
    cfg = TfidfTrainingConfig()
    assert cfg.max_features == 50_000
    assert cfg.ngram_range == (1, 2)
    assert cfg.sublinear_tf is True
    assert cfg.min_df == 2
    assert cfg.max_df == 0.95


def test_method_identifiers_correct():
    from economy_classifier.tfidf import METHOD_IDENTIFIERS

    assert METHOD_IDENTIFIERS["logreg"] == "logreg"
    assert METHOD_IDENTIFIERS["linearsvc"] == "linearsvc"
    assert METHOD_IDENTIFIERS["multinomialnb"] == "nb"


def test_train_with_synthetic_data(synthetic_corpus, tmp_path):
    train, val, _ = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)
    result = train_tfidf_classifier(
        balanced, val, run_dir=tmp_path, config=_small_config("logreg"),
    )
    assert result["metrics"]["f1"] > 0
    assert len(result["predictions"]) == len(val)
