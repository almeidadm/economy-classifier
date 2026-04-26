"""Integration tests — end-to-end pipeline with synthetic data."""

import json

import pandas as pd
import pytest

from economy_classifier.datasets import (
    build_balanced_training_frame,
    build_train_val_test_split,
)
from economy_classifier.ensemble import majority_vote, weighted_vote
from economy_classifier.evaluation import compute_binary_metrics
from economy_classifier.project import build_run_metadata, persist_run_artifacts
from economy_classifier.tfidf import TfidfTrainingConfig, train_tfidf_classifier


@pytest.mark.integration
def test_full_tfidf_pipeline_synthetic(synthetic_corpus, tmp_path):
    """corpus -> split -> balance -> train M1 -> predict -> metrics -> artifacts."""
    train, val, test = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)

    config = TfidfTrainingConfig(classifier="logreg", max_features=100, min_df=1)
    result = train_tfidf_classifier(balanced, val, run_dir=tmp_path, config=config)

    assert result["metrics"]["f1"] > 0
    preds = result["predictions"]
    assert len(preds) == len(val)

    metrics = compute_binary_metrics(preds["y_true"], preds["y_pred"])
    assert "precision" in metrics

    meta = build_run_metadata(
        run_dir=tmp_path, stage="tfidf-training",
        parameters=config.to_dict(), inputs={}, outputs={},
        summary=metrics, timing=result["timing"],
    )
    persist_run_artifacts(run_dir=tmp_path, metadata=meta, predictions=preds, metrics=metrics)

    assert (tmp_path / "run_metadata.json").exists()
    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "metrics.json").exists()


@pytest.mark.integration
def test_ensemble_on_synthetic_predictions(synthetic_corpus, tmp_path):
    """3 TF-IDF methods -> majority_vote + weighted_vote -> metrics."""
    train, val, _ = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)

    all_preds = {}
    all_scores = {}
    weights = {}

    for clf in ["logreg", "linearsvc", "multinomialnb"]:
        config = TfidfTrainingConfig(classifier=clf, max_features=100, min_df=1)
        run_dir = tmp_path / clf
        result = train_tfidf_classifier(balanced, val, run_dir=run_dir, config=config)
        preds = result["predictions"]
        method = preds["method"].iloc[0]
        all_preds[method] = preds["y_pred"].reset_index(drop=True)
        all_scores[method] = preds["y_score"].reset_index(drop=True)
        weights[method] = result["metrics"]["f1"]

    mv = majority_vote(all_preds, threshold=2)
    assert set(mv["y_pred"].unique()).issubset({0, 1})

    wv = weighted_vote(all_scores, weights)
    assert (wv["y_score"] >= 0).all()
    assert (wv["y_score"] <= 1).all()


@pytest.mark.integration
def test_artifacts_structure_complete(synthetic_corpus, tmp_path):
    """Verify run directory contains all expected files."""
    train, val, _ = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)
    config = TfidfTrainingConfig(classifier="logreg", max_features=100, min_df=1)
    result = train_tfidf_classifier(balanced, val, run_dir=tmp_path, config=config)

    meta = build_run_metadata(
        run_dir=tmp_path, stage="test",
        parameters={}, inputs={}, outputs={},
        summary=result["metrics"], timing=result["timing"],
    )
    persist_run_artifacts(
        run_dir=tmp_path, metadata=meta,
        predictions=result["predictions"], metrics=result["metrics"],
    )

    assert (tmp_path / "run_metadata.json").exists()
    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "metrics.json").exists()

    loaded_meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert "stage" in loaded_meta

    loaded_preds = pd.read_csv(tmp_path / "predictions.csv")
    assert list(loaded_preds.columns) == ["index", "y_true", "y_pred", "y_score", "method"]


@pytest.mark.integration
def test_end_to_end_no_data_leakage(synthetic_corpus):
    """Test predictions never include training indices."""
    train, val, test = build_train_val_test_split(synthetic_corpus, seed=42)
    train_idx = set(train.index)
    val_idx = set(val.index)
    test_idx = set(test.index)

    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
    assert train_idx.isdisjoint(val_idx)
