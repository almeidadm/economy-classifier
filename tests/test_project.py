"""Tests for economy_classifier.project — run management and metadata."""

import json

import pandas as pd
import pytest

from economy_classifier.project import (
    build_result_card,
    build_run_metadata,
    compute_artifact_size_mb,
    create_run_directory,
    get_git_commit_short,
    persist_result_card,
    persist_run_artifacts,
    slugify,
)


def test_create_run_directory_creates_timestamped_dir(tmp_path, monkeypatch):
    import economy_classifier.project as proj

    monkeypatch.setattr(proj, "RUNS_DIR", tmp_path / "runs")
    run_dir = create_run_directory("tfidf-training", run_name="logreg")
    assert run_dir.exists()
    assert run_dir.is_dir()
    assert "tfidf-training" in run_dir.name
    assert "logreg" in run_dir.name


def test_create_run_directory_distinct_calls(tmp_path, monkeypatch):
    import economy_classifier.project as proj

    monkeypatch.setattr(proj, "RUNS_DIR", tmp_path / "runs")
    dir_a = create_run_directory("training", run_name="a")
    dir_b = create_run_directory("training", run_name="b")
    assert dir_a != dir_b


def test_build_run_metadata_contains_required_fields(tmp_path):
    meta = build_run_metadata(
        run_dir=tmp_path,
        stage="tfidf-training",
        parameters={"C": 1.0},
        inputs={"train": "train.csv"},
        outputs={"model": "model/"},
        summary={"f1": 0.9},
        timing={"train_seconds": 12.0},
    )
    required = {"run_id", "stage", "git_commit", "generated_at", "parameters", "inputs", "outputs", "summary", "timing"}
    assert required.issubset(meta.keys())
    assert meta["stage"] == "tfidf-training"
    assert meta["parameters"]["C"] == 1.0


def test_persist_run_artifacts_saves_metadata_json(tmp_path):
    meta = build_run_metadata(
        run_dir=tmp_path,
        stage="test",
        parameters={},
        inputs={},
        outputs={},
        summary={},
        timing={},
    )
    persist_run_artifacts(run_dir=tmp_path, metadata=meta)
    path = tmp_path / "run_metadata.json"
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["stage"] == "test"


def test_persist_run_artifacts_saves_predictions_csv(tmp_path):
    meta = build_run_metadata(
        run_dir=tmp_path, stage="test",
        parameters={}, inputs={}, outputs={}, summary={}, timing={},
    )
    predictions = pd.DataFrame({
        "index": [0, 1, 2],
        "y_true": [1, 0, 1],
        "y_pred": [1, 0, 0],
        "y_score": [0.9, 0.1, 0.4],
        "method": ["logreg", "logreg", "logreg"],
    })
    persist_run_artifacts(run_dir=tmp_path, metadata=meta, predictions=predictions)
    path = tmp_path / "predictions.csv"
    assert path.exists()
    loaded = pd.read_csv(path)
    assert list(loaded.columns) == ["index", "y_true", "y_pred", "y_score", "method"]
    assert len(loaded) == 3


def test_persist_run_artifacts_saves_metrics_json(tmp_path):
    meta = build_run_metadata(
        run_dir=tmp_path, stage="test",
        parameters={}, inputs={}, outputs={}, summary={}, timing={},
    )
    metrics = {"precision": 0.9, "recall": 0.8, "f1": 0.85, "accuracy": 0.95}
    persist_run_artifacts(run_dir=tmp_path, metadata=meta, metrics=metrics)
    path = tmp_path / "metrics.json"
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["f1"] == 0.85


def test_slugify_handles_special_characters():
    assert slugify("Olá Mundo / teste") == "ola-mundo-teste"
    assert slugify("tfidf--training") == "tfidf-training"
    assert slugify("  ") == "run"
    assert slugify("BERT_fine_tuned") == "bert-fine-tuned"


def test_get_git_commit_short_returns_string():
    result = get_git_commit_short()
    assert isinstance(result, str)
    assert len(result) > 0


def test_build_result_card_minimal():
    card = build_result_card(
        model_id="tfidf_logreg", task="binary", regime="fixed_split",
        metrics={"f1": 0.81}, cost={"train_seconds_mean": 2.0}, config={"C": 1.0},
    )
    assert card["model_id"] == "tfidf_logreg"
    assert card["task"] == "binary"
    assert card["regime"] == "fixed_split"
    assert card["metrics"]["f1"] == 0.81
    assert "git_commit" in card and "generated_at" in card


def test_build_result_card_rejects_invalid_task():
    with pytest.raises(ValueError):
        build_result_card(
            model_id="x", task="ternary", regime="fixed_split",
            metrics={}, cost={}, config={},
        )


def test_build_result_card_rejects_invalid_regime():
    with pytest.raises(ValueError):
        build_result_card(
            model_id="x", task="binary", regime="kfold_99",
            metrics={}, cost={}, config={},
        )


def test_build_result_card_with_hyperparameter_search():
    card = build_result_card(
        model_id="tfidf_logreg", task="binary", regime="test_set",
        metrics={"f1": 0.83}, cost={}, config={"C": 0.5},
        hyperparameter_search={
            "best_params": {"C": 0.5, "ngram_range": [1, 2]},
            "best_score": 0.812,
            "n_trials": 60,
            "search_seconds": 320.5,
            "scoring": "f1",
            "search_space": {"clf__C": {"type": "loguniform", "args": [0.001, 100]}},
        },
    )
    assert card["hyperparameter_search"]["n_trials"] == 60
    assert card["hyperparameter_search"]["best_params"]["C"] == 0.5


def test_build_result_card_default_hyperparameter_search_is_none():
    card = build_result_card(
        model_id="x", task="binary", regime="fixed_split",
        metrics={}, cost={}, config={},
    )
    assert card["hyperparameter_search"] is None


def test_build_result_card_rejects_binary_with_macro_scoring():
    with pytest.raises(ValueError, match="incompatible"):
        build_result_card(
            model_id="tfidf_logreg", task="binary", regime="test_set",
            metrics={}, cost={}, config={},
            hyperparameter_search={
                "best_params": {}, "best_score": 0.0, "n_trials": 60,
                "search_seconds": 1.0, "scoring": "f1_macro",
                "search_space": {},
            },
        )


def test_build_result_card_rejects_multiclass_with_binary_scoring():
    with pytest.raises(ValueError, match="incompatible"):
        build_result_card(
            model_id="tfidf_logreg", task="multiclass", regime="test_set",
            metrics={}, cost={}, config={},
            hyperparameter_search={
                "best_params": {}, "best_score": 0.0, "n_trials": 60,
                "search_seconds": 1.0, "scoring": "f1",
                "search_space": {},
            },
        )


def test_build_result_card_accepts_macro_f1_alias_for_multiclass():
    """BERT uses 'macro_f1' (its internal name) instead of sklearn's 'f1_macro'."""
    card = build_result_card(
        model_id="bert", task="multiclass", regime="test_set",
        metrics={}, cost={}, config={},
        hyperparameter_search={
            "best_params": {}, "best_score": 0.0, "n_trials": 25,
            "search_seconds": 1.0, "scoring": "macro_f1",
            "search_space": {},
        },
    )
    assert card["hyperparameter_search"]["scoring"] == "macro_f1"


def test_build_result_card_allows_null_scoring_in_search_payload():
    """Some search payloads may carry scoring=None; treat as unspecified, not invalid."""
    card = build_result_card(
        model_id="x", task="binary", regime="test_set",
        metrics={}, cost={}, config={},
        hyperparameter_search={
            "best_params": {}, "best_score": 0.0, "n_trials": 0,
            "search_seconds": 0.0, "scoring": None, "search_space": {},
        },
    )
    assert card["hyperparameter_search"]["scoring"] is None


def test_persist_result_card_writes_json(tmp_path):
    card = build_result_card(
        model_id="x", task="binary", regime="test_set",
        metrics={"f1": 0.5}, cost={}, config={},
    )
    out = persist_result_card(card, tmp_path)
    assert out == tmp_path / "result_card.json"
    loaded = json.loads(out.read_text())
    assert loaded["model_id"] == "x"


def test_compute_artifact_size_mb_file(tmp_path):
    f = tmp_path / "model.bin"
    f.write_bytes(b"x" * 2_000_000)
    size = compute_artifact_size_mb(f)
    assert 1.9 < size < 2.1


def test_compute_artifact_size_mb_directory(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 1_000_000)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"x" * 500_000)
    size = compute_artifact_size_mb(tmp_path)
    assert 1.4 < size < 1.6
