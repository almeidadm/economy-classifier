"""Tests for economy_classifier.project — run management and metadata."""

import json

import pandas as pd

from economy_classifier.project import (
    build_run_metadata,
    create_run_directory,
    get_git_commit_short,
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
