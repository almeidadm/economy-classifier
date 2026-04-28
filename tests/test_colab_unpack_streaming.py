"""Tests for the streaming Colab unpack script."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import colab_unpack_streaming as ucs  # noqa: E402


# ---------------------------------------------------------------------------
# is_safe_path — defesa contra zip-slip e caminhos fora de runs/
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "runs/foo/result_card.json",
    "runs/sub/dir/predictions.csv",
])
def test_is_safe_path_accepts_runs_entries(name):
    assert ucs.is_safe_path(name)


@pytest.mark.parametrize("name", [
    "/etc/passwd",
    "../escape/file",
    "runs/../etc/passwd",
    "outside/runs/file.json",
    "",
    "runs\\..\\nope",
])
def test_is_safe_path_rejects_unsafe(name):
    assert not ucs.is_safe_path(name)


# ---------------------------------------------------------------------------
# should_extract — whitelist e blacklist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "runs/bert_x/result_card.json",
    "runs/bert_x/predictions.csv",
    "runs/bert_x/confusion_matrix.csv",
    "runs/bert_x/run_metadata.json",
    "runs/tfidf_y/search_result.json",
    "runs/llm_z/examples_binary.json",
    "runs/llm_z/examples_multiclass.json",
    "runs/tfidf_y/model/tfidf_config.json",
    # Ensemble artifacts (notebook 43).
    "runs/ensemble_stacking_binary_test_set/meta_classifier.joblib",
    "runs/ensemble_stacking_binary_test_set/meta_classifier_meta.json",
    "runs/ensemble_stacking_multiclass_test_set/meta_classifier.joblib",
    "runs/ensemble_agreement_binary_fixed_split/agreement_matrix.csv",
    "runs/ensemble_agreement_multiclass_cv_5fold/fleiss_kappa.json",
    "runs/ensemble_agreement_binary_test_set/contingency_table.csv",
])
def test_should_extract_whitelist(name):
    assert ucs.should_extract(name, keep_tfidf_joblib=False)


@pytest.mark.parametrize("name", [
    "runs/bert_x/checkpoints/checkpoint-100/optimizer.pt",
    "runs/bert_x/checkpoints/checkpoint-100/model.safetensors",
    "runs/bert_x/checkpoints/checkpoint-100/trainer_state.json",
    "runs/bert_x/model/model.safetensors",
    "runs/bert_x/model/tokenizer.json",
    "runs/bert_x/model/tokenizer_config.json",
    "runs/bert_x/model/config.json",
    "runs/bert_x/model/training_args.bin",
    "runs/llm_z/predictions_checkpoint.csv",
    "runs/foo/random_other_file.txt",
    "runs/foo/",
])
def test_should_extract_blacklist(name):
    assert not ucs.should_extract(name, keep_tfidf_joblib=False)


def test_should_extract_joblib_flag_off():
    assert not ucs.should_extract(
        "runs/tfidf_y/model/tfidf_pipeline.joblib",
        keep_tfidf_joblib=False,
    )


def test_should_extract_joblib_flag_on():
    assert ucs.should_extract(
        "runs/tfidf_y/model/tfidf_pipeline.joblib",
        keep_tfidf_joblib=True,
    )


# ---------------------------------------------------------------------------
# extract_zip_streaming — integracao com zip in-memory
# ---------------------------------------------------------------------------


def _build_test_zip(zip_path: Path) -> dict[str, bytes]:
    """Build a zip with mixed whitelisted + blacklisted entries; return contents."""
    contents = {
        "runs/bert_a/result_card.json": b'{"model_id":"bert_a"}',
        "runs/bert_a/predictions.csv": b"index,y_true,y_pred\n0,1,1\n",
        "runs/bert_a/confusion_matrix.csv": b"col1,col2\n1,2\n",
        "runs/bert_a/model/model.safetensors": b"X" * 4096,
        "runs/bert_a/model/tokenizer.json": b'{"big":"tokenizer"}',
        "runs/bert_a/checkpoints/checkpoint-100/optimizer.pt": b"Y" * 8192,
        "runs/bert_a/checkpoints/checkpoint-100/trainer_state.json": b'{"epoch":1}',
        "runs/llm_b/predictions.csv": b"index,y_true\n0,0\n",
        "runs/llm_b/predictions_checkpoint.csv": b"index,y_true\n0,0\n",
        "runs/llm_b/result_card.json": b'{"model_id":"llm_b"}',
        "runs/tfidf_c/model/tfidf_pipeline.joblib": b"Z" * 2048,
        "runs/tfidf_c/model/tfidf_config.json": b'{"ngram":1}',
        "runs/tfidf_c/result_card.json": b'{"model_id":"tfidf_c"}',
    }
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, payload in contents.items():
            zf.writestr(name, payload)
    return contents


def test_extract_zip_streaming_filters_correctly(tmp_path):
    zip_path = tmp_path / "fake.zip"
    expected_contents = _build_test_zip(zip_path)
    dest = tmp_path / "out"

    stats = ucs.extract_zip_streaming(
        zip_path,
        dest,
        keep_tfidf_joblib=False,
        min_free_gb=0.0,
        dry_run=False,
    )

    assert ucs.validate_extraction(stats)
    assert stats.errors == []

    expected_extracted = {
        "runs/bert_a/result_card.json",
        "runs/bert_a/predictions.csv",
        "runs/bert_a/confusion_matrix.csv",
        "runs/llm_b/predictions.csv",
        "runs/llm_b/result_card.json",
        "runs/tfidf_c/model/tfidf_config.json",
        "runs/tfidf_c/result_card.json",
    }
    for rel in expected_extracted:
        full = dest / rel
        assert full.exists(), f"esperava {rel} extraido"
        assert full.read_bytes() == expected_contents[rel]

    blacklisted = [
        "runs/bert_a/model/model.safetensors",
        "runs/bert_a/model/tokenizer.json",
        "runs/bert_a/checkpoints/checkpoint-100/optimizer.pt",
        "runs/bert_a/checkpoints/checkpoint-100/trainer_state.json",
        "runs/llm_b/predictions_checkpoint.csv",
        "runs/tfidf_c/model/tfidf_pipeline.joblib",
    ]
    for rel in blacklisted:
        assert not (dest / rel).exists(), f"{rel} nao deveria estar extraido"

    assert stats.extracted_count == len(expected_extracted)
    assert stats.result_cards == 3
    assert stats.predictions == 2


def test_extract_zip_streaming_keep_joblib(tmp_path):
    zip_path = tmp_path / "fake.zip"
    _build_test_zip(zip_path)
    dest = tmp_path / "out"

    stats = ucs.extract_zip_streaming(
        zip_path,
        dest,
        keep_tfidf_joblib=True,
        min_free_gb=0.0,
        dry_run=False,
    )

    assert ucs.validate_extraction(stats)
    assert (dest / "runs/tfidf_c/model/tfidf_pipeline.joblib").exists()


def test_extract_zip_streaming_dry_run(tmp_path):
    zip_path = tmp_path / "fake.zip"
    _build_test_zip(zip_path)
    dest = tmp_path / "out"

    stats = ucs.extract_zip_streaming(
        zip_path,
        dest,
        keep_tfidf_joblib=False,
        min_free_gb=0.0,
        dry_run=True,
    )

    assert ucs.validate_extraction(stats)
    assert stats.extracted_count == 7
    assert stats.result_cards == 3
    assert stats.predictions == 2
    assert not dest.exists() or not any(dest.rglob("*")), "dry-run nao deve escrever"
