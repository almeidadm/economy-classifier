"""Tests for economy_classifier.bert — BERT pipeline (M4a, M4b, M4c), fully mocked."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import economy_classifier.bert as bert
from economy_classifier.bert import BertTrainingConfig, MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Fakes for inference tests (predict_texts)
# ---------------------------------------------------------------------------

class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self, texts, *, return_tensors="pt", truncation=True, padding=True, max_length=256,
    ):
        self.calls.append(list(texts))
        encoded = [[1] if "mercado" in t else [0] for t in texts]
        return {"input_ids": torch.tensor(encoded, dtype=torch.long)}


class FakeModel:
    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        logits = []
        for row in input_ids:
            if int(row[0].item()) == 1:
                logits.append([0.1, 0.9])
            else:
                logits.append([0.8, 0.2])
        return type("Out", (), {"logits": torch.tensor(logits, dtype=torch.float32)})()


def _patch_load_classifier(monkeypatch):
    tok = FakeTokenizer()
    model = FakeModel()
    monkeypatch.setattr(bert, "load_classifier", lambda _: (tok, model, torch.device("cpu")))
    return tok


# ---------------------------------------------------------------------------
# Fakes for training tests (train_bert_classifier)
# ---------------------------------------------------------------------------

class _FakeDataset:
    """Minimal stand-in for a HuggingFace Dataset in mocked training."""
    def __init__(self, n: int) -> None:
        self.n = n
    def __len__(self) -> int:
        return self.n


class _FakeTokenizerForTraining:
    """Tokenizer stub that only needs to support save_pretrained."""
    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)


def _patch_training(monkeypatch):
    """Mock all heavy components: tokenizer loading, model loading,
    dataset tokenization, and the Trainer itself.

    Returns a dict that captures kwargs passed to the Trainer constructor.
    """
    captured: dict = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def train(self):
            pass

        def evaluate(self):
            return {"eval_f1": 0.8, "eval_accuracy": 0.9}

        def save_model(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

        def remove_callback(self, callback_cls):
            pass

        def predict(self, dataset):
            n = len(dataset)
            preds = np.random.RandomState(42).randn(n, 2)
            return type("PredOut", (), {"predictions": preds})()

    monkeypatch.setattr(
        bert, "_tokenize_dataframe",
        lambda df, tok, ml, **kw: _FakeDataset(len(df)),
    )
    monkeypatch.setattr(
        bert, "AutoTokenizer",
        type("AT", (), {
            "from_pretrained": staticmethod(lambda *a, **k: _FakeTokenizerForTraining()),
        }),
    )
    monkeypatch.setattr(
        bert, "AutoModelForSequenceClassification",
        type("AM", (), {
            "from_pretrained": staticmethod(lambda *a, **k: None),
        }),
    )
    monkeypatch.setattr(bert, "_create_trainer", lambda **kw: FakeTrainer(**kw))
    return captured


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

def test_bert_training_config_defaults():
    cfg = BertTrainingConfig()
    assert cfg.max_length == 256
    assert cfg.learning_rate == 2e-5
    assert cfg.gradient_accumulation_steps == 8
    assert cfg.num_train_epochs == 3
    assert cfg.warmup_ratio == 0.1
    assert cfg.per_device_eval_batch_size == 8
    assert cfg.early_stopping_patience == 1
    assert cfg.save_total_limit == 2


def test_bert_config_for_each_variant():
    for key, model_name in MODEL_REGISTRY.items():
        cfg = BertTrainingConfig(model_name=model_name)
        assert cfg.model_name == model_name


def test_bert_config_to_dict():
    cfg = BertTrainingConfig()
    d = cfg.to_dict()
    assert "early_stopping_patience" in d
    assert "save_total_limit" in d
    assert d["per_device_eval_batch_size"] == 8


# ---------------------------------------------------------------------------
# Inference tests (predict_texts)
# ---------------------------------------------------------------------------

def test_predict_texts_batches_correctly(monkeypatch):
    tok = _patch_load_classifier(monkeypatch)
    bert.predict_texts(
        ["mercado sobe", "debate geral", "mercado fecha"],
        model_dir="unused", method="bertimbau", batch_size=2,
    )
    assert len(tok.calls) == 2
    assert tok.calls[0] == ["mercado sobe", "debate geral"]
    assert tok.calls[1] == ["mercado fecha"]


def test_predict_texts_returns_standard_format(monkeypatch):
    _patch_load_classifier(monkeypatch)
    preds = bert.predict_texts(
        ["mercado sobe", "debate geral"],
        model_dir="unused", method="bertimbau",
    )
    assert list(preds.columns) == ["y_pred", "y_score"]
    assert len(preds) == 2


def test_predict_texts_y_score_in_0_1(monkeypatch):
    _patch_load_classifier(monkeypatch)
    preds = bert.predict_texts(
        ["mercado sobe", "debate geral"],
        model_dir="unused", method="bertimbau",
    )
    assert (preds["y_score"] >= 0).all()
    assert (preds["y_score"] <= 1).all()


def test_predict_texts_rejects_invalid_batch_size():
    try:
        bert.predict_texts(["texto"], model_dir="unused", method="bertimbau", batch_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Should reject batch_size=0")


# ---------------------------------------------------------------------------
# Training tests (train_bert_classifier)
# ---------------------------------------------------------------------------

def test_train_returns_expected_keys(monkeypatch, synthetic_corpus, tmp_path):
    """Mock the Trainer to verify train_bert_classifier returns correct keys."""
    from economy_classifier.datasets import build_balanced_training_frame, build_train_val_test_split

    train, val, _ = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)

    _patch_training(monkeypatch)

    result = bert.train_bert_classifier(
        balanced, val, run_dir=tmp_path, config=BertTrainingConfig(),
    )
    assert "metrics" in result
    assert "model_dir" in result
    assert "timing" in result
    assert "predictions" in result


def test_train_predictions_format(monkeypatch, synthetic_corpus, tmp_path):
    """Verify predictions DataFrame has the standard columns."""
    from economy_classifier.datasets import build_balanced_training_frame, build_train_val_test_split

    train, val, _ = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)

    _patch_training(monkeypatch)

    result = bert.train_bert_classifier(
        balanced, val, run_dir=tmp_path, config=BertTrainingConfig(),
    )
    preds = result["predictions"]
    assert list(preds.columns) == ["index", "y_true", "y_pred", "y_score", "method"]
    assert len(preds) == len(val)
    assert preds["method"].iloc[0] == "bertimbau"


def test_training_args_early_stopping(monkeypatch, synthetic_corpus, tmp_path):
    """Verify TrainingArguments include early stopping config."""
    from economy_classifier.datasets import build_balanced_training_frame, build_train_val_test_split

    train, val, _ = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)

    captured = _patch_training(monkeypatch)

    bert.train_bert_classifier(
        balanced, val, run_dir=tmp_path, config=BertTrainingConfig(),
    )

    args = captured["args"]
    assert args.load_best_model_at_end is True
    assert args.metric_for_best_model == "f1"
    assert args.warmup_ratio == 0.1
    assert args.save_total_limit == 2
    assert captured["compute_metrics"] is bert._compute_metrics
    assert len(captured["callbacks"]) == 1


def test_training_saves_tokenizer(monkeypatch, synthetic_corpus, tmp_path):
    """Verify tokenizer.save_pretrained is called to model_dir."""
    from economy_classifier.datasets import build_balanced_training_frame, build_train_val_test_split

    train, val, _ = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)

    _patch_training(monkeypatch)

    result = bert.train_bert_classifier(
        balanced, val, run_dir=tmp_path, config=BertTrainingConfig(),
    )
    model_dir = Path(result["model_dir"])
    assert model_dir.exists()


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_method_identifiers():
    assert "bertimbau" in MODEL_REGISTRY
    assert "finbert" in MODEL_REGISTRY
    assert "finbert_ptbr" in MODEL_REGISTRY
