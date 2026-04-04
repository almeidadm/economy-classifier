"""BERT fine-tuning and inference for binary text classification."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

if TYPE_CHECKING:
    import pandas as pd

MODEL_REGISTRY = {
    "bertimbau": "neuralmind/bert-base-portuguese-cased",
    "finbert": "ProsusAI/finbert",
    "finbert_ptbr": "lucas-leme/FinBERT-PT-BR",
}

LABELS = {0: "outros", 1: "mercado"}


@dataclass(slots=True)
class BertTrainingConfig:
    """Training configuration for BERT fine-tuning."""

    model_name: str = "neuralmind/bert-base-portuguese-cased"
    max_length: int = 256
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    early_stopping_patience: int = 1
    save_total_limit: int = 2

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_epochs": self.num_train_epochs,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "seed": self.seed,
            "early_stopping_patience": self.early_stopping_patience,
            "save_total_limit": self.save_total_limit,
        }


def _tokenize_dataframe(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    *,
    text_column: str = "text",
    label_column: str = "label",
):
    """Tokenize a DataFrame and return a HuggingFace Dataset."""
    from datasets import Dataset

    texts = df[text_column].fillna("").tolist()
    labels = df[label_column].tolist()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    dataset = Dataset.from_dict({
        **encodings,
        "labels": labels,
    })
    dataset.set_format("torch")
    return dataset


def _compute_metrics(eval_pred):
    """Compute F1 and accuracy for the HuggingFace Trainer."""
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def _build_predictions_df(
    val_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    method: str,
) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame({
        "index": val_df.index.tolist(),
        "y_true": val_df["label"].tolist(),
        "y_pred": y_pred.tolist(),
        "y_score": np.round(y_score, 4).tolist(),
        "method": method,
    })


def _create_trainer(**kwargs):
    """Create a HuggingFace Trainer (separated for testability)."""
    from transformers import Trainer
    return Trainer(**kwargs)


def load_classifier(model_dir: str | Path) -> tuple:
    """Load tokenizer, model and device from a saved checkpoint."""
    model_path = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def train_bert_classifier(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    run_dir: str | Path,
    config: BertTrainingConfig | None = None,
) -> dict[str, object]:
    """Fine-tune a BERT model for binary classification.

    Loads the pre-trained tokenizer and model, tokenizes the DataFrames,
    trains with early stopping on F1, generates validation predictions,
    and persists the model and tokenizer to ``run_dir/model/``.
    """
    config = config or BertTrainingConfig()
    run_path = Path(run_dir)
    model_dir = run_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=2, ignore_mismatched_sizes=True,
    )

    train_dataset = _tokenize_dataframe(train_df, tokenizer, config.max_length)
    val_dataset = _tokenize_dataframe(validation_df, tokenizer, config.max_length)

    training_args = TrainingArguments(
        output_dir=str(run_path / "checkpoints"),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=config.seed,
        fp16=torch.cuda.is_available(),
        disable_tqdm=False,
        logging_steps=100,
        save_total_limit=config.save_total_limit,
    )

    from transformers import EarlyStoppingCallback

    t0 = time.perf_counter()
    trainer = _create_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
        )],
    )
    trainer.train()
    train_time = time.perf_counter() - t0

    # Remove NotebookProgressCallback to avoid RuntimeError when calling
    # evaluate() or predict() outside the training loop — after train() ends
    # the callback's internal tracker is None, causing it to raise.
    try:
        from transformers.utils.notebook import NotebookProgressCallback
        trainer.remove_callback(NotebookProgressCallback)
    except Exception:
        pass

    metrics = trainer.evaluate()

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    val_output = trainer.predict(val_dataset)
    val_probs = torch.softmax(
        torch.tensor(val_output.predictions, dtype=torch.float32), dim=-1,
    ).numpy()
    y_pred = np.argmax(val_output.predictions, axis=-1)
    y_score = val_probs[:, 1]

    method_key = next(
        (k for k, v in MODEL_REGISTRY.items() if v == config.model_name),
        config.model_name,
    )

    predictions = _build_predictions_df(validation_df, y_pred, y_score, method_key)

    return {
        "metrics": metrics,
        "model_dir": str(model_dir),
        "predictions": predictions,
        "timing": {"train_seconds": round(train_time, 2)},
    }


def predict_texts(
    texts: list[str],
    *,
    model_dir: str | Path,
    method: str,
    max_length: int = 256,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Run batch inference and return standard prediction DataFrame."""
    import pandas as pd

    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero.")

    tokenizer, model, device = load_classifier(model_dir)

    all_preds = []
    all_scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        scores = probs[:, 1]

        all_preds.extend(preds.tolist())
        all_scores.extend(np.round(scores, 4).tolist())

    return pd.DataFrame({
        "y_pred": all_preds,
        "y_score": all_scores,
    })
