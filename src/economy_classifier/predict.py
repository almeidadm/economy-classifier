"""Production inference API for the S2 stacking ensemble.

Usage::

    from economy_classifier.predict import load_ensemble, predict, predict_single

    ensemble = load_ensemble("artifacts/ensemble_s2")
    results = predict(ensemble, ["Bolsa sobe 2% com alta do dolar"])
    single  = predict_single(ensemble, "Debate eleitoral movimenta redes")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = {0: "outros", 1: "mercado"}


@dataclass
class EnsembleClassifier:
    """Holds all loaded sub-models and the meta-classifier."""

    bert_models: dict[str, tuple]  # method -> (tokenizer, model)
    tfidf_pipeline: Pipeline
    tfidf_method: str
    meta_classifier: LogisticRegression
    feature_order: list[str]
    device: torch.device


def load_ensemble(model_dir: str | Path) -> EnsembleClassifier:
    """Load the S2 stacking ensemble from an exported directory.

    Expects the directory to contain ``ensemble_config.json`` and
    ``meta_classifier.joblib``, as produced by ``scripts/export_ensemble.py``.
    """
    model_path = Path(model_dir)
    config_file = model_path / "ensemble_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Ensemble config not found at {config_file}")

    config = json.loads(config_file.read_text())
    methods = config["methods"]
    model_paths = config["model_paths"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load meta-classifier
    meta_clf = joblib.load(model_path / "meta_classifier.joblib")

    # Load sub-models
    bert_models: dict[str, tuple] = {}
    tfidf_pipeline = None
    tfidf_method = None

    for method in methods:
        mdir = Path(model_paths[method])

        # Detect model type by checking for tfidf_pipeline.joblib
        tfidf_file = mdir / "tfidf_pipeline.joblib"
        if tfidf_file.exists():
            tfidf_pipeline = joblib.load(tfidf_file)
            tfidf_method = method
        else:
            tokenizer = AutoTokenizer.from_pretrained(mdir)
            model = AutoModelForSequenceClassification.from_pretrained(mdir)
            model.to(device)
            model.eval()
            bert_models[method] = (tokenizer, model)

    if tfidf_pipeline is None:
        raise ValueError("No TF-IDF model found in ensemble config")

    return EnsembleClassifier(
        bert_models=bert_models,
        tfidf_pipeline=tfidf_pipeline,
        tfidf_method=tfidf_method,
        meta_classifier=meta_clf,
        feature_order=methods,
        device=device,
    )


def _bert_scores(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    *,
    batch_size: int = 16,
    max_length: int = 256,
) -> np.ndarray:
    """Run BERT inference and return positive-class probabilities."""
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
        all_scores.extend(probs[:, 1].tolist())
    return np.array(all_scores)


def _tfidf_scores(texts: list[str], pipeline: Pipeline) -> np.ndarray:
    """Run TF-IDF inference and return positive-class probabilities."""
    return pipeline.predict_proba(texts)[:, 1]


def predict(
    ensemble: EnsembleClassifier,
    texts: list[str],
    *,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Run the S2 stacking ensemble on a list of texts.

    Returns a DataFrame with columns: y_pred, y_score, label, and
    per-method scores (score_<method>).
    """
    scores = {}

    for method in ensemble.feature_order:
        if method == ensemble.tfidf_method:
            scores[method] = _tfidf_scores(texts, ensemble.tfidf_pipeline)
        else:
            tokenizer, model = ensemble.bert_models[method]
            scores[method] = _bert_scores(
                texts, tokenizer, model, ensemble.device,
                batch_size=batch_size,
            )

    # Build feature matrix in the correct order
    X = np.column_stack([scores[m] for m in ensemble.feature_order])

    y_pred = ensemble.meta_classifier.predict(X)
    y_score = ensemble.meta_classifier.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        "y_pred": y_pred.tolist(),
        "y_score": np.round(y_score, 4).tolist(),
        "label": [LABELS[p] for p in y_pred],
    })

    for method in ensemble.feature_order:
        result[f"score_{method}"] = np.round(scores[method], 4).tolist()

    return result


def load_texts_from_jsonl(
    path: str | Path,
    *,
    text_column: str = "analysis_text",
) -> tuple[pd.DataFrame, list[str]]:
    """Read a JSONL file and extract texts for prediction.

    Returns the full DataFrame (preserving all original fields) and the
    extracted texts as a list.
    """
    path = Path(path)
    df = pd.read_json(path, lines=True)
    if text_column not in df.columns:
        raise ValueError(
            f"Coluna '{text_column}' nao encontrada no JSONL. "
            f"Colunas disponiveis: {list(df.columns)}"
        )
    texts = df[text_column].fillna("").tolist()
    return df, texts


def predict_single(
    ensemble: EnsembleClassifier,
    text: str,
    *,
    batch_size: int = 16,
) -> dict:
    """Classify a single text. Returns a dict with prediction details."""
    df = predict(ensemble, [text], batch_size=batch_size)
    row = df.iloc[0]
    return {
        "y_pred": int(row["y_pred"]),
        "y_score": float(row["y_score"]),
        "label": row["label"],
        "sub_scores": {
            m: float(row[f"score_{m}"]) for m in ensemble.feature_order
        },
    }
