"""TF-IDF + linear classifier baseline for binary text classification."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

if TYPE_CHECKING:
    import pandas as pd

METHOD_IDENTIFIERS = {
    "logreg": "logreg",
    "linearsvc": "linearsvc",
    "multinomialnb": "nb",
}

MODEL_FILENAME = "tfidf_pipeline.joblib"
CONFIG_FILENAME = "tfidf_config.json"


@dataclass(slots=True)
class TfidfTrainingConfig:
    """Training configuration for the TF-IDF pipeline."""

    classifier: str = "logreg"
    max_features: int = 50_000
    ngram_range: tuple[int, int] = (1, 2)
    sublinear_tf: bool = True
    min_df: int = 2
    max_df: float = 0.95
    C: float = 1.0
    alpha: float = 1.0
    max_iter: int = 1000
    seed: int = 42

    def to_dict(self) -> dict[str, object]:
        return {
            "classifier": self.classifier,
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "sublinear_tf": self.sublinear_tf,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "C": self.C,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "seed": self.seed,
        }


def _build_pipeline(config: TfidfTrainingConfig) -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        sublinear_tf=config.sublinear_tf,
        min_df=config.min_df,
        max_df=config.max_df,
        strip_accents="unicode",
        lowercase=True,
    )

    if config.classifier == "linearsvc":
        base_clf = LinearSVC(
            C=config.C, max_iter=config.max_iter,
            random_state=config.seed, dual="auto",
        )
        clf = CalibratedClassifierCV(base_clf, cv=3)
    elif config.classifier == "multinomialnb":
        clf = MultinomialNB(alpha=config.alpha)
    else:
        clf = LogisticRegression(
            C=config.C, max_iter=config.max_iter,
            random_state=config.seed, solver="lbfgs",
        )

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


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


def train_tfidf_classifier(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    run_dir: str | Path,
    config: TfidfTrainingConfig | None = None,
) -> dict[str, object]:
    """Train a TF-IDF pipeline, evaluate on validation, persist model."""
    config = config or TfidfTrainingConfig()
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    pipeline = _build_pipeline(config)

    t0 = time.perf_counter()
    pipeline.fit(train_df["text"].fillna("").tolist(), train_df["label"].tolist())
    train_time = time.perf_counter() - t0

    val_texts = validation_df["text"].fillna("").tolist()
    val_labels = validation_df["label"].values

    t0 = time.perf_counter()
    y_pred = pipeline.predict(val_texts)
    probas = pipeline.predict_proba(val_texts)
    inference_time = time.perf_counter() - t0

    y_score = probas[:, 1]

    from sklearn.metrics import accuracy_score, f1_score
    metrics = {
        "accuracy": round(float(accuracy_score(val_labels, y_pred)), 4),
        "f1": round(float(f1_score(val_labels, y_pred, zero_division=0)), 4),
    }

    model_dir = run_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / MODEL_FILENAME)
    (model_dir / CONFIG_FILENAME).write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8",
    )

    method = METHOD_IDENTIFIERS.get(config.classifier, config.classifier)
    predictions = _build_predictions_df(validation_df, y_pred, y_score, method)

    return {
        "metrics": metrics,
        "model_dir": str(model_dir),
        "predictions": predictions,
        "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        "timing": {"train_seconds": round(train_time, 2), "inference_seconds": round(inference_time, 2)},
    }


def load_tfidf_pipeline(model_dir: str | Path) -> Pipeline:
    """Load a persisted TF-IDF pipeline from disk."""
    model_path = Path(model_dir) / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"TF-IDF model not found at {model_path}")
    return joblib.load(model_path)


def predict_texts(
    texts: list[str],
    *,
    model_dir: str | Path,
    method: str,
) -> pd.DataFrame:
    """Run batch inference and return standard prediction DataFrame."""
    import pandas as pd

    pipeline = load_tfidf_pipeline(model_dir)
    y_pred = pipeline.predict(texts)
    probas = pipeline.predict_proba(texts)
    y_score = probas[:, 1]

    return pd.DataFrame({
        "y_pred": y_pred.tolist(),
        "y_score": np.round(y_score, 4).tolist(),
    })
