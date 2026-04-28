"""Random hyperparameter search for TF-IDF and BERT pipelines.

Two protocols:

- **TF-IDF**: ``RandomizedSearchCV`` over a Pipeline (tfidf + classifier),
  scored with inner ``StratifiedKFold``. Fast and parallel via sklearn's
  built-in machinery.
- **BERT**: custom random-search loop, one trial = one fine-tune. Each trial
  trains on ``train_df`` and scores on ``val_df`` (single inner split) — declared
  asymmetry vs TF-IDF inner-CV in result cards.

Both produce a :class:`SearchResult` whose ``card_payload`` is meant to live in
the ``hyperparameter_search`` field of a ``result_card.json``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import loguniform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

if TYPE_CHECKING:
    import pandas as pd

VALID_TFIDF_CLASSIFIERS = {"logreg", "linearsvc", "multinomialnb"}
VALID_STRATEGIES = {"native", "ovr"}


@dataclass(slots=True)
class SearchResult:
    """Outcome of a hyperparameter search."""

    best_params: dict
    best_score: float
    n_trials: int
    search_space: dict
    search_seconds: float
    scoring: str
    trials: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "best_params": _to_jsonable(self.best_params),
            "best_score": round(float(self.best_score), 4),
            "n_trials": int(self.n_trials),
            "search_space": self.search_space,
            "search_seconds": round(float(self.search_seconds), 2),
            "scoring": self.scoring,
            "trials": self.trials,
        }

    def card_payload(self) -> dict:
        """Compact form for inclusion in a result_card (no per-trial detail)."""
        return {
            "best_params": _to_jsonable(self.best_params),
            "best_score": round(float(self.best_score), 4),
            "n_trials": int(self.n_trials),
            "search_space": self.search_space,
            "search_seconds": round(float(self.search_seconds), 2),
            "scoring": self.scoring,
        }


# ---------------------------------------------------------------------------
# Search-space builders
# ---------------------------------------------------------------------------


def build_tfidf_search_space(
    classifier: str,
    *,
    multiclass: bool = False,
    strategy: str = "native",
) -> dict:
    """Standard distribution dict for ``RandomizedSearchCV`` on a TF-IDF pipeline.

    Parameter prefixing handles ``OneVsRestClassifier`` (multiclass+ovr).
    The search pipeline never wraps LinearSVC in CalibratedClassifierCV — calibration
    only affects ``y_score``, not F1 ranking; the final-training helpers add it back.
    """
    if classifier not in VALID_TFIDF_CLASSIFIERS:
        raise ValueError(
            f"classifier must be one of {sorted(VALID_TFIDF_CLASSIFIERS)}, "
            f"got {classifier!r}"
        )
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {sorted(VALID_STRATEGIES)}, got {strategy!r}"
        )

    # Memory-conscious search space (target: stable on Colab L4, 53 GB):
    # - max_features=None excluded: unbounded vocab balloons past millions of n-grams
    # - min_df=1 excluded: hapax n-grams aren't pruned, vocab dict explodes
    # - ngram_range=(1,3) excluded: with 150k PT-BR articles, sklearn's
    #   _count_vocab materializes the full trigram dict (~30-100M unique strings,
    #   ~3-10 GB of peak memory PER WORKER) before max_features can prune. Two
    #   parallel workers easily exceed L4 RAM. Trigrams rarely improve F1 on
    #   long news articles; (1,2) is the practical ceiling here. Re-add (1,3)
    #   only on machines with >64 GB and n_jobs=1.
    space: dict = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 5, 10, 20],
        "tfidf__max_df": [0.85, 0.9, 0.95, 1.0],
        "tfidf__max_features": [50_000, 100_000, 200_000],
        "tfidf__sublinear_tf": [True, False],
    }

    prefix = "clf__estimator__" if (multiclass and strategy == "ovr") else "clf__"

    if classifier in ("logreg", "linearsvc"):
        space[f"{prefix}C"] = loguniform(1e-3, 1e2)
        space[f"{prefix}class_weight"] = [None, "balanced"]
    elif classifier == "multinomialnb":
        space[f"{prefix}alpha"] = loguniform(1e-3, 1e1)
        space[f"{prefix}fit_prior"] = [True, False]

    return space


def build_bert_search_space() -> dict:
    """Standard BERT search space (sampled by :func:`_sample_bert_params`).

    Each entry is a tuple ``(kind, *args)`` with ``kind`` in
    ``{"loguniform", "uniform", "int", "choice"}``.
    """
    return {
        "learning_rate": ("loguniform", 1e-5, 5e-5),
        "per_device_train_batch_size": ("choice", [8, 16, 32]),
        "num_train_epochs": ("int", 2, 5),
        "weight_decay": ("loguniform", 1e-3, 1e-1),
        "warmup_ratio": ("uniform", 0.0, 0.2),
        "gradient_accumulation_steps": ("choice", [1, 2, 4]),
    }


# ---------------------------------------------------------------------------
# TF-IDF search
# ---------------------------------------------------------------------------


def _build_search_pipeline(
    classifier: str,
    *,
    multiclass: bool = False,
    strategy: str = "native",
    seed: int = 2026,
) -> Pipeline:
    """Pipeline used during hyperparameter search.

    Never wraps LinearSVC in ``CalibratedClassifierCV`` — calibration is only
    needed to expose ``predict_proba`` at evaluation time, not to rank
    hyperparameters by F1.
    """
    if classifier not in VALID_TFIDF_CLASSIFIERS:
        raise ValueError(
            f"classifier must be one of {sorted(VALID_TFIDF_CLASSIFIERS)}, "
            f"got {classifier!r}"
        )

    vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True)

    if classifier == "logreg":
        base = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
    elif classifier == "linearsvc":
        base = LinearSVC(max_iter=1000, random_state=seed, dual="auto")
    else:  # multinomialnb
        base = MultinomialNB()

    clf = OneVsRestClassifier(base) if (multiclass and strategy == "ovr") else base
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def random_search_tfidf(
    train_val_df: "pd.DataFrame",
    *,
    classifier: str,
    label_column: str = "label",
    multiclass: bool = False,
    strategy: str = "native",
    n_iter: int = 60,
    cv_n_splits: int = 5,
    cv_seed: int = 2027,
    scoring: str | None = None,
    n_jobs: int = 2,
    seed: int = 2026,
    search_space: dict | None = None,
    verbose: int = 1,
) -> SearchResult:
    """Run ``RandomizedSearchCV`` over a TF-IDF pipeline.

    The inner CV uses ``cv_seed`` (default 2027) so it produces *different* folds
    from ``artifacts/splits/cv_folds.json`` (historically seed=42, current default
    seed=2026) — the post-search ``cv_5fold`` regime then provides an independent
    variance estimate.

    ``n_jobs`` defaults to 2 because Colab/free instances run out of memory
    with ``-1`` once the pool exceeds ~100k samples (each worker pickles the
    full text list + its own TF-IDF matrix). Bump to ``-1`` only on machines
    with >32 GB RAM. ``pre_dispatch="n_jobs"`` caps in-flight jobs to avoid
    a backlog amplifying memory pressure.

    Returns a :class:`SearchResult` whose ``best_params`` keep the sklearn
    pipeline-prefixed keys; use :func:`tfidf_best_params_to_kwargs` to strip
    prefixes before instantiating ``TfidfTrainingConfig``.
    """
    if scoring is None:
        scoring = "f1_macro" if multiclass else "f1"
    if search_space is None:
        search_space = build_tfidf_search_space(
            classifier, multiclass=multiclass, strategy=strategy,
        )

    pipeline = _build_search_pipeline(
        classifier, multiclass=multiclass, strategy=strategy, seed=seed,
    )
    cv = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=cv_seed)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=search_space,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        pre_dispatch="n_jobs",  # cap in-flight jobs; default "2*n_jobs" can OOM
        random_state=seed,
        verbose=verbose,
        refit=False,
        return_train_score=False,
    )

    X = train_val_df["text"].fillna("").tolist()
    y = train_val_df[label_column].tolist()

    t0 = time.perf_counter()
    search.fit(X, y)
    search_seconds = time.perf_counter() - t0

    return SearchResult(
        best_params=dict(search.best_params_),
        best_score=float(search.best_score_),
        n_trials=int(len(search.cv_results_["params"])),
        search_space=_describe_tfidf_space(search_space),
        search_seconds=round(search_seconds, 2),
        trials=_summarize_sklearn_cv_results(search.cv_results_),
        scoring=scoring,
    )


def tfidf_best_params_to_kwargs(
    best_params: dict,
) -> dict:
    """Strip sklearn pipeline prefixes from ``best_params`` for use in TfidfTrainingConfig.

    Handles ``tfidf__*``, ``clf__*`` and ``clf__estimator__*`` prefixes
    (the last for OneVsRestClassifier multiclass-OvR pipelines).
    """
    prefixes = ("clf__estimator__estimator__", "clf__estimator__", "clf__", "tfidf__")
    out: dict = {}
    for key, value in best_params.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                out[key[len(prefix):]] = value
                break
        else:
            out[key] = value
    if "ngram_range" in out and isinstance(out["ngram_range"], list):
        out["ngram_range"] = tuple(out["ngram_range"])
    return out


# ---------------------------------------------------------------------------
# BERT search
# ---------------------------------------------------------------------------


def _sample_bert_params(space: dict, rng: np.random.Generator) -> dict:
    params: dict = {}
    for key, spec in space.items():
        kind = spec[0]
        if kind == "loguniform":
            lo, hi = spec[1], spec[2]
            params[key] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        elif kind == "uniform":
            lo, hi = spec[1], spec[2]
            params[key] = float(rng.uniform(lo, hi))
        elif kind == "int":
            lo, hi = spec[1], spec[2]
            params[key] = int(rng.integers(lo, hi + 1))
        elif kind == "choice":
            choices = spec[1]
            params[key] = choices[int(rng.integers(0, len(choices)))]
        else:
            raise ValueError(f"Unknown sampling kind: {kind!r}")
    return params


def random_search_bert(
    train_df: "pd.DataFrame",
    val_df: "pd.DataFrame",
    *,
    model_name: str,
    work_dir: str | Path,
    label_column: str = "label",
    multiclass: bool = False,
    label_set: tuple[str, ...] = (),
    base_config_overrides: dict | None = None,
    search_space: dict | None = None,
    n_iter: int = 25,
    scoring: str | None = None,
    seed: int = 2026,
    keep_trial_artifacts: bool = False,
) -> SearchResult:
    """Custom random search for BERT (HF Trainer doesn't fit sklearn's CV API).

    Each trial fine-tunes from ``model_name`` on ``train_df`` and scores on
    ``val_df`` (single inner split, not CV). Trial directories are wiped after
    each trial unless ``keep_trial_artifacts=True``.
    """
    from .bert import (
        BertMulticlassConfig,
        BertTrainingConfig,
        train_bert_classifier,
        train_bert_multiclass,
    )

    if scoring is None:
        scoring = "macro_f1" if multiclass else "f1"
    if search_space is None:
        search_space = build_bert_search_space()
    if multiclass and not label_set:
        raise ValueError("label_set must be a non-empty tuple when multiclass=True")

    rng = np.random.default_rng(seed)
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    overrides = base_config_overrides or {}
    trials: list[dict] = []
    t_total = time.perf_counter()

    for i in range(n_iter):
        sampled = _sample_bert_params(search_space, rng)
        trial_dir = work_path / f"trial_{i:03d}"
        trial_dir.mkdir(exist_ok=True)

        config_kwargs = {
            **overrides,
            **sampled,
            "model_name": model_name,
            "seed": seed,
        }

        try:
            t0 = time.perf_counter()
            if multiclass:
                cfg = BertMulticlassConfig(label_set=tuple(label_set), **config_kwargs)
                result = train_bert_multiclass(
                    train_df, val_df,
                    label_column=label_column, run_dir=trial_dir, config=cfg,
                )
            else:
                cfg = BertTrainingConfig(**config_kwargs)
                result = train_bert_classifier(
                    train_df, val_df, run_dir=trial_dir, config=cfg,
                )
            duration = time.perf_counter() - t0

            metrics = result["metrics"]
            score = float(
                metrics.get(f"eval_{scoring}",
                            metrics.get(scoring, float("nan"))),
            )

            trials.append({
                "trial": i,
                "params": _to_jsonable(sampled),
                "score": round(score, 4),
                "duration_seconds": round(duration, 2),
            })
        except Exception as exc:  # noqa: BLE001
            trials.append({
                "trial": i,
                "params": _to_jsonable(sampled),
                "score": float("-inf"),
                "duration_seconds": None,
                "error": str(exc),
            })
        finally:
            if not keep_trial_artifacts:
                _wipe_dir(trial_dir)
            _free_gpu_memory()

    search_seconds = time.perf_counter() - t_total
    valid = [t for t in trials if t["score"] != float("-inf")]
    if not valid:
        raise RuntimeError(
            f"All {n_iter} BERT trials failed. See trial errors in the returned object."
        )
    best = max(valid, key=lambda t: t["score"])

    return SearchResult(
        best_params=best["params"],
        best_score=best["score"],
        n_trials=int(n_iter),
        search_space=_describe_bert_space(search_space),
        search_seconds=round(search_seconds, 2),
        trials=trials,
        scoring=scoring,
    )


# ---------------------------------------------------------------------------
# Helpers — serialization, cleanup
# ---------------------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)


_DIST_NAME_ALIASES = {
    # scipy.stats.loguniform is implemented as reciprocal_gen
    "reciprocal": "loguniform",
}


def _describe_tfidf_space(space: dict) -> dict:
    out: dict = {}
    for key, value in space.items():
        if isinstance(value, list):
            out[key] = {"type": "choice", "values": [_to_jsonable(v) for v in value]}
        elif hasattr(value, "rvs") and hasattr(value, "args"):
            args = list(getattr(value, "args", ()))
            dist = getattr(value, "dist", value)
            raw_name = type(dist).__name__.replace("_gen", "")
            name = _DIST_NAME_ALIASES.get(raw_name, raw_name)
            out[key] = {"type": name, "args": args}
        else:
            out[key] = {"type": "scalar", "value": _to_jsonable(value)}
    return out


def _describe_bert_space(space: dict) -> dict:
    out: dict = {}
    for key, spec in space.items():
        kind = spec[0]
        if kind == "choice":
            out[key] = {"type": "choice", "values": list(spec[1])}
        elif kind == "loguniform":
            out[key] = {"type": "loguniform", "low": spec[1], "high": spec[2]}
        elif kind == "uniform":
            out[key] = {"type": "uniform", "low": spec[1], "high": spec[2]}
        elif kind == "int":
            out[key] = {"type": "int", "low": spec[1], "high": spec[2]}
        else:
            out[key] = {"type": kind, "spec": list(spec)}
    return out


def _summarize_sklearn_cv_results(cv_results: dict) -> list[dict]:
    """Collapse RandomizedSearchCV.cv_results_ into one row per trial."""
    n_trials = len(cv_results["params"])
    out: list[dict] = []
    for i in range(n_trials):
        out.append({
            "trial": i,
            "params": _to_jsonable(dict(cv_results["params"][i])),
            "mean_test_score": float(cv_results["mean_test_score"][i]),
            "std_test_score": float(cv_results["std_test_score"][i]),
            "mean_fit_time": float(cv_results["mean_fit_time"][i]),
            "rank_test_score": int(cv_results["rank_test_score"][i]),
        })
    return out


def _wipe_dir(path: Path) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def _free_gpu_memory() -> None:
    try:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
