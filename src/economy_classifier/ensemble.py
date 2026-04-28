"""Ensemble strategies: voting, stacking and agreement analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score


def majority_vote(
    predictions: dict[str, pd.Series],
    *,
    threshold: int = 4,
) -> pd.DataFrame:
    """Binary majority vote across classifiers.

    A sample is positive when at least *threshold* classifiers predict 1.
    """
    votes = pd.DataFrame(predictions)
    vote_sum = votes.sum(axis=1)
    y_pred = (vote_sum >= threshold).astype(int)
    return pd.DataFrame({"y_pred": y_pred})


def weighted_vote(
    scores: dict[str, pd.Series],
    weights: dict[str, float],
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Weighted average of continuous scores with F1-based weights."""
    score_df = pd.DataFrame(scores)
    w = np.array([weights[m] for m in score_df.columns])
    w_sum = w.sum()
    if w_sum == 0:
        w_normalized = np.ones_like(w) / len(w)
    else:
        w_normalized = w / w_sum

    y_score = score_df.values @ w_normalized
    y_pred = (y_score >= threshold).astype(int)

    return pd.DataFrame({
        "y_pred": y_pred,
        "y_score": np.round(y_score, 4),
    })


def _stack_features(
    features_per_model: dict[str, pd.Series | pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate per-model features into a single named DataFrame.

    Each model contributes one column when given a Series (binary case,
    ``y_score``) or ``n_classes`` columns when given a DataFrame
    (multiclass case, one column per ``y_proba_<class>``). Column names
    are ``<model>`` for Series inputs and ``<model>__<class>`` for
    DataFrame inputs, so the meta-classifier's ``feature_names_in_``
    stays aligned across fit and predict calls.
    """
    parts: list[pd.DataFrame] = []
    for name, feats in features_per_model.items():
        if isinstance(feats, pd.Series):
            parts.append(feats.to_frame(name=name).reset_index(drop=True))
        elif isinstance(feats, pd.DataFrame):
            renamed = feats.rename(columns=lambda c: f"{name}__{c}")
            parts.append(renamed.reset_index(drop=True))
        else:
            raise TypeError(
                f"features for {name!r} must be Series or DataFrame, got {type(feats).__name__}"
            )
    return pd.concat(parts, axis=1)


def train_stacking_classifier(
    val_features: dict[str, pd.Series | pd.DataFrame],
    val_true: pd.Series,
    *,
    seed: int = 42,
) -> LogisticRegression:
    """Train a meta-classifier on validation-set features (no leakage).

    ``val_features`` accepts either:

    - ``dict[model -> Series]`` for binary stacking (one ``y_score`` per model)
    - ``dict[model -> DataFrame]`` for multiclass stacking (one column per
      class, in a fixed order). All DataFrames must share the same columns
      so the meta features stay aligned.
    """
    X = _stack_features(val_features)
    y = np.asarray(val_true)
    clf = LogisticRegression(random_state=seed, solver="lbfgs", max_iter=1000)
    clf.fit(X, y)
    return clf


def predict_stacking(
    model: LogisticRegression,
    test_features: dict[str, pd.Series | pd.DataFrame],
) -> pd.DataFrame:
    """Predict using the stacking meta-classifier.

    Binary input returns ``y_pred`` + ``y_score``. Multiclass input returns
    ``y_pred`` + one ``y_proba_<class>`` column per learned class.
    """
    X = _stack_features(test_features)
    y_pred = model.predict(X)
    proba = model.predict_proba(X)

    if proba.shape[1] == 2:
        return pd.DataFrame({
            "y_pred": y_pred.tolist(),
            "y_score": np.round(proba[:, 1], 4).tolist(),
        })

    out = pd.DataFrame({"y_pred": y_pred.tolist()})
    for j, cls in enumerate(model.classes_):
        out[f"y_proba_{cls}"] = np.round(proba[:, j], 4)
    return out


def save_stacking_classifier(
    model: LogisticRegression,
    path: str | Path,
    *,
    feature_names: list[str] | None = None,
) -> None:
    """Persist a stacking meta-classifier and its metadata.

    For binary meta (1D ``coef_``) the metadata records a single coefficient
    vector and intercept. For multiclass meta (2D ``coef_``) the metadata
    records per-class coefficients and intercepts, plus the class order.
    """
    import joblib

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "meta_classifier.joblib")

    coef = np.asarray(model.coef_)
    intercept = np.asarray(model.intercept_)

    meta: dict = {
        "n_features": int(model.n_features_in_),
        "classes": model.classes_.tolist(),
    }
    if coef.shape[0] == 1:
        meta["coefficients"] = coef[0].tolist()
        meta["intercept"] = float(intercept[0])
    else:
        meta["coefficients_per_class"] = coef.tolist()
        meta["intercepts"] = intercept.tolist()

    if feature_names is None and getattr(model, "feature_names_in_", None) is not None:
        feature_names = list(model.feature_names_in_)
    if feature_names:
        meta["feature_names"] = feature_names
    (path / "meta_classifier_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
    )


def load_stacking_classifier(path: str | Path) -> LogisticRegression:
    """Load a persisted stacking meta-classifier."""
    import joblib

    model_file = Path(path) / "meta_classifier.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"Meta-classifier not found at {model_file}")
    return joblib.load(model_file)


def compute_agreement_matrix(
    predictions: dict[str, pd.Series],
) -> pd.DataFrame:
    """NxN matrix of Cohen's Kappa between every pair of classifiers."""
    methods = list(predictions.keys())
    n = len(methods)
    matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            kappa = cohen_kappa_score(predictions[methods[i]], predictions[methods[j]])
            matrix[i, j] = round(kappa, 4)
            matrix[j, i] = round(kappa, 4)

    return pd.DataFrame(matrix, index=methods, columns=methods)


def compute_fleiss_kappa(
    predictions: dict[str, pd.Series],
    *,
    categories: list | None = None,
) -> float:
    """Fleiss' Kappa for multiple raters across k categories.

    Each rater (key in ``predictions``) assigns a category to each subject.
    Categories may be integers (e.g. binary 0/1) or strings (e.g. multiclass
    labels). When ``categories`` is omitted it is inferred from the union
    of observed values, so empty categories that exist in the schema but
    not in the data are ignored — pass ``categories`` explicitly to keep
    the schema fixed across runs.
    """
    votes = pd.DataFrame(predictions)
    n_subjects, n_raters = votes.shape

    if categories is None:
        categories = sorted(pd.unique(votes.values.ravel()).tolist())

    # N x k matrix: counts of raters assigning each category per subject
    counts = np.column_stack([
        (votes == cat).sum(axis=1).values for cat in categories
    ])

    # P_i: proportion of agreeing pairs per subject
    p_i = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = np.mean(p_i)

    # P_j: proportion of assignments to each category
    p_j = counts.sum(axis=0) / (n_subjects * n_raters)
    p_e = np.sum(p_j ** 2)

    if abs(1 - p_e) < 1e-10:
        return 1.0 if abs(p_bar - 1.0) < 1e-10 else 0.0

    return float((p_bar - p_e) / (1 - p_e))


def optimize_voting_threshold(
    val_scores: dict[str, pd.Series],
    val_true: pd.Series,
    weights: dict[str, float],
    *,
    grid_start: float = 0.3,
    grid_stop: float = 0.71,
    grid_step: float = 0.01,
) -> dict:
    """Grid search for the best weighted-vote threshold on validation set.

    Returns dict with best_threshold, best_f1, and all_results.
    """
    score_df = pd.DataFrame(val_scores)
    w = np.array([weights[m] for m in score_df.columns])
    w_sum = w.sum()
    w_norm = w / w_sum if w_sum > 0 else np.ones_like(w) / len(w)
    combined = score_df.values @ w_norm
    y_true = np.asarray(val_true)

    thresholds = np.arange(grid_start, grid_stop, grid_step)
    results = []
    best_f1 = -1.0
    best_threshold = 0.5

    for t in thresholds:
        y_pred = (combined >= t).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        results.append({"threshold": round(float(t), 4), "f1": round(f1, 4)})
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    return {
        "best_threshold": round(best_threshold, 4),
        "best_f1": round(best_f1, 4),
        "all_results": results,
    }


def load_run_predictions(run_dir: str | Path, *, split: str) -> pd.DataFrame | None:
    """Load predictions from a run directory.

    Handles both naming conventions:
    - Legacy BERT runs: predictions_val.csv / predictions_test.csv
    - Standard runs (TF-IDF, BERT, ensembles): predictions.csv

    Returns None if the file does not exist.
    """
    run_path = Path(run_dir)

    split_file = run_path / f"predictions_{split}.csv"
    if split_file.exists():
        return pd.read_csv(split_file)

    generic_file = run_path / "predictions.csv"
    if generic_file.exists():
        return pd.read_csv(generic_file)

    return None


def discover_runs(runs_dir: str | Path) -> dict[str, dict]:
    """Scan run directories and return metadata indexed by method name.

    Returns {method_name: {"run_dir": Path, "stage": str, "summary": dict}}.
    """
    runs_path = Path(runs_dir)
    discovered = {}

    for run_dir in sorted(runs_path.iterdir()):
        meta_file = run_dir / "run_metadata.json"
        if not meta_file.exists():
            continue

        meta = json.loads(meta_file.read_text())
        stage = meta.get("stage", "")
        summary = meta.get("summary", {})

        # Extract method name depending on stage
        if stage == "bert-training":
            method = summary.get("variant", run_dir.name)
        else:
            method = summary.get("method", run_dir.name)

        discovered[method] = {
            "run_dir": run_dir,
            "stage": stage,
            "summary": summary,
        }

    return discovered


def compute_contingency_table(
    predictions: dict[str, pd.Series],
    y_true: pd.Series,
) -> pd.DataFrame:
    """Agreement level (1..N methods) vs actual class contingency table."""
    votes = pd.DataFrame(predictions)
    agreement_level = votes.sum(axis=1)
    y_true = pd.Series(np.asarray(y_true), name="y_true")
    agreement_level.name = "agreement"

    table = pd.crosstab(agreement_level, y_true, margins=True)
    return table
