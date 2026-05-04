"""Qualitative error analysis — pools, sampling, annotation templates.

Companion to ``evaluation.py``: where evaluation reports aggregate metrics,
this module supports the qualitative review of individual errors that
underpins the construct-validity defense in the dissertation.

Pipeline (orchestrated from ``notebooks/44_analise_qualitativa_erros.ipynb``)::

    predictions.csv + test.parquet
        -> load_predictions_with_text()            # join text + editorial metadata
        -> build_{binary,multiclass,disagreement}_error_pool()
        -> summarize_errors_by_*()                  # pre-annotation overview
        -> stratified_error_sample()                # reproducible sample
        -> export_annotation_template()             # CSV for manual annotation
        -> [manual annotation in spreadsheet]
        -> load_annotated_sample() + summarize_annotations()

Annotation schema (controlled vocabulary in ``ANNOTATION_TYPES``) lets the
dissertation report, e.g., "of 100 false positives, 38% are
``rotulagem_editorial`` (text was thematically about economy but the editor
placed it elsewhere); adjusting for label noise, F1 rises from 0.82 to 0.89."

All sampling uses ``DEFAULT_SEED = 2026`` (project-wide seed) for reproducibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_SEED: int = 2026

# Controlled vocabulary for manual annotation. The four buckets answer:
# "is this error the model's fault, or a property of the editorial labels?"
ANNOTATION_TYPES: tuple[str, ...] = (
    "rotulagem_editorial",  # text is thematically class A, editor labelled B
    "tema_misto",           # text crosses themes; both labels are defensible
    "modelo_erra",          # text is clearly class A; model predicted B
    "ambiguo",              # text has no clear theme to classify
)

ANNOTATION_COLUMNS: tuple[str, ...] = (
    "subtema_real",
    "tipo_erro_anotado",
    "editorialmente_economia",
    "signal_palavras",
    "notas",
)

# Columns that must exist in any predictions.csv emitted by the project.
_REQUIRED_PRED_COLUMNS: tuple[str, ...] = ("index", "y_true", "y_pred", "method")


# ---------------------------------------------------------------------------
# Loading & joining
# ---------------------------------------------------------------------------


def load_predictions_with_text(
    predictions: pd.DataFrame | str | Path,
    test_df: pd.DataFrame,
    *,
    text_columns: Sequence[str] = ("title", "text", "category", "subcategory", "date", "link"),
) -> pd.DataFrame:
    """Join a predictions table to the test corpus on the original index.

    ``predictions`` may be a DataFrame or a path to a CSV in the project's
    standard schema (``index, y_true, y_pred, [y_score,] method[, label]``).

    Returns a DataFrame indexed positionally with one row per prediction,
    carrying both the prediction columns and the requested ``text_columns``
    from ``test_df``. Rows whose ``index`` is missing from ``test_df`` are
    dropped (with a count surfaced via ``attrs['n_dropped']``).
    """
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.read_csv(predictions)

    missing = [c for c in _REQUIRED_PRED_COLUMNS if c not in predictions.columns]
    if missing:
        raise ValueError(
            f"predictions is missing required columns {missing}; "
            f"got {list(predictions.columns)}"
        )

    available_text_cols = [c for c in text_columns if c in test_df.columns]
    test_subset = test_df[available_text_cols].copy()
    test_subset.index.name = "index"

    joined = predictions.merge(
        test_subset, left_on="index", right_index=True, how="left", validate="many_to_one",
    )
    n_dropped = int(joined[available_text_cols[0]].isna().sum()) if available_text_cols else 0
    if n_dropped:
        joined = joined.dropna(subset=[available_text_cols[0]])
    joined.attrs["n_dropped"] = n_dropped
    return joined.reset_index(drop=True)


def detect_task(joined: pd.DataFrame) -> str:
    """Return ``'binary'`` or ``'multiclass'`` based on ``y_true`` dtype/values."""
    y_true = joined["y_true"]
    if pd.api.types.is_numeric_dtype(y_true):
        unique = set(pd.unique(y_true.dropna()))
        if unique <= {0, 1, 0.0, 1.0}:
            return "binary"
    return "multiclass"


# ---------------------------------------------------------------------------
# Error pool builders
# ---------------------------------------------------------------------------


def build_binary_error_pool(joined: pd.DataFrame) -> pd.DataFrame:
    """Return rows where ``y_true != y_pred``, tagged with ``error_type``.

    ``error_type`` is ``'FP'`` (true=0, pred=1) or ``'FN'`` (true=1, pred=0).
    Requires binary labels in ``{0, 1}``.
    """
    if detect_task(joined) != "binary":
        raise ValueError("build_binary_error_pool expects a binary task; got multiclass.")

    y_true = joined["y_true"].astype(int)
    y_pred = joined["y_pred"].astype(int)
    errors = joined[y_true != y_pred].copy()
    errors["error_type"] = np.where(
        errors["y_pred"].astype(int) == 1, "FP", "FN",
    )
    return errors.reset_index(drop=True)


def build_multiclass_error_pool(
    joined: pd.DataFrame,
    *,
    focus_classes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return rows where ``y_true != y_pred``, tagged with directional ``error_type``.

    ``error_type`` is the string ``"{true}->{pred}"`` (e.g. ``"colunas->mercado"``).
    If ``focus_classes`` is given, keep only rows where the true label OR the
    predicted label is in that set — useful for zooming in on editorially
    heterogeneous classes (``colunas``, ``outros``, ``mercado``).
    """
    if detect_task(joined) != "multiclass":
        raise ValueError("build_multiclass_error_pool expects a multiclass task; got binary.")

    errors = joined[joined["y_true"].astype(str) != joined["y_pred"].astype(str)].copy()
    errors["error_type"] = (
        errors["y_true"].astype(str) + "->" + errors["y_pred"].astype(str)
    )
    if focus_classes is not None:
        keep = set(focus_classes)
        mask = errors["y_true"].astype(str).isin(keep) | errors["y_pred"].astype(str).isin(keep)
        errors = errors[mask]
    return errors.reset_index(drop=True)


def build_disagreement_pool(
    predictions_by_method: dict[str, pd.DataFrame],
    test_df: pd.DataFrame,
    *,
    text_columns: Sequence[str] = ("title", "text", "category", "subcategory", "date", "link"),
) -> pd.DataFrame:
    """Cross-method disagreement on the same ``index``, tagged by pattern.

    Input is a dict ``{method_name: predictions_df}``; every frame must share
    the same task (binary or multiclass), the same set of indices, and the
    standard predictions schema.

    Returns one row per ``index`` with one ``pred_<method>`` column per method,
    a ``y_true`` column, and a ``disagreement_pattern`` ∈
    ``{'all_correct', 'all_wrong', 'majority_wrong_one_right',
    'majority_right_one_wrong', 'split'}``. The first two are filtered out
    (no disagreement to analyze).
    """
    if len(predictions_by_method) < 2:
        raise ValueError("build_disagreement_pool needs at least 2 methods.")

    methods = sorted(predictions_by_method.keys())
    base = None
    for method in methods:
        df = predictions_by_method[method]
        missing = [c for c in _REQUIRED_PRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"method {method!r} predictions missing {missing}")
        slim = df[["index", "y_true", "y_pred"]].rename(columns={"y_pred": f"pred_{method}"})
        if base is None:
            base = slim
        else:
            other = slim.drop(columns=["y_true"])
            base = base.merge(other, on="index", how="inner", validate="one_to_one")

    if base is None or base.empty:
        return pd.DataFrame()

    pred_cols = [f"pred_{m}" for m in methods]
    correctness = pd.DataFrame({
        m: (base[f"pred_{m}"].astype(str) == base["y_true"].astype(str)).astype(int)
        for m in methods
    })
    n_correct = correctness.sum(axis=1)
    k = len(methods)

    pattern = pd.Series("split", index=base.index, dtype="object")
    pattern[n_correct == k] = "all_correct"
    pattern[n_correct == 0] = "all_wrong"
    pattern[n_correct == 1] = "majority_wrong_one_right"
    pattern[n_correct == k - 1] = "majority_right_one_wrong"
    base["disagreement_pattern"] = pattern

    base = base[base["disagreement_pattern"].isin({
        "all_wrong", "majority_wrong_one_right", "majority_right_one_wrong", "split",
    })]

    available_text_cols = [c for c in text_columns if c in test_df.columns]
    if available_text_cols:
        test_subset = test_df[available_text_cols].copy()
        test_subset.index.name = "index"
        base = base.merge(test_subset, left_on="index", right_index=True, how="left")

    return base.reset_index(drop=True)


def cross_binary_multiclass_errors_for_class(
    binary_joined: pd.DataFrame,
    multiclass_joined: pd.DataFrame,
    *,
    target_class: str = "mercado",
    binary_positive_label: int = 1,
    text_columns: Sequence[str] = ("title", "text", "category", "subcategory", "date", "link"),
) -> pd.DataFrame:
    """Link binary and multiclass predictions on shared indices for a target class.

    Filters to rows where ``y_true_binary == binary_positive_label`` **and**
    ``y_true_multi == target_class`` — i.e., news items that genuinely belong
    to the target class. For each such index computes whether each pipeline
    got it right and tags the row with an ``agreement_pattern`` ∈
    ``{'both_correct', 'both_wrong', 'binary_only_correct', 'multi_only_correct'}``.

    Answers: "are the multiclass errors the same as the binary errors?". A
    row with ``binary_only_correct`` is a case where the binary head said
    ``mercado`` but the multiclass head sent the article elsewhere — and vice
    versa for ``multi_only_correct``. ``both_wrong`` is the genuinely hard
    overlap between the two pipelines.

    Inner-joins on ``index``: indices missing from either side are dropped
    (count surfaced via ``attrs['n_dropped_binary_only']`` and
    ``attrs['n_dropped_multi_only']``).
    """
    if detect_task(binary_joined) != "binary":
        raise ValueError("binary_joined must be a binary task")
    if detect_task(multiclass_joined) != "multiclass":
        raise ValueError("multiclass_joined must be a multiclass task")

    binary_idx = set(binary_joined["index"])
    multi_idx = set(multiclass_joined["index"])
    n_binary_only = len(binary_idx - multi_idx)
    n_multi_only = len(multi_idx - binary_idx)

    binary_slim = binary_joined[["index", "y_true", "y_pred"]].rename(columns={
        "y_true": "y_true_binary", "y_pred": "y_pred_binary",
    })
    multi_slim = multiclass_joined[["index", "y_true", "y_pred"]].rename(columns={
        "y_true": "y_true_multi", "y_pred": "y_pred_multi",
    })
    merged = binary_slim.merge(
        multi_slim, on="index", how="inner", validate="one_to_one",
    )

    mask_target = (
        (merged["y_true_binary"].astype(int) == int(binary_positive_label))
        & (merged["y_true_multi"].astype(str) == str(target_class))
    )
    merged = merged[mask_target].copy()

    merged["binary_correct"] = (
        merged["y_pred_binary"].astype(int) == merged["y_true_binary"].astype(int)
    )
    merged["multi_correct"] = (
        merged["y_pred_multi"].astype(str) == merged["y_true_multi"].astype(str)
    )

    pattern = pd.Series("both_correct", index=merged.index, dtype="object")
    pattern[~merged["binary_correct"] & ~merged["multi_correct"]] = "both_wrong"
    pattern[merged["binary_correct"] & ~merged["multi_correct"]] = "binary_only_correct"
    pattern[~merged["binary_correct"] & merged["multi_correct"]] = "multi_only_correct"
    merged["agreement_pattern"] = pattern

    text_cols_present = [c for c in text_columns if c in binary_joined.columns]
    if text_cols_present:
        text_subset = (
            binary_joined[["index", *text_cols_present]]
            .drop_duplicates("index", keep="first")
        )
        merged = merged.merge(
            text_subset, on="index", how="left", validate="one_to_one",
        )

    out = merged.reset_index(drop=True)
    out.attrs["n_dropped_binary_only"] = n_binary_only
    out.attrs["n_dropped_multi_only"] = n_multi_only
    return out


def filter_disagreement_by_true_class(
    disagreement_pool: pd.DataFrame,
    *,
    target_class: str | int,
    y_true_column: str = "y_true",
) -> pd.DataFrame:
    """Restrict a disagreement pool to rows whose ``y_true`` equals ``target_class``.

    Works for both binary (``target_class=1``) and multiclass
    (``target_class="mercado"``) — comparison is done on string-cast values
    so int/str labels mix gracefully. Does not recompute
    ``disagreement_pattern``; only filters.
    """
    if y_true_column not in disagreement_pool.columns:
        raise KeyError(f"column {y_true_column!r} not in disagreement_pool")
    mask = disagreement_pool[y_true_column].astype(str) == str(target_class)
    return disagreement_pool[mask].reset_index(drop=True)


def hard_examples_for_class(
    predictions_by_method: dict[str, pd.DataFrame],
    test_df: pd.DataFrame,
    *,
    target_class: str | int,
    text_columns: Sequence[str] = ("title", "text", "category", "subcategory", "date", "link"),
) -> pd.DataFrame:
    """Rows where ``y_true == target_class`` and **every** method got it wrong.

    Builds a disagreement pool internally and intersects
    ``disagreement_pattern == 'all_wrong'`` with ``y_true == target_class``.
    The result surfaces genuinely hard examples — candidates for
    re-labelling, or evidence that a label is editorially noisy.
    """
    pool = build_disagreement_pool(
        predictions_by_method, test_df, text_columns=text_columns,
    )
    if pool.empty:
        return pool
    mask = (
        (pool["disagreement_pattern"] == "all_wrong")
        & (pool["y_true"].astype(str) == str(target_class))
    )
    return pool[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pre-annotation summaries
# ---------------------------------------------------------------------------


def summarize_errors_by_category(
    error_pool: pd.DataFrame,
    *,
    column: str = "category",
) -> pd.DataFrame:
    """Counts and shares of errors per editorial *column* (default ``category``).

    Use ``column="subcategory"`` for finer granularity, ``column="error_type"``
    for the FP/FN or ``"true->pred"`` direction.
    """
    if column not in error_pool.columns:
        raise KeyError(f"column {column!r} not in error_pool: {list(error_pool.columns)}")
    counts = error_pool[column].fillna("<missing>").value_counts()
    total = int(counts.sum())
    return pd.DataFrame({
        column: counts.index,
        "n_errors": counts.values,
        "share": (counts.values / total).round(4) if total else 0.0,
    })


def summarize_errors_by_confidence(
    error_pool: pd.DataFrame,
    *,
    n_bins: int = 5,
    score_column: str = "y_score",
) -> pd.DataFrame:
    """Distribution of errors across equal-width bins of the score column.

    Returns one row per bin with ``bin_left``, ``bin_right``, ``n_errors``,
    ``share``. High-confidence-yet-wrong errors (top bins) are diagnostic of
    systematic bias and should be inspected qualitatively first.

    Returns an empty DataFrame when the score column is absent (e.g. multiclass
    BERT predictions, which omit ``y_score`` by contract).
    """
    if score_column not in error_pool.columns:
        return pd.DataFrame(columns=["bin_left", "bin_right", "n_errors", "share"])

    scores = pd.to_numeric(error_pool[score_column], errors="coerce").dropna()
    if scores.empty:
        return pd.DataFrame(columns=["bin_left", "bin_right", "n_errors", "share"])

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(scores.to_numpy(), edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins)
    total = int(counts.sum())
    return pd.DataFrame({
        "bin_left": edges[:-1].round(2),
        "bin_right": edges[1:].round(2),
        "n_errors": counts,
        "share": (counts / total).round(4) if total else 0.0,
    })


def summarize_errors_by_text_length(
    error_pool: pd.DataFrame,
    *,
    n_bins: int = 5,
    text_column: str = "text",
) -> pd.DataFrame:
    """Distribution of errors across quintiles of text length (in characters).

    Quintile edges are computed from the error pool itself, so this answers:
    "within errors, are short or long texts overrepresented?".
    """
    if text_column not in error_pool.columns:
        raise KeyError(f"column {text_column!r} not in error_pool")

    lengths = error_pool[text_column].fillna("").astype(str).str.len()
    if lengths.empty:
        return pd.DataFrame(columns=["bin_left", "bin_right", "n_errors", "share"])

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(lengths.to_numpy(), quantiles))
    if len(edges) < 2:
        return pd.DataFrame({
            "bin_left": [int(lengths.min())],
            "bin_right": [int(lengths.max())],
            "n_errors": [len(lengths)],
            "share": [1.0],
        })

    bin_idx = np.clip(np.digitize(lengths.to_numpy(), edges[1:-1]), 0, len(edges) - 2)
    counts = np.bincount(bin_idx, minlength=len(edges) - 1)
    total = int(counts.sum())
    return pd.DataFrame({
        "bin_left": edges[:-1].astype(int),
        "bin_right": edges[1:].astype(int),
        "n_errors": counts,
        "share": (counts / total).round(4) if total else 0.0,
    })


def summarize_errors_by_date(
    error_pool: pd.DataFrame,
    *,
    date_column: str = "date",
    freq: str = "YS",
) -> pd.DataFrame:
    """Errors aggregated over time periods (``freq`` follows pandas offset aliases).

    ``YS`` = year start (one row per year). Use ``MS`` for monthly, ``QS`` for
    quarterly. Useful for detecting concept drift: if errors concentrate in a
    specific period, the model may have learned period-specific cues.
    """
    if date_column not in error_pool.columns:
        raise KeyError(f"column {date_column!r} not in error_pool")

    parsed = pd.to_datetime(error_pool[date_column], errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return pd.DataFrame(columns=["period", "n_errors", "share"])

    grouped = valid.dt.to_period(freq.rstrip("S")[0] if freq.endswith("S") else freq)
    counts = grouped.value_counts().sort_index()
    total = int(counts.sum())
    return pd.DataFrame({
        "period": counts.index.astype(str),
        "n_errors": counts.values,
        "share": (counts.values / total).round(4) if total else 0.0,
    })


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def stratified_error_sample(
    error_pool: pd.DataFrame,
    *,
    n_per_stratum: int | dict[str | int, int],
    stratify_by: str = "error_type",
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Draw up to ``n_per_stratum`` rows from each unique value of ``stratify_by``.

    ``n_per_stratum`` may be:

    - an ``int`` — every stratum gets the same target count (legacy behavior);
    - a ``dict`` mapping stratum value to count — strata absent from the dict
      are skipped silently and their values listed in ``attrs['skipped_strata']``.
      Useful for asymmetric sampling (e.g. ``{"FN": 50, "FP": 30}`` when the
      analysis foregrounds false negatives).

    Strata smaller than the requested count contribute all their rows. The
    output is shuffled deterministically with ``seed``.
    """
    if stratify_by not in error_pool.columns:
        raise KeyError(f"column {stratify_by!r} not in error_pool")

    if isinstance(n_per_stratum, dict):
        if not n_per_stratum:
            raise ValueError("n_per_stratum dict cannot be empty")
        if any(int(n) < 1 for n in n_per_stratum.values()):
            raise ValueError(
                f"n_per_stratum values must be >= 1, got {n_per_stratum}"
            )
        n_dict: dict | None = dict(n_per_stratum)
        default_n: int | None = None
    else:
        if n_per_stratum < 1:
            raise ValueError(f"n_per_stratum must be >= 1, got {n_per_stratum}")
        n_dict = None
        default_n = int(n_per_stratum)

    rng = np.random.RandomState(seed)
    sampled_chunks: list[pd.DataFrame] = []
    skipped: list[object] = []
    for value, chunk in error_pool.groupby(stratify_by, sort=True):
        if n_dict is not None:
            if value not in n_dict:
                skipped.append(value)
                continue
            n_target = int(n_dict[value])
        else:
            assert default_n is not None
            n_target = default_n
        n = min(n_target, len(chunk))
        idx = rng.choice(len(chunk), size=n, replace=False)
        sampled_chunks.append(chunk.iloc[idx])

    if not sampled_chunks:
        out = error_pool.iloc[0:0].copy()
    else:
        concat = pd.concat(sampled_chunks, ignore_index=True)
        shuffle_idx = rng.permutation(len(concat))
        out = concat.iloc[shuffle_idx].reset_index(drop=True)
    if skipped:
        out.attrs["skipped_strata"] = skipped
    return out


# ---------------------------------------------------------------------------
# Annotation template export / import
# ---------------------------------------------------------------------------


def export_annotation_template(
    sample: pd.DataFrame,
    output_path: str | Path,
    *,
    max_text_chars: int = 2000,
    text_column: str = "text",
) -> Path:
    """Write a CSV with context, predictions, and empty annotation columns.

    Columns written (in order):
        index, method, error_type, title, text_preview, text_full_chars,
        category, subcategory, date, link, y_true, y_pred, y_score (if present),
        + ANNOTATION_COLUMNS (empty).

    Long texts are truncated to ``max_text_chars`` and suffixed with ``"…"``.
    The full character count is preserved in ``text_full_chars`` so the
    annotator can decide whether to open the link.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    template = pd.DataFrame(index=sample.index)
    for col in ("index", "method", "error_type", "title"):
        if col in sample.columns:
            template[col] = sample[col].values

    if text_column in sample.columns:
        text_str = sample[text_column].fillna("").astype(str)
        truncated = text_str.where(
            text_str.str.len() <= max_text_chars,
            text_str.str.slice(0, max_text_chars) + "…",
        )
        template["text_preview"] = truncated.values
        template["text_full_chars"] = text_str.str.len().values

    for col in ("category", "subcategory", "date", "link",
                "y_true", "y_pred", "y_score"):
        if col in sample.columns:
            template[col] = sample[col].values

    for col in ANNOTATION_COLUMNS:
        template[col] = ""

    template.to_csv(output_path, index=False)
    return output_path


def load_annotated_sample(
    path: str | Path,
    *,
    require_complete: bool = False,
) -> pd.DataFrame:
    """Read an annotated CSV and validate the controlled vocabulary.

    Raises ``ValueError`` if ``tipo_erro_anotado`` contains values outside
    ``ANNOTATION_TYPES`` (empty strings are allowed unless ``require_complete``).

    When ``require_complete=True``, every row must have a non-empty
    ``tipo_erro_anotado`` — use this before computing summaries for the paper.
    """
    df = pd.read_csv(path)
    missing_cols = [c for c in ANNOTATION_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"annotated file is missing columns {missing_cols}")

    types = df["tipo_erro_anotado"].fillna("").astype(str).str.strip()
    bad = sorted(set(types) - set(ANNOTATION_TYPES) - {""})
    if bad:
        raise ValueError(
            f"tipo_erro_anotado has values outside ANNOTATION_TYPES: {bad}. "
            f"Allowed: {list(ANNOTATION_TYPES)}"
        )
    if require_complete and (types == "").any():
        n_missing = int((types == "").sum())
        raise ValueError(f"{n_missing} row(s) have empty tipo_erro_anotado")

    df["tipo_erro_anotado"] = types
    return df


def summarize_annotations(annotated: pd.DataFrame) -> dict[str, object]:
    """Aggregate counts and shares per ``tipo_erro_anotado`` (and split by error_type).

    The return shape is suitable for direct inclusion in the dissertation:

        {
          "n_total": 100,
          "n_annotated": 100,
          "by_type": {"rotulagem_editorial": {"n": 38, "share": 0.38}, ...},
          "by_error_type": {"FP": {...}, "FN": {...}},  # if error_type column present
          "adjusted_correctness": 0.43,  # share whose error is NOT model_erra
        }

    ``adjusted_correctness`` is the headline number for the construct-validity
    defense: it is the fraction of nominal errors that are *not* clearly the
    model's fault (rotulagem_editorial + tema_misto + ambiguo). The dissertation
    can use it to argue that observed F1 is a lower bound on true performance.
    """
    types = annotated["tipo_erro_anotado"].fillna("").astype(str).str.strip()
    annotated_mask = types != ""
    n_total = int(len(annotated))
    n_annotated = int(annotated_mask.sum())

    def _counts(series: pd.Series) -> dict[str, dict[str, float]]:
        counts = series[series != ""].value_counts()
        total = int(counts.sum())
        return {
            key: {"n": int(value), "share": round(value / total, 4) if total else 0.0}
            for key, value in counts.items()
        }

    summary: dict[str, object] = {
        "n_total": n_total,
        "n_annotated": n_annotated,
        "by_type": _counts(types),
    }

    if "error_type" in annotated.columns:
        summary["by_error_type"] = {
            str(et): _counts(group["tipo_erro_anotado"].fillna("").astype(str).str.strip())
            for et, group in annotated.groupby("error_type", sort=True)
        }

    not_model_erra = types[annotated_mask].isin(
        {"rotulagem_editorial", "tema_misto", "ambiguo"}
    )
    summary["adjusted_correctness"] = (
        round(float(not_model_erra.mean()), 4) if n_annotated else 0.0
    )
    return summary
