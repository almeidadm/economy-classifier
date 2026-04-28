"""Project-wide helpers for paths, versioning and run metadata."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
SPLITS_DIR = ARTIFACTS_DIR / "splits"


def _run_git_command(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def get_git_commit_short() -> str:
    """Return the short commit hash or ``unknown`` outside git history."""
    return _run_git_command("rev-parse", "--short", "HEAD") or "unknown"


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    """Build a filesystem-safe slug from a human-readable label."""
    value = unicodedata.normalize("NFD", value)
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    allowed = []
    for char in value.lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {" ", "-", "_", "/"}:
            allowed.append("-")
    slug = "".join(allowed).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


def build_run_id(stage: str, run_name: str | None = None) -> str:
    """Create a run identifier with timestamp, stage and suffix."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = slugify(run_name or "run")
    return f"{timestamp}-{slugify(stage)}-{suffix}"


def create_run_directory(stage: str, run_name: str | None = None) -> Path:
    """Create a timestamped run directory under ``artifacts/runs``."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / build_run_id(stage, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_run_metadata(
    *,
    run_dir: Path,
    stage: str,
    parameters: dict,
    inputs: dict,
    outputs: dict,
    summary: dict,
    timing: dict,
) -> dict:
    """Build a metadata dict for a pipeline run."""
    return {
        "run_id": run_dir.name,
        "stage": stage,
        "git_commit": get_git_commit_short(),
        "generated_at": utc_now_iso(),
        "parameters": parameters,
        "inputs": inputs,
        "outputs": outputs,
        "summary": summary,
        "timing": timing,
    }


def persist_run_artifacts(
    *,
    run_dir: Path,
    metadata: dict,
    predictions: pd.DataFrame | None = None,
    metrics: dict | None = None,
) -> None:
    """Persist run metadata, predictions and metrics to the run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )

    if predictions is not None:
        predictions.to_csv(run_dir / "predictions.csv", index=False)

    if metrics is not None:
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False)
        )


# ---------------------------------------------------------------------------
# Result card (Fase 2) — schema unico para comparacao entre modelos
# ---------------------------------------------------------------------------

VALID_TASKS = {"binary", "multiclass"}
VALID_REGIMES = {"fixed_split", "cv_5fold", "test_set"}

# Scoring strings expected per task. TF-IDF uses sklearn naming ("f1_macro"),
# BERT uses an internal name ("macro_f1"); both are accepted.
_VALID_SCORING_BY_TASK = {
    "binary": {"f1"},
    "multiclass": {"f1_macro", "macro_f1"},
}


def compute_artifact_size_mb(path: Path) -> float:
    """File size in MB if *path* is a file, sum of files if directory."""
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def build_result_card(
    *,
    model_id: str,
    task: str,
    regime: str,
    metrics: dict,
    cost: dict,
    config: dict,
    n_train_samples: int | None = None,
    n_eval_samples: int | None = None,
    predictions_path: str | None = None,
    notes: str | None = None,
    hyperparameter_search: dict | None = None,
) -> dict:
    """Standardized result card for cross-model comparison.

    All notebooks emit this JSON so notebook 42 can aggregate without per-model logic.
    The optional ``hyperparameter_search`` field carries the
    :meth:`SearchResult.card_payload` from a preceding RandomizedSearchCV run.
    """
    if task not in VALID_TASKS:
        raise ValueError(f"task must be one of {VALID_TASKS}, got {task!r}")
    if regime not in VALID_REGIMES:
        raise ValueError(f"regime must be one of {VALID_REGIMES}, got {regime!r}")

    # Guard against the class of bug where a SearchResult.card_payload from a
    # different task is passed in (e.g. multiclass payload reused in a binary
    # card). Catches scoring/task mismatch at the boundary.
    if hyperparameter_search is not None:
        hp_scoring = hyperparameter_search.get("scoring")
        if hp_scoring is not None:
            allowed = _VALID_SCORING_BY_TASK[task]
            if hp_scoring not in allowed:
                raise ValueError(
                    f"hyperparameter_search.scoring={hp_scoring!r} is incompatible "
                    f"with task={task!r}; expected one of {sorted(allowed)}. "
                    f"Likely a SearchResult from the other task was passed."
                )

    return {
        "model_id": model_id,
        "task": task,
        "regime": regime,
        "metrics": metrics,
        "cost": cost,
        "config": config,
        "n_train_samples": n_train_samples,
        "n_eval_samples": n_eval_samples,
        "predictions_path": predictions_path,
        "notes": notes,
        "hyperparameter_search": hyperparameter_search,
        "git_commit": get_git_commit_short(),
        "generated_at": utc_now_iso(),
    }


def persist_result_card(card: dict, run_dir: Path) -> Path:
    """Write *card* as ``result_card.json`` inside *run_dir* and return its path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "result_card.json"
    out.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
