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
