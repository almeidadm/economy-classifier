"""Visualization utilities — figure generation in PNG 300 DPI + PDF."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    roc_auc_score,
)


def configure_style() -> None:
    """Apply the project-wide matplotlib style (REQUIREMENTS section 6.1)."""
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })
    sns.set_style("whitegrid")


def save_figure(fig: Figure, path: Path, name: str) -> dict[str, Path]:
    """Save figure as PNG (300 DPI) and PDF. Returns paths dict."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    png_path = path / f"{name}.png"
    pdf_path = path / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return {"png": png_path, "pdf": pdf_path}


def plot_confusion_matrix(y_true, y_pred, *, title: str = "") -> Figure:
    """Heatmap confusion matrix with counts."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["outros", "mercado"], ax=ax, cmap="Blues",
    )
    if title:
        ax.set_title(title)
    return fig


def plot_pr_curve(y_true, y_score, *, title: str = "") -> Figure:
    """Precision-recall curve."""
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    if title:
        ax.set_title(title)
    return fig


def plot_roc_curve(y_true, y_score, *, title: str = "") -> Figure:
    """ROC curve with AUC annotation."""
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    auc = roc_auc_score(y_true, y_score)
    ax.annotate(f"AUC = {auc:.3f}", xy=(0.6, 0.3), fontsize=12)
    if title:
        ax.set_title(title)
    return fig


def plot_comparative_barplot(metrics_df, *, metric: str = "f1") -> Figure:
    """Horizontal barplot comparing a metric across methods."""
    fig, ax = plt.subplots(figsize=(8, max(4, len(metrics_df) * 0.5)))
    sns.barplot(data=metrics_df, x=metric, y="method", ax=ax, orient="h")
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("")
    ax.set_title(f"Comparativo — {metric.upper()}")
    return fig
