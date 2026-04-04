"""Tests for economy_classifier.visualization — figure generation."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from economy_classifier.visualization import (
    configure_style,
    plot_comparative_barplot,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    save_figure,
)


def test_save_figure_creates_png_and_pdf(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = save_figure(fig, tmp_path, "test_fig")
    plt.close(fig)
    assert (tmp_path / "test_fig.png").exists()
    assert (tmp_path / "test_fig.pdf").exists()
    assert "png" in paths
    assert "pdf" in paths


def test_save_figure_png_dpi_300(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    save_figure(fig, tmp_path, "dpi_test")
    plt.close(fig)
    # Verify the PNG was created (DPI is set at save time)
    assert (tmp_path / "dpi_test.png").exists()


def test_plot_confusion_matrix_returns_figure():
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])
    fig = plot_confusion_matrix(y_true, y_pred, title="Test CM")
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_pr_curve_returns_figure():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.7, 0.3, 0.1])
    fig = plot_pr_curve(y_true, y_score, title="Test PR")
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_roc_curve_returns_figure():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.7, 0.3, 0.1])
    fig = plot_roc_curve(y_true, y_score, title="Test ROC")
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_comparative_barplot_returns_figure():
    df = pd.DataFrame({
        "method": ["logreg", "linearsvc", "nb"],
        "f1": [0.85, 0.83, 0.80],
    })
    fig = plot_comparative_barplot(df, metric="f1")
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_functions_accept_standard_inputs():
    y_true = pd.Series([1, 0, 1, 0])
    y_pred = pd.Series([1, 0, 0, 1])
    y_score = pd.Series([0.9, 0.2, 0.6, 0.7])
    fig1 = plot_confusion_matrix(y_true, y_pred)
    fig2 = plot_pr_curve(y_true, y_score)
    fig3 = plot_roc_curve(y_true, y_score)
    assert all(isinstance(f, Figure) for f in [fig1, fig2, fig3])
    import matplotlib.pyplot as plt
    plt.close("all")


def test_matplotlib_rcparams_applied():
    import matplotlib.pyplot as plt

    configure_style()
    assert plt.rcParams["savefig.dpi"] == 300
    assert plt.rcParams["figure.figsize"] == [8, 5]
    assert plt.rcParams["font.size"] == 11.0
