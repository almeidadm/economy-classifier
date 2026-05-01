"""Tests for economy_classifier.error_analysis — qualitative error analysis."""

from __future__ import annotations

import pandas as pd
import pytest

from economy_classifier.error_analysis import (
    ANNOTATION_COLUMNS,
    ANNOTATION_TYPES,
    DEFAULT_SEED,
    build_binary_error_pool,
    build_disagreement_pool,
    build_multiclass_error_pool,
    detect_task,
    export_annotation_template,
    load_annotated_sample,
    load_predictions_with_text,
    stratified_error_sample,
    summarize_annotations,
    summarize_errors_by_category,
    summarize_errors_by_confidence,
    summarize_errors_by_date,
    summarize_errors_by_text_length,
)


@pytest.fixture
def test_corpus() -> pd.DataFrame:
    """A 12-row test set with text, category, subcategory, date, link.

    Indices are non-contiguous to exercise the index-based join.
    """
    rows = [
        # idx, title, text, date, category, subcategory, link, label, label_multi
        (1001, "Selic sobe para 13,75%", "Banco Central elevou a taxa Selic " * 30, "2023-03-15", "mercado", "economia", "http://x/1", 1, "mercado"),
        (1002, "Petrobras registra lucro recorde", "Companhia anunciou lucro de R$ 50 bi " * 30, "2023-04-20", "mercado", "empresas", "http://x/2", 1, "mercado"),
        (1003, "Coluna do Reinaldo: economia em queda", "Texto opinativo sobre economia " * 5, "2023-05-10", "colunas", None, "http://x/3", 0, "colunas"),
        (1004, "Coluna sobre cinema", "Crítica do novo filme " * 50, "2023-05-12", "colunas", None, "http://x/4", 0, "colunas"),
        (1005, "Time vence campeonato", "Final emocionante no estádio " * 20, "2023-06-01", "esporte", "futebol", "http://x/5", 0, "esporte"),
        (1006, "Eleição municipal", "Campanha entra na reta final " * 25, "2023-09-15", "poder", "politica", "http://x/6", 0, "poder"),
        (1007, "Reforma tributária aprovada", "Câmara aprova texto base " * 40, "2023-12-22", "poder", "politica", "http://x/7", 0, "poder"),
        (1008, "Festival de cinema", "Mostra em Veneza recebe filme brasileiro " * 15, "2024-02-10", "ilustrada", "cinema", "http://x/8", 0, "ilustrada"),
        (1009, "Ataque na fronteira", "Conflito se intensifica no oriente " * 30, "2024-03-05", "mundo", None, "http://x/9", 0, "mundo"),
        (1010, "Trânsito caótico após chuva", "Avenidas alagadas em SP " * 10, "2024-04-18", "cotidiano", "saopaulo", "http://x/10", 0, "cotidiano"),
        (1011, "Greve nas multinacionais", "Trabalhadores param produção " * 35, "2024-05-22", "cotidiano", "trabalho", "http://x/11", 0, "cotidiano"),
        (1012, "Ciência: nova vacina", "Pesquisadores anunciam imunizante " * 20, "2024-06-30", "ciencia", None, "http://x/12", 0, "outros"),
    ]
    df = pd.DataFrame(rows, columns=[
        "_idx", "title", "text", "date", "category", "subcategory", "link", "label", "label_multi",
    ]).set_index("_idx")
    df.index.name = None
    return df


@pytest.fixture
def binary_predictions() -> pd.DataFrame:
    """Predictions matching test_corpus indices: 4 errors (2 FP, 2 FN)."""
    return pd.DataFrame([
        # mercado correctly predicted (TP)
        {"index": 1001, "y_true": 1, "y_pred": 1, "y_score": 0.95, "method": "model_a"},
        # mercado missed (FN, low confidence)
        {"index": 1002, "y_true": 1, "y_pred": 0, "y_score": 0.32, "method": "model_a"},
        # colunas-about-economy predicted as mercado (FP, classic noise)
        {"index": 1003, "y_true": 0, "y_pred": 1, "y_score": 0.78, "method": "model_a"},
        # colunas-about-cinema correctly predicted as outros
        {"index": 1004, "y_true": 0, "y_pred": 0, "y_score": 0.05, "method": "model_a"},
        # esporte correctly predicted as outros
        {"index": 1005, "y_true": 0, "y_pred": 0, "y_score": 0.10, "method": "model_a"},
        # politics predicted as mercado (FP, high confidence — systematic bias signal)
        {"index": 1006, "y_true": 0, "y_pred": 1, "y_score": 0.88, "method": "model_a"},
        # tributária correctly predicted as outros
        {"index": 1007, "y_true": 0, "y_pred": 0, "y_score": 0.40, "method": "model_a"},
        # cinema correctly predicted as outros
        {"index": 1008, "y_true": 0, "y_pred": 0, "y_score": 0.02, "method": "model_a"},
        # mundo correctly outros
        {"index": 1009, "y_true": 0, "y_pred": 0, "y_score": 0.15, "method": "model_a"},
        # cotidiano correctly outros
        {"index": 1010, "y_true": 0, "y_pred": 0, "y_score": 0.20, "method": "model_a"},
        # greve in cotidiano — mercado-like content missed (FN, this looks ambiguous)
        {"index": 1011, "y_true": 0, "y_pred": 1, "y_score": 0.71, "method": "model_a"},
        # ciencia correctly outros
        {"index": 1012, "y_true": 0, "y_pred": 0, "y_score": 0.08, "method": "model_a"},
    ])


@pytest.fixture
def multiclass_predictions() -> pd.DataFrame:
    """Multiclass predictions with 4 errors, including colunas confusion."""
    return pd.DataFrame([
        {"index": 1001, "y_true": "mercado", "y_pred": "mercado", "method": "model_a"},
        {"index": 1002, "y_true": "mercado", "y_pred": "mercado", "method": "model_a"},
        # colunas about economy → predicted mercado
        {"index": 1003, "y_true": "colunas", "y_pred": "mercado", "method": "model_a"},
        # colunas about cinema → predicted ilustrada
        {"index": 1004, "y_true": "colunas", "y_pred": "ilustrada", "method": "model_a"},
        {"index": 1005, "y_true": "esporte", "y_pred": "esporte", "method": "model_a"},
        # poder → predicted mercado
        {"index": 1006, "y_true": "poder", "y_pred": "mercado", "method": "model_a"},
        {"index": 1007, "y_true": "poder", "y_pred": "poder", "method": "model_a"},
        {"index": 1008, "y_true": "ilustrada", "y_pred": "ilustrada", "method": "model_a"},
        {"index": 1009, "y_true": "mundo", "y_pred": "mundo", "method": "model_a"},
        # cotidiano → predicted poder
        {"index": 1010, "y_true": "cotidiano", "y_pred": "poder", "method": "model_a"},
        {"index": 1011, "y_true": "cotidiano", "y_pred": "cotidiano", "method": "model_a"},
        {"index": 1012, "y_true": "outros", "y_pred": "outros", "method": "model_a"},
    ])


# ---------------------------------------------------------------------------
# load_predictions_with_text
# ---------------------------------------------------------------------------


def test_load_predictions_joins_on_index(binary_predictions, test_corpus):
    joined = load_predictions_with_text(binary_predictions, test_corpus)
    assert set(["title", "text", "category", "subcategory", "date", "link"]).issubset(joined.columns)
    assert set(["index", "y_true", "y_pred", "y_score", "method"]).issubset(joined.columns)
    assert len(joined) == len(binary_predictions)
    row = joined[joined["index"] == 1001].iloc[0]
    assert row["category"] == "mercado"
    assert row["title"].startswith("Selic")


def test_load_predictions_drops_orphans(binary_predictions, test_corpus):
    extra = pd.concat([binary_predictions, pd.DataFrame([
        {"index": 9999, "y_true": 0, "y_pred": 0, "y_score": 0.1, "method": "model_a"}
    ])], ignore_index=True)
    joined = load_predictions_with_text(extra, test_corpus)
    assert len(joined) == len(binary_predictions)
    assert joined.attrs["n_dropped"] == 1


def test_load_predictions_validates_schema(test_corpus):
    bad = pd.DataFrame({"foo": [1], "bar": [2]})
    with pytest.raises(ValueError, match="missing required columns"):
        load_predictions_with_text(bad, test_corpus)


def test_load_predictions_accepts_csv_path(binary_predictions, test_corpus, tmp_path):
    csv_path = tmp_path / "preds.csv"
    binary_predictions.to_csv(csv_path, index=False)
    joined = load_predictions_with_text(csv_path, test_corpus)
    assert len(joined) == len(binary_predictions)


# ---------------------------------------------------------------------------
# detect_task
# ---------------------------------------------------------------------------


def test_detect_task_binary(binary_predictions, test_corpus):
    joined = load_predictions_with_text(binary_predictions, test_corpus)
    assert detect_task(joined) == "binary"


def test_detect_task_multiclass(multiclass_predictions, test_corpus):
    joined = load_predictions_with_text(multiclass_predictions, test_corpus)
    assert detect_task(joined) == "multiclass"


# ---------------------------------------------------------------------------
# build_binary_error_pool
# ---------------------------------------------------------------------------


def test_binary_error_pool_filters_correct_predictions(binary_predictions, test_corpus):
    joined = load_predictions_with_text(binary_predictions, test_corpus)
    errors = build_binary_error_pool(joined)
    assert len(errors) == 4  # 2 FP + 1 FN + 1 cotidiano-greve FP
    assert set(errors["error_type"]) == {"FP", "FN"}


def test_binary_error_pool_tags_fp_fn(binary_predictions, test_corpus):
    joined = load_predictions_with_text(binary_predictions, test_corpus)
    errors = build_binary_error_pool(joined)
    fps = errors[errors["error_type"] == "FP"]
    fns = errors[errors["error_type"] == "FN"]
    assert (fps["y_true"] == 0).all() and (fps["y_pred"] == 1).all()
    assert (fns["y_true"] == 1).all() and (fns["y_pred"] == 0).all()


def test_binary_error_pool_rejects_multiclass(multiclass_predictions, test_corpus):
    joined = load_predictions_with_text(multiclass_predictions, test_corpus)
    with pytest.raises(ValueError, match="binary task"):
        build_binary_error_pool(joined)


# ---------------------------------------------------------------------------
# build_multiclass_error_pool
# ---------------------------------------------------------------------------


def test_multiclass_error_pool_directional(multiclass_predictions, test_corpus):
    joined = load_predictions_with_text(multiclass_predictions, test_corpus)
    errors = build_multiclass_error_pool(joined)
    assert len(errors) == 4
    assert "colunas->mercado" in errors["error_type"].values
    assert "colunas->ilustrada" in errors["error_type"].values
    assert "poder->mercado" in errors["error_type"].values
    assert "cotidiano->poder" in errors["error_type"].values


def test_multiclass_error_pool_focus_classes(multiclass_predictions, test_corpus):
    joined = load_predictions_with_text(multiclass_predictions, test_corpus)
    errors = build_multiclass_error_pool(joined, focus_classes={"mercado"})
    # Only errors involving 'mercado' (as true OR pred) survive
    assert len(errors) == 2  # colunas->mercado and poder->mercado
    assert all("mercado" in et for et in errors["error_type"])


def test_multiclass_error_pool_rejects_binary(binary_predictions, test_corpus):
    joined = load_predictions_with_text(binary_predictions, test_corpus)
    with pytest.raises(ValueError, match="multiclass task"):
        build_multiclass_error_pool(joined)


# ---------------------------------------------------------------------------
# build_disagreement_pool
# ---------------------------------------------------------------------------


def test_disagreement_pool_three_methods(test_corpus):
    pred_a = pd.DataFrame([
        {"index": 1001, "y_true": 1, "y_pred": 1, "method": "a"},  # all correct
        {"index": 1002, "y_true": 1, "y_pred": 0, "method": "a"},  # all wrong
        {"index": 1003, "y_true": 0, "y_pred": 1, "method": "a"},  # majority wrong, B right
        {"index": 1004, "y_true": 0, "y_pred": 0, "method": "a"},  # majority right, C wrong
    ])
    pred_b = pd.DataFrame([
        {"index": 1001, "y_true": 1, "y_pred": 1, "method": "b"},
        {"index": 1002, "y_true": 1, "y_pred": 0, "method": "b"},
        {"index": 1003, "y_true": 0, "y_pred": 0, "method": "b"},  # B right
        {"index": 1004, "y_true": 0, "y_pred": 0, "method": "b"},
    ])
    pred_c = pd.DataFrame([
        {"index": 1001, "y_true": 1, "y_pred": 1, "method": "c"},
        {"index": 1002, "y_true": 1, "y_pred": 0, "method": "c"},
        {"index": 1003, "y_true": 0, "y_pred": 1, "method": "c"},
        {"index": 1004, "y_true": 0, "y_pred": 1, "method": "c"},  # C wrong
    ])
    pool = build_disagreement_pool({"a": pred_a, "b": pred_b, "c": pred_c}, test_corpus)

    # 'all_correct' (1001) should be filtered out
    assert 1001 not in pool["index"].values
    # 'all_wrong' (1002) kept
    assert (pool[pool["index"] == 1002]["disagreement_pattern"] == "all_wrong").all()
    # majority_wrong_one_right (1003: only B right)
    assert (pool[pool["index"] == 1003]["disagreement_pattern"] == "majority_wrong_one_right").all()
    # majority_right_one_wrong (1004: only C wrong)
    assert (pool[pool["index"] == 1004]["disagreement_pattern"] == "majority_right_one_wrong").all()


def test_disagreement_pool_requires_two_methods(test_corpus):
    pred = pd.DataFrame([{"index": 1001, "y_true": 1, "y_pred": 1, "method": "a"}])
    with pytest.raises(ValueError, match="at least 2 methods"):
        build_disagreement_pool({"a": pred}, test_corpus)


# ---------------------------------------------------------------------------
# Pre-annotation summaries
# ---------------------------------------------------------------------------


def test_summarize_errors_by_category(binary_predictions, test_corpus):
    joined = load_predictions_with_text(binary_predictions, test_corpus)
    errors = build_binary_error_pool(joined)
    summary = summarize_errors_by_category(errors, column="category")
    # 4 errors: 1 FN in mercado, 1 FP in colunas, 1 FP in poder, 1 FP in cotidiano
    assert summary["n_errors"].sum() == 4
    assert set(summary["category"]) == {"mercado", "colunas", "poder", "cotidiano"}
    assert (summary["share"].sum() - 1.0) < 1e-6


def test_summarize_errors_by_confidence_bins(binary_predictions, test_corpus):
    joined = load_predictions_with_text(binary_predictions, test_corpus)
    errors = build_binary_error_pool(joined)
    summary = summarize_errors_by_confidence(errors, n_bins=5)
    assert len(summary) == 5
    assert summary["n_errors"].sum() == 4
    # The high-confidence FP (poder, score=0.88) lands in the top bin
    assert summary.iloc[-1]["n_errors"] >= 1


def test_summarize_errors_by_confidence_no_score_column():
    errors = pd.DataFrame({"index": [1], "y_true": [0], "y_pred": [1], "method": ["m"]})
    summary = summarize_errors_by_confidence(errors)
    assert summary.empty
    assert list(summary.columns) == ["bin_left", "bin_right", "n_errors", "share"]


def test_summarize_errors_by_text_length(binary_predictions, test_corpus):
    joined = load_predictions_with_text(binary_predictions, test_corpus)
    errors = build_binary_error_pool(joined)
    summary = summarize_errors_by_text_length(errors, n_bins=3)
    assert summary["n_errors"].sum() == 4


def test_summarize_errors_by_date(binary_predictions, test_corpus):
    joined = load_predictions_with_text(binary_predictions, test_corpus)
    errors = build_binary_error_pool(joined)
    summary = summarize_errors_by_date(errors, freq="YS")
    assert summary["n_errors"].sum() == 4
    assert "period" in summary.columns


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def test_stratified_sample_balances_strata(binary_predictions, test_corpus):
    joined = load_predictions_with_text(binary_predictions, test_corpus)
    errors = build_binary_error_pool(joined)
    sample = stratified_error_sample(errors, n_per_stratum=2, stratify_by="error_type")
    counts = sample["error_type"].value_counts()
    # 3 FPs available, 1 FN available; n_per_stratum=2 means 2 FP + 1 FN
    assert counts.get("FP", 0) == 2
    assert counts.get("FN", 0) == 1


def test_stratified_sample_deterministic(binary_predictions, test_corpus):
    joined = load_predictions_with_text(binary_predictions, test_corpus)
    errors = build_binary_error_pool(joined)
    a = stratified_error_sample(errors, n_per_stratum=2, seed=DEFAULT_SEED)
    b = stratified_error_sample(errors, n_per_stratum=2, seed=DEFAULT_SEED)
    pd.testing.assert_frame_equal(a, b)


def test_stratified_sample_caps_at_stratum_size(binary_predictions, test_corpus):
    joined = load_predictions_with_text(binary_predictions, test_corpus)
    errors = build_binary_error_pool(joined)
    sample = stratified_error_sample(errors, n_per_stratum=100)
    # All 4 errors are returned even though n_per_stratum > stratum size
    assert len(sample) == 4


def test_stratified_sample_validates_inputs(binary_predictions, test_corpus):
    joined = load_predictions_with_text(binary_predictions, test_corpus)
    errors = build_binary_error_pool(joined)
    with pytest.raises(KeyError):
        stratified_error_sample(errors, n_per_stratum=2, stratify_by="nope")
    with pytest.raises(ValueError):
        stratified_error_sample(errors, n_per_stratum=0)


# ---------------------------------------------------------------------------
# Annotation template export & load
# ---------------------------------------------------------------------------


def test_export_annotation_template_columns(binary_predictions, test_corpus, tmp_path):
    joined = load_predictions_with_text(binary_predictions, test_corpus)
    errors = build_binary_error_pool(joined)
    out = export_annotation_template(errors, tmp_path / "annot.csv")
    df = pd.read_csv(out)
    for col in ("index", "method", "error_type", "title", "text_preview", "text_full_chars",
                "category", "y_true", "y_pred"):
        assert col in df.columns
    for col in ANNOTATION_COLUMNS:
        assert col in df.columns
        # Empty annotation columns serialize as NaN through CSV roundtrip
        assert df[col].isna().all() or (df[col].fillna("") == "").all()


def test_export_annotation_template_truncates_long_text(test_corpus):
    long_pred = pd.DataFrame([
        {"index": 1001, "y_true": 1, "y_pred": 0, "y_score": 0.1, "method": "m"},
    ])
    joined = load_predictions_with_text(long_pred, test_corpus)
    errors = build_binary_error_pool(joined)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        out = export_annotation_template(errors, f.name, max_text_chars=50)
        df = pd.read_csv(out)
    # Text in fixture is "Banco Central elevou a taxa Selic " * 30 = 1020 chars
    assert df.iloc[0]["text_full_chars"] > 50
    assert len(df.iloc[0]["text_preview"]) <= 51 + 1  # 50 + ellipsis


def test_load_annotated_sample_validates_vocabulary(tmp_path):
    df = pd.DataFrame({
        "index": [1, 2],
        "error_type": ["FP", "FN"],
        "subtema_real": ["", ""],
        "tipo_erro_anotado": ["modelo_erra", "INVALID_TYPE"],
        "editorialmente_economia": ["", ""],
        "signal_palavras": ["", ""],
        "notas": ["", ""],
    })
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="outside ANNOTATION_TYPES"):
        load_annotated_sample(path)


def test_load_annotated_sample_allows_empty_unless_required(tmp_path):
    df = pd.DataFrame({
        "index": [1, 2],
        "error_type": ["FP", "FN"],
        "subtema_real": ["", ""],
        "tipo_erro_anotado": ["modelo_erra", ""],
        "editorialmente_economia": ["", ""],
        "signal_palavras": ["", ""],
        "notas": ["", ""],
    })
    path = tmp_path / "partial.csv"
    df.to_csv(path, index=False)
    out = load_annotated_sample(path)  # empty allowed by default
    assert len(out) == 2
    with pytest.raises(ValueError, match="empty tipo_erro_anotado"):
        load_annotated_sample(path, require_complete=True)


# ---------------------------------------------------------------------------
# summarize_annotations
# ---------------------------------------------------------------------------


def test_summarize_annotations_headline_numbers():
    df = pd.DataFrame({
        "error_type": ["FP", "FP", "FP", "FN", "FN"],
        "tipo_erro_anotado": [
            "rotulagem_editorial",
            "modelo_erra",
            "modelo_erra",
            "tema_misto",
            "ambiguo",
        ],
    })
    summary = summarize_annotations(df)
    assert summary["n_total"] == 5
    assert summary["n_annotated"] == 5
    # 3 of 5 are NOT 'modelo_erra' → adjusted_correctness = 0.6
    assert summary["adjusted_correctness"] == 0.6
    assert summary["by_type"]["modelo_erra"]["n"] == 2
    assert summary["by_type"]["modelo_erra"]["share"] == 0.4
    # FP split: 1 rotulagem + 2 modelo_erra
    assert summary["by_error_type"]["FP"]["modelo_erra"]["n"] == 2


def test_summarize_annotations_handles_empty():
    df = pd.DataFrame({
        "error_type": ["FP", "FN"],
        "tipo_erro_anotado": ["", ""],
    })
    summary = summarize_annotations(df)
    assert summary["n_annotated"] == 0
    assert summary["adjusted_correctness"] == 0.0


def test_annotation_constants_match_summary_logic():
    # Make sure the four annotation types are exactly the ones the summary
    # divides into 'model fault' and 'not model fault'.
    not_model = {"rotulagem_editorial", "tema_misto", "ambiguo"}
    model = {"modelo_erra"}
    assert set(ANNOTATION_TYPES) == not_model | model
