"""Tests for economy_classifier.heuristics — scoring, bands, penalties."""

from economy_classifier.heuristics import (
    MARKET_SCORE_THRESHOLD,
    SIGNAL_WEIGHTS,
    build_default_catalog,
    classify_score,
    score_text,
)


def _catalog():
    return build_default_catalog()


def test_known_terms_score_above_threshold():
    result = score_text(
        "Banco Central elevou a taxa Selic e a inflação subiu no país",
        _catalog(),
    )
    assert result["classification"] == "mercado"
    assert result["score"] >= MARKET_SCORE_THRESHOLD


def test_non_financial_text_below_threshold():
    result = score_text(
        "O time venceu o campeonato regional após uma temporada invicta",
        _catalog(),
    )
    assert result["classification"] == "outros"
    assert result["score"] < 1.0


def test_cambio_automotive_penalized():
    result = score_text(
        "A oficina trocou o câmbio do carro e a embreagem do motor",
        _catalog(),
    )
    penalties = [p["rule"] for p in result["penalty_reasons"]]
    assert "cambio_contexto_automotivo" in penalties


def test_token_technical_penalized():
    result = score_text(
        "A API exige token JWT para autenticação no endpoint",
        _catalog(),
    )
    penalties = [p["rule"] for p in result["penalty_reasons"]]
    assert "token_contexto_tecnico" in penalties


def test_political_density_penalized():
    result = score_text(
        "Os candidatos à presidência debatem eleições e corrupção no congresso",
        _catalog(),
    )
    penalties = [p["rule"] for p in result["penalty_reasons"]]
    assert "densidade_politico_social" in penalties


def test_economia_non_financial_penalized():
    result = score_text(
        "O celular entrou em modo economia de bateria para poupar energia",
        _catalog(),
    )
    penalties = [p["rule"] for p in result["penalty_reasons"]]
    assert "economia_contexto_nao_financeiro" in penalties


def test_tech_rh_context_penalized():
    result = score_text(
        "Vaga de emprego home office para desenvolvedor backend com autenticação JWT e token de API",
        _catalog(),
    )
    penalties = [p["rule"] for p in result["penalty_reasons"]]
    assert "contexto_tecnico_ou_rh" in penalties


def test_score_text_returns_expected_keys():
    result = score_text("texto qualquer", _catalog())
    expected_keys = {
        "score", "raw_score", "adjusted_score", "classification",
        "terms_found", "themes_found", "signal_counts",
        "word_count", "penalty_reasons",
    }
    assert expected_keys.issubset(result.keys())


def test_classify_score_strict_mode():
    assert classify_score(3.0, has_strong_signal=True) == "mercado"
    assert classify_score(3.0, has_strong_signal=False) == "ambiguo"
    assert classify_score(1.5, has_strong_signal=False) == "ambiguo"
    assert classify_score(0.5, has_strong_signal=False) == "outros"


def test_classify_score_thresholds():
    assert classify_score(2.6, has_strong_signal=True) == "mercado"
    assert classify_score(2.59, has_strong_signal=True) == "ambiguo"
    assert classify_score(1.0, has_strong_signal=False) == "ambiguo"
    assert classify_score(0.99, has_strong_signal=False) == "outros"


def test_log1p_normalization():
    catalog = _catalog()
    short = score_text("inflação e juros", catalog)
    long = score_text(
        "inflação e juros " + "texto irrelevante adicional " * 50,
        catalog,
    )
    assert long["score"] <= short["score"]


def test_catalog_term_count():
    catalog = _catalog()
    assert len(catalog) >= 190


def test_signal_weights_consistent():
    assert SIGNAL_WEIGHTS["nuclear"] == 4
    assert SIGNAL_WEIGHTS["setorial"] == 3
    assert SIGNAL_WEIGHTS["contextual"] == 2
    assert SIGNAL_WEIGHTS["fraco"] == 1
