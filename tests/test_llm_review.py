"""Tests for economy_classifier.llm_review — LLM review of ensemble predictions."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from economy_classifier.llm_review import (
    build_review_prompt,
    classify_single,
    compute_review_concordance,
    parse_llm_response,
)


# --- build_review_prompt ---


def test_build_review_prompt_structure():
    messages = build_review_prompt("Bolsa sobe 2%")
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "Bolsa sobe 2%"


def test_build_review_prompt_preserves_text():
    text = "Texto com acentuação e números 123"
    messages = build_review_prompt(text)
    assert messages[-1]["content"] == text


# --- parse_llm_response ---


def test_parse_llm_response_pure_json():
    raw = '{"label": "mercado", "justificativa": "Tema de mercado financeiro."}'
    result = parse_llm_response(raw)
    assert result["label"] == "mercado"
    assert "financeiro" in result["justificativa"]


def test_parse_llm_response_markdown_fenced():
    raw = '```json\n{"label": "outros", "justificativa": "Politica publica."}\n```'
    result = parse_llm_response(raw)
    assert result["label"] == "outros"


def test_parse_llm_response_with_surrounding_text():
    raw = 'Aqui está a classificação: {"label": "mercado", "justificativa": "Bolsa."} Fim.'
    result = parse_llm_response(raw)
    assert result["label"] == "mercado"


def test_parse_llm_response_invalid():
    result = parse_llm_response("Isso nao e JSON nenhum")
    assert result["label"] == "erro"


def test_parse_llm_response_wrong_label():
    raw = '{"label": "economia", "justificativa": "Tema economico."}'
    result = parse_llm_response(raw)
    assert result["label"] == "erro"


def test_parse_llm_response_missing_justificativa():
    raw = '{"label": "mercado"}'
    result = parse_llm_response(raw)
    assert result["label"] == "erro"


# --- classify_single ---


def test_classify_single_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(
        {"label": "mercado", "justificativa": "Bolsa de valores."}
    )
    mock_client.chat.completions.create.return_value = mock_response

    result = classify_single(mock_client, "Bolsa sobe 2%")
    assert result["label"] == "mercado"
    assert "raw_response" in result

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "sabia-4"
    assert call_kwargs["temperature"] == 0


def test_classify_single_api_failure():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API down")

    result = classify_single(
        mock_client, "texto",
        max_retries=2, retry_delay=0.01,
    )
    assert result["label"] == "erro"
    assert mock_client.chat.completions.create.call_count == 2


def test_classify_single_unparseable_response():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Nao sei classificar"
    mock_client.chat.completions.create.return_value = mock_response

    result = classify_single(mock_client, "texto")
    assert result["label"] == "erro"
    assert result["raw_response"] == "Nao sei classificar"


# --- compute_review_concordance ---


def test_compute_review_concordance_perfect():
    s2 = pd.Series(["mercado", "outros", "mercado", "outros"])
    sabia = pd.Series(["mercado", "outros", "mercado", "outros"])
    result = compute_review_concordance(s2, sabia)
    assert result["total"] == 4
    assert result["concordant"] == 4
    assert result["discordant"] == 0
    assert result["concordance_rate"] == 1.0
    assert result["cohen_kappa"] == 1.0


def test_compute_review_concordance_with_disagreements():
    s2 = pd.Series(["mercado", "outros", "outros", "mercado"])
    sabia = pd.Series(["mercado", "mercado", "outros", "outros"])
    result = compute_review_concordance(s2, sabia)
    assert result["total"] == 4
    assert result["concordant"] == 2
    assert result["discordant"] == 2
    assert result["outros_to_mercado"] == 1
    assert result["mercado_to_outros"] == 1


def test_compute_review_concordance_ignores_errors():
    s2 = pd.Series(["mercado", "outros", "mercado"])
    sabia = pd.Series(["mercado", "erro", "outros"])
    result = compute_review_concordance(s2, sabia)
    assert result["total"] == 2
    assert result["erros_parsing"] == 1
    assert result["concordant"] == 1
    assert result["mercado_to_outros"] == 1
