"""Tests for economy_classifier.llm_review — LLM review of ensemble predictions."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from economy_classifier.llm_review import (
    LLM_REGISTRY,
    VALID_BINARY_LABELS,
    VALID_MULTI_LABELS,
    build_few_shot_examples,
    build_review_prompt,
    build_review_prompt_few_shot,
    build_review_prompt_multiclass,
    classify_batch_hf,
    classify_single,
    classify_single_hf,
    compute_review_concordance,
    hf_results_to_multiclass_predictions,
    hf_results_to_predictions,
    parse_llm_response,
    parse_llm_response_multiclass,
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


def test_parse_llm_response_bare_label():
    """Primary format: prompts now ask for a single bare word."""
    assert parse_llm_response("mercado")["label"] == "mercado"
    assert parse_llm_response("outros")["label"] == "outros"
    assert parse_llm_response("MERCADO")["label"] == "mercado"  # case-insensitive


def test_parse_llm_response_bare_label_with_punctuation():
    assert parse_llm_response('"mercado"')["label"] == "mercado"
    assert parse_llm_response("mercado.")["label"] == "mercado"
    assert parse_llm_response(" mercado ")["label"] == "mercado"
    assert parse_llm_response("'outros'")["label"] == "outros"


def test_parse_llm_response_pure_json_still_supported():
    """Backward compat: JSON-wrapped responses still parse (for legacy Sabia outputs)."""
    raw = '{"label": "mercado"}'
    assert parse_llm_response(raw)["label"] == "mercado"


def test_parse_llm_response_json_without_justificativa_now_valid():
    """Justificativa is no longer required in the response."""
    raw = '{"label": "mercado"}'
    assert parse_llm_response(raw)["label"] == "mercado"


def test_parse_llm_response_json_with_extra_fields():
    raw = '{"label": "outros", "confidence": 0.92, "explanation": "x"}'
    assert parse_llm_response(raw)["label"] == "outros"


def test_parse_llm_response_markdown_fenced():
    raw = '```json\n{"label": "outros"}\n```'
    assert parse_llm_response(raw)["label"] == "outros"


def test_parse_llm_response_label_in_sentence():
    """Last-resort: label found as a whole word inside a sentence."""
    assert parse_llm_response("A categoria correta e mercado.")["label"] == "mercado"


def test_parse_llm_response_invalid():
    assert parse_llm_response("Isso nao e nada nem mesmo um label")["label"] == "erro"
    assert parse_llm_response("")["label"] == "erro"


def test_parse_llm_response_wrong_label():
    raw = "economia"
    assert parse_llm_response(raw)["label"] == "erro"
    raw_json = '{"label": "economia"}'
    assert parse_llm_response(raw_json)["label"] == "erro"


# --- classify_single ---


def test_classify_single_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    # New format: bare label
    mock_response.choices[0].message.content = "mercado"
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


# --- HF backend ---


def test_llm_registry_contains_recommended_models():
    assert "qwen2.5-7b-instruct" in LLM_REGISTRY
    assert "llama-3.1-8b-instruct" in LLM_REGISTRY
    assert LLM_REGISTRY["qwen2.5-7b-instruct"].startswith("Qwen/")
    assert LLM_REGISTRY["llama-3.1-8b-instruct"].startswith("meta-llama/")


def _fake_tokenizer_and_model(generated_strings: list[str]):
    """Mocks that simulate the tokenizer.apply_chat_template + model.generate flow.

    `generated_strings` is a queue of decoded outputs across ALL batches;
    the mock pops from it sequentially as ``tokenizer.decode`` is called.
    The mock tracks per-call batch size so ``model.generate`` returns the
    right number of outputs per batch.
    """
    last_batch_size = {"n": 0}

    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.padding_side = "right"
    tokenizer.apply_chat_template = MagicMock(
        side_effect=lambda msgs, **kw: f"PROMPT::{msgs[-1]['content']}",
    )

    def fake_tokenize(prompts, **kw):
        last_batch_size["n"] = len(prompts)
        encoded = MagicMock()
        encoded.input_ids = MagicMock()
        encoded.input_ids.shape = (len(prompts), 5)  # batch x seq_len=5
        encoded.to = MagicMock(return_value=encoded)
        # __iter__ over the encoded mapping needed by `**inputs` unpacking
        encoded.keys = MagicMock(return_value=["input_ids", "attention_mask"])
        encoded.__iter__ = MagicMock(return_value=iter(["input_ids", "attention_mask"]))
        encoded.__getitem__ = MagicMock(side_effect=lambda k: MagicMock())
        return encoded
    tokenizer.side_effect = fake_tokenize

    decode_iter = iter(generated_strings)
    tokenizer.decode = MagicMock(side_effect=lambda ids, **kw: next(decode_iter))

    model = MagicMock()
    model.device = "cpu"

    def fake_generate(**kwargs):
        # Return an iterable of length matching the most recent tokenize() call,
        # so _hf_generate's `for output_ids in outputs` loop yields exactly the
        # right number of items.
        return [MagicMock() for _ in range(last_batch_size["n"])]
    model.generate = MagicMock(side_effect=fake_generate)

    return tokenizer, model


def test_classify_single_hf_parses_valid_response():
    tokenizer, model = _fake_tokenizer_and_model([
        '{"label": "mercado", "justificativa": "Bolsa subiu."}',
    ])
    result = classify_single_hf(tokenizer, model, "Texto sobre bolsa")
    assert result["label"] == "mercado"
    assert "raw_response" in result
    tokenizer.apply_chat_template.assert_called_once()
    model.generate.assert_called_once()


def test_classify_single_hf_handles_unparseable_response():
    tokenizer, model = _fake_tokenizer_and_model(["nao sei classificar"])
    result = classify_single_hf(tokenizer, model, "texto")
    assert result["label"] == "erro"


def test_classify_batch_hf_propagates_method_and_record_ids():
    tokenizer, model = _fake_tokenizer_and_model([
        '{"label": "mercado", "justificativa": "X"}',
        '{"label": "outros", "justificativa": "Y"}',
    ])
    results = classify_batch_hf(
        tokenizer, model,
        texts=["t1", "t2"],
        record_ids=[101, 202],
        method="qwen2.5-7b-instruct",
        batch_size=2,
    )
    assert len(results) == 2
    assert results[0]["record_id"] == 101
    assert results[0]["method"] == "qwen2.5-7b-instruct"
    assert results[0]["label"] == "mercado"
    assert results[1]["record_id"] == 202
    assert results[1]["label"] == "outros"


def test_classify_batch_hf_validates_length_mismatch():
    tokenizer, model = _fake_tokenizer_and_model([])
    with pytest.raises(ValueError, match="length mismatch"):
        classify_batch_hf(
            tokenizer, model,
            texts=["a", "b"],
            record_ids=[1],
            method="qwen2.5-7b-instruct",
        )


def test_classify_batch_hf_writes_checkpoint(tmp_path):
    tokenizer, model = _fake_tokenizer_and_model([
        '{"label": "mercado", "justificativa": "x"}',
        '{"label": "outros", "justificativa": "y"}',
        '{"label": "mercado", "justificativa": "z"}',
    ])
    cp = tmp_path / "checkpoint.csv"
    classify_batch_hf(
        tokenizer, model,
        texts=["a", "b", "c"], record_ids=[1, 2, 3],
        method="qwen2.5-7b-instruct",
        batch_size=1,
        checkpoint_path=cp,
        checkpoint_every=1,
    )
    assert cp.exists()
    saved = pd.read_csv(cp)
    assert len(saved) == 3
    assert set(saved["record_id"]) == {1, 2, 3}


def test_classify_batch_hf_recovers_from_batch_error(monkeypatch):
    """If model.generate raises, the batch is recorded as errors and the loop continues."""
    tokenizer, model = _fake_tokenizer_and_model([
        '{"label": "mercado", "justificativa": "X"}',
    ])
    call_count = {"n": 0}
    original_generate = model.generate

    def flaky_generate(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated CUDA OOM")
        return original_generate(**kwargs)
    model.generate = MagicMock(side_effect=flaky_generate)

    results = classify_batch_hf(
        tokenizer, model,
        texts=["bad", "good"],
        record_ids=[1, 2],
        method="qwen2.5-7b-instruct",
        batch_size=1,
    )
    assert len(results) == 2
    assert results[0]["label"] == "erro"
    assert results[1]["label"] == "mercado"


def test_hf_results_to_predictions_drops_errors_and_maps_labels():
    results = [
        {"label": "mercado", "justificativa": "A", "record_id": 1, "method": "qwen2.5-7b-instruct"},
        {"label": "outros",  "justificativa": "B", "record_id": 2, "method": "qwen2.5-7b-instruct"},
        {"label": "erro",    "justificativa": "C", "record_id": 3, "method": "qwen2.5-7b-instruct"},
    ]
    preds = hf_results_to_predictions(results)
    assert len(preds) == 2  # error row dropped
    assert list(preds.columns) >= ["index", "y_pred", "y_score", "method", "label"]
    row_mercado = preds.iloc[0]
    assert row_mercado["y_pred"] == 1
    assert row_mercado["y_score"] == 1.0
    row_outros = preds.iloc[1]
    assert row_outros["y_pred"] == 0
    assert row_outros["y_score"] == 0.0


def test_hf_results_to_predictions_empty_input():
    preds = hf_results_to_predictions([])
    assert len(preds) == 0


# --- Multiclass: prompt, parser, predictions ---


def test_valid_multi_labels_has_8_classes():
    assert len(VALID_MULTI_LABELS) == 8
    assert set(VALID_MULTI_LABELS) == {
        "poder", "colunas", "mercado", "esporte",
        "mundo", "cotidiano", "ilustrada", "outros",
    }


def test_valid_binary_labels_unchanged():
    assert VALID_BINARY_LABELS == ("mercado", "outros")


def test_build_review_prompt_multiclass_structure():
    messages = build_review_prompt_multiclass("Lula assina decreto sobre Petrobras")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "8 categorias" in messages[0]["content"]
    assert messages[1]["content"] == "Lula assina decreto sobre Petrobras"


def test_parse_llm_response_multiclass_accepts_top7_labels_bare():
    """Multiclass parser accepts bare labels (new format)."""
    for label in VALID_MULTI_LABELS:
        result = parse_llm_response_multiclass(label)
        assert result["label"] == label, f"failed for bare label {label!r}"


def test_parse_llm_response_multiclass_accepts_top7_labels_json():
    """Backward compat: also accepts JSON-wrapped (no justificativa needed)."""
    for label in VALID_MULTI_LABELS:
        raw = f'{{"label": "{label}"}}'
        result = parse_llm_response_multiclass(raw)
        assert result["label"] == label, f"failed for JSON label {label!r}"


def test_parse_llm_response_multiclass_rejects_binary_only_labels():
    # "esportes" (with trailing s) should be rejected — not in VALID_MULTI_LABELS exactly.
    raw = "esportes"
    result = parse_llm_response_multiclass(raw)
    assert result["label"] == "erro"


def test_parse_llm_response_multiclass_rejects_arbitrary_labels():
    raw = "tecnologia"
    result = parse_llm_response_multiclass(raw)
    assert result["label"] == "erro"


def test_parse_llm_response_with_explicit_valid_labels():
    """The base parse_llm_response respects a custom valid_labels argument."""
    raw = "esporte"
    binary = parse_llm_response(raw)  # default: binary labels
    multi = parse_llm_response(raw, valid_labels=VALID_MULTI_LABELS)
    assert binary["label"] == "erro"
    assert multi["label"] == "esporte"


def test_classify_single_hf_with_multiclass_parser():
    tokenizer, model = _fake_tokenizer_and_model([
        '{"label": "esporte", "justificativa": "Futebol Cup."}',
    ])
    result = classify_single_hf(
        tokenizer, model, "Brasil bate Argentina por 3x1",
        prompt_builder=build_review_prompt_multiclass,
        parser=parse_llm_response_multiclass,
    )
    assert result["label"] == "esporte"


def test_classify_batch_hf_with_multiclass_routes_through_custom_parser():
    tokenizer, model = _fake_tokenizer_and_model([
        '{"label": "poder", "justificativa": "X"}',
        '{"label": "ilustrada", "justificativa": "Y"}',
    ])
    results = classify_batch_hf(
        tokenizer, model,
        texts=["Eleicao", "Filme"],
        record_ids=[1, 2],
        method="qwen2.5-7b-instruct",
        batch_size=2,
        prompt_builder=build_review_prompt_multiclass,
        parser=parse_llm_response_multiclass,
    )
    assert len(results) == 2
    assert results[0]["label"] == "poder"
    assert results[1]["label"] == "ilustrada"


def test_hf_results_to_multiclass_predictions_drops_invalid():
    results = [
        {"label": "mercado", "justificativa": "A", "record_id": 1, "method": "qwen"},
        {"label": "esporte", "justificativa": "B", "record_id": 2, "method": "qwen"},
        {"label": "erro", "justificativa": "C", "record_id": 3, "method": "qwen"},
        {"label": "tecnologia", "justificativa": "D", "record_id": 4, "method": "qwen"},
    ]
    preds = hf_results_to_multiclass_predictions(results)
    assert len(preds) == 2
    assert list(preds["y_pred"]) == ["mercado", "esporte"]
    assert "y_score" not in preds.columns  # multiclass has no probabilistic score


def test_hf_results_to_multiclass_predictions_empty():
    preds = hf_results_to_multiclass_predictions([])
    assert len(preds) == 0


# --- Few-shot ---


def test_build_few_shot_examples_binary_balanced():
    df = pd.DataFrame({
        "text": [f"texto {i}" for i in range(20)],
        "label": [1] * 10 + [0] * 10,
    })
    exs = build_few_shot_examples(df, n_per_class=2, seed=0)
    # 2 per class x 2 classes = 4 examples
    assert len(exs) == 4
    labels = [lbl for _, lbl in exs]
    assert labels.count("mercado") == 2
    assert labels.count("outros") == 2


def test_build_few_shot_examples_multiclass_one_per_class():
    df = pd.DataFrame({
        "text": [f"texto {i}" for i in range(40)],
        "label_multi": ["poder"] * 5 + ["colunas"] * 5 + ["mercado"] * 5
                       + ["esporte"] * 5 + ["mundo"] * 5 + ["cotidiano"] * 5
                       + ["ilustrada"] * 5 + ["outros"] * 5,
    })
    exs = build_few_shot_examples(
        df,
        label_column="label_multi",
        valid_labels=VALID_MULTI_LABELS,
        n_per_class=1,
        seed=0,
    )
    assert len(exs) == 8
    labels = [lbl for _, lbl in exs]
    assert set(labels) == set(VALID_MULTI_LABELS)


def test_build_few_shot_examples_truncates_long_texts():
    long_text = "a" * 5000
    df = pd.DataFrame({"text": [long_text] * 10, "label": [0, 1] * 5})
    exs = build_few_shot_examples(df, n_per_class=1, text_max_chars=200, seed=0)
    for text, _ in exs:
        assert len(text) <= 200


def test_build_few_shot_examples_deterministic_with_seed():
    df = pd.DataFrame({
        "text": [f"texto {i}" for i in range(20)],
        "label": [1] * 10 + [0] * 10,
    })
    exs1 = build_few_shot_examples(df, n_per_class=2, seed=42)
    exs2 = build_few_shot_examples(df, n_per_class=2, seed=42)
    assert exs1 == exs2


def test_build_few_shot_examples_skips_missing_class():
    """If a class has zero rows, just skip it (don't fail)."""
    df = pd.DataFrame({
        "text": ["t1", "t2", "t3"],
        "label_multi": ["poder", "poder", "esporte"],
    })
    exs = build_few_shot_examples(
        df, label_column="label_multi", valid_labels=VALID_MULTI_LABELS,
        n_per_class=1, seed=0,
    )
    # Only "poder" and "esporte" exist; expect 2 examples
    assert len(exs) == 2
    assert {lbl for _, lbl in exs} == {"poder", "esporte"}


def test_build_review_prompt_few_shot_binary_message_structure():
    examples = [("texto sobre bolsa", "mercado"), ("texto sobre futebol", "outros")]
    msgs = build_review_prompt_few_shot("texto novo", examples=examples)
    # system + 2*(user, assistant) pairs + final user = 6 messages
    assert len(msgs) == 6
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "texto sobre bolsa"
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] == "mercado"
    assert msgs[3]["role"] == "user"
    assert msgs[4]["role"] == "assistant"
    assert msgs[5]["role"] == "user"
    assert msgs[5]["content"] == "texto novo"


def test_build_review_prompt_few_shot_multiclass_uses_multi_system():
    examples = [("texto X", "esporte")]
    msgs_bin = build_review_prompt_few_shot("q", examples=examples, multiclass=False)
    msgs_multi = build_review_prompt_few_shot("q", examples=examples, multiclass=True)
    assert "binario" in msgs_bin[0]["content"].lower()
    assert "8 categorias" in msgs_multi[0]["content"]


def test_classify_single_hf_with_few_shot_prompt():
    """End-to-end: pass a few-shot prompt builder via prompt_builder param."""
    from functools import partial
    examples = [("texto bolsa", "mercado"), ("texto gol", "outros")]
    builder = partial(build_review_prompt_few_shot, examples=examples)

    tokenizer, model = _fake_tokenizer_and_model(["mercado"])
    result = classify_single_hf(
        tokenizer, model, "texto novo sobre Selic",
        prompt_builder=builder,
    )
    assert result["label"] == "mercado"
    # Verify the chat template received a multi-message conversation (5 entries)
    call = tokenizer.apply_chat_template.call_args_list[0]
    msgs_arg = call[0][0]
    assert len(msgs_arg) == 6  # system + 2*(u,a) + final user
