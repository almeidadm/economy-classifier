"""LLM-based classification of news articles (mercado vs outros).

Two backends:

- **API** (OpenAI-compatible client like Maritaca AI Sabia): ``classify_single``,
  ``classify_batch``. Legacy — kept for reproducibility of earlier runs.
- **Local HuggingFace** (Qwen2.5, Llama-3.1, etc.): ``load_hf_model``,
  ``classify_single_hf``, ``classify_batch_hf``. Primary backend going forward
  (no API costs, fully reproducible). Designed for Colab L4 (24 GB) in FP16.

Both backends share the same ``build_review_prompt`` and ``parse_llm_response``,
so the prompt and the parsing of ``{"label": ..., "justificativa": ...}`` are
identical across backends.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.metrics import cohen_kappa_score

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Voce e um classificador binario de textos jornalisticos brasileiros.

Sua tarefa: dado um trecho de texto, classifique-o segundo a sua tematica \
central como "mercado" (texto de tematica economica em sentido amplo) ou \
"outros" (qualquer outra tematica).

- "mercado": o tema central do texto e economia em sentido amplo, abrangendo \
mercado financeiro (bolsa, cambio, juros, bancos, commodities), atividade \
empresarial (resultados, fusoes, aquisicoes, IPOs, governanca corporativa), \
indicadores macroeconomicos (PIB, inflacao, Selic, desemprego, balanca \
comercial), politica economica e fiscal (impostos, reforma tributaria, \
gastos publicos, divida), politicas trabalhistas e sociais com dimensao \
economica (salario minimo, previdencia, programas de transferencia de \
renda, legislacao trabalhista, greves), comercio, industria e agronegocio.
- "outros": o tema central nao e economico — politica partidaria sem \
recorte economico, esportes, cultura, policia, ciencia, saude, educacao, \
internacional sem foco economico, cotidiano.

Em caso de duvida, considere o tema dominante do texto, nao mencoes \
incidentais.

Responda APENAS com uma unica palavra: "mercado" ou "outros". \
Sem JSON, sem aspas, sem explicacao — apenas a palavra."""


SYSTEM_PROMPT_MULTICLASS = """\
Voce e um classificador multiclasse de textos jornalisticos brasileiros \
da Folha de Sao Paulo.

Sua tarefa: dado um trecho de texto, classifique-o em UMA das 8 categorias \
editoriais abaixo, escolhendo a que melhor descreve o tema CENTRAL do texto.

- "poder": politica partidaria, governo federal/estadual/municipal, congresso, \
eleicoes, judiciario, sem foco economico dominante.
- "colunas": opiniao assinada, colunas de opiniao, editoriais (qualquer tema, \
formato opinativo).
- "mercado": economia em sentido amplo — mercado financeiro (bolsa, cambio, \
juros, bancos, commodities), atividade empresarial (resultados, fusoes, IPOs), \
indicadores macroeconomicos (PIB, inflacao, Selic, desemprego), politica \
economica e fiscal, agronegocio, comercio.
- "esporte": futebol (todas as competicoes), olimpiadas, automobilismo, \
qualquer modalidade esportiva.
- "mundo": noticias internacionais sem foco economico dominante (geopolitica, \
conflitos, eleicoes em outros paises).
- "cotidiano": vida urbana, transito, clima, eventos locais, policia, \
acidentes, infraestrutura, mobilidade.
- "ilustrada": cultura, cinema, TV, musica, artes plasticas, literatura, \
celebridades, entretenimento.
- "outros": qualquer tema que NAO se encaixe nas 7 categorias acima (ex: \
ciencia, saude publica, educacao, tecnologia, religiao, meio ambiente sem \
recorte de cotidiano).

Em caso de duvida, escolha a categoria que melhor representa o tema dominante \
— nao mencoes incidentais.

Responda APENAS com uma unica palavra: uma das 8 categorias acima. \
Sem JSON, sem aspas, sem explicacao — apenas a palavra."""


VALID_BINARY_LABELS: tuple[str, ...] = ("mercado", "outros")
VALID_MULTI_LABELS: tuple[str, ...] = (
    "poder", "colunas", "mercado", "esporte", "mundo",
    "cotidiano", "ilustrada", "outros",
)


def build_review_prompt(text: str) -> list[dict]:
    """Build the binary-classification messages array (zero-shot, mercado vs outros)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def build_review_prompt_multiclass(text: str) -> list[dict]:
    """Build the multiclass messages array (zero-shot, top-7 + outros)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_MULTICLASS},
        {"role": "user", "content": text},
    ]


def parse_llm_response(raw: str, valid_labels: tuple[str, ...] = VALID_BINARY_LABELS) -> dict:
    """Extract a class label from a raw LLM response.

    Tries multiple formats in order (most-likely-first given our prompts ask for
    a bare label):

    1. Bare label as the entire response (after stripping quotes/punctuation).
    2. JSON-wrapped (``{"label": "..."}``), with or without markdown fence —
       still parsed for backward compatibility with older Sabia responses.
    3. Any valid label found as a whole word anywhere in the response.

    ``valid_labels`` defaults to the binary labels; pass ``VALID_MULTI_LABELS``
    for the 8-class task.

    Returns ``{"label": str}`` (lowercase, in ``valid_labels``) on success or
    ``{"label": "erro"}`` if no valid label could be extracted.
    """
    if not raw or not raw.strip():
        return {"label": "erro"}

    text = raw.strip()
    text_lower = text.lower()
    valid_set = set(l.lower() for l in valid_labels)

    # 1. Fast path: bare label (with or without surrounding punctuation/quotes)
    cleaned = re.sub(r"^[\s\"'.\(\[{`]+|[\s\"'.,;:!?\)\]}`]+$", "", text_lower)
    first_word = cleaned.split()[0] if cleaned.split() else cleaned
    first_word = re.sub(r"[^\w]", "", first_word)
    if first_word in valid_set:
        return {"label": first_word}

    # 2. JSON parsing (direct, markdown-fenced, or first {...} block)
    candidates = [text]
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        candidates.insert(0, fence_match.group(1))
    brace_match = re.search(r"\{[^{}]*\}", text)
    if brace_match:
        candidates.append(brace_match.group())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            label = str(parsed.get("label", "")).strip().lower()
            if label in valid_set:
                return {"label": label}

    # 3. Last resort: whole-word scan (handles "A categoria correta e mercado.")
    pattern = r"\b(" + "|".join(re.escape(l) for l in valid_set) + r")\b"
    match = re.search(pattern, text_lower)
    if match:
        return {"label": match.group(1)}

    return {"label": "erro"}


def parse_llm_response_multiclass(raw: str) -> dict:
    """Convenience wrapper: ``parse_llm_response(raw, valid_labels=VALID_MULTI_LABELS)``."""
    return parse_llm_response(raw, valid_labels=VALID_MULTI_LABELS)


def _valid_response(parsed: dict) -> bool:
    """Backward-compat helper used by older API tests."""
    return (
        isinstance(parsed, dict)
        and parsed.get("label") in VALID_BINARY_LABELS
    )


# ---------------------------------------------------------------------------
# Few-shot helpers
# ---------------------------------------------------------------------------


def build_few_shot_examples(
    df: "pd.DataFrame",
    *,
    label_column: str = "label",
    text_column: str = "text",
    valid_labels: tuple[str, ...] = VALID_BINARY_LABELS,
    n_per_class: int = 2,
    text_max_chars: int = 500,
    seed: int = 2026,
) -> list[tuple[str, str]]:
    """Sample ``n_per_class`` labeled examples from *df* for few-shot prompting.

    For binary tasks: ``df[label_column]`` is int (0/1). For multiclass tasks:
    ``df[label_column]`` is a string (one of *valid_labels*). The mapping
    ``int -> string`` is fixed: 0 -> "outros", 1 -> "mercado".

    Each text is truncated to ``text_max_chars`` (most Folha articles' lede
    has the topic in the first paragraph; few-shot demos don't need full text).

    Returns a shuffled list of ``(text, label_str)`` tuples ordered randomly so
    classes are interleaved (avoids implicit "all class A then all class B"
    bias in autoregressive models).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    examples: list[tuple[str, str]] = []

    label_dtype = df[label_column].dtype
    is_int_label = label_dtype.kind in ("i", "u", "b")

    for class_idx, class_str in enumerate(valid_labels):
        if is_int_label:
            # binary path
            if class_str == "mercado":
                subset = df[df[label_column] == 1]
            elif class_str == "outros":
                subset = df[df[label_column] == 0]
            else:
                continue  # unknown binary label, skip
        else:
            subset = df[df[label_column] == class_str]

        if len(subset) == 0:
            continue
        n = min(n_per_class, len(subset))
        picked = subset.sample(n=n, random_state=int(rng.integers(0, 10**9)))
        for _, row in picked.iterrows():
            text = (row[text_column] or "")[:text_max_chars]
            examples.append((text, class_str))

    # Shuffle so classes are interleaved
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    return [examples[i] for i in indices]


def build_review_prompt_few_shot(
    text: str,
    *,
    examples: list[tuple[str, str]],
    multiclass: bool = False,
) -> list[dict]:
    """Build a few-shot chat messages array.

    Structure: ``[system, ex1_user, ex1_assistant, ex2_user, ex2_assistant, ...,
    target_user]`` — each example becomes a (user, assistant) pair where the
    assistant message is the bare label (matching the format the prompt asks
    the model to use).

    ``multiclass=True`` uses ``SYSTEM_PROMPT_MULTICLASS``; otherwise ``SYSTEM_PROMPT``.
    """
    system = SYSTEM_PROMPT_MULTICLASS if multiclass else SYSTEM_PROMPT
    messages: list[dict] = [{"role": "system", "content": system}]
    for ex_text, ex_label in examples:
        messages.append({"role": "user", "content": ex_text})
        messages.append({"role": "assistant", "content": ex_label})
    messages.append({"role": "user", "content": text})
    return messages


def classify_single(
    client,
    text: str,
    *,
    model: str = "sabia-4",
    temperature: float = 0,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> dict:
    """Classify a single text via the Sabiá 4 API.

    Returns ``{"label": str, "justificativa": str, "raw_response": str}``.
    On API failure after retries, returns label="erro".
    """
    messages = build_review_prompt(text)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=150,
            )
            raw = response.choices[0].message.content
            parsed = parse_llm_response(raw)
            parsed["raw_response"] = raw
            return parsed

        except Exception as exc:
            wait = retry_delay * (2 ** attempt)
            logger.warning(
                "Tentativa %d/%d falhou: %s. Aguardando %.1fs...",
                attempt + 1, max_retries, exc, wait,
            )
            if attempt < max_retries - 1:
                time.sleep(wait)

    return {
        "label": "erro",
        "justificativa": f"Falha apos {max_retries} tentativas",
        "raw_response": "",
    }


def classify_batch(
    client,
    texts: list[str],
    record_ids: list[str],
    *,
    model: str = "sabia-4",
    temperature: float = 0,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    checkpoint_path=None,
    checkpoint_every: int = 50,
    progress_callback=None,
) -> list[dict]:
    """Classify a list of texts, with optional checkpointing and progress.

    Parameters
    ----------
    checkpoint_path : Path or None
        If provided, saves intermediate results every *checkpoint_every*
        records. On resume, pass previously saved results separately.
    progress_callback : callable or None
        Called with no arguments after each record (e.g., ``tqdm.update``).
    """
    results = []

    for i, (rid, text) in enumerate(zip(record_ids, texts)):
        result = classify_single(
            client, text, model=model, temperature=temperature,
            max_retries=max_retries, retry_delay=retry_delay,
        )
        result["record_id"] = rid
        results.append(result)

        if progress_callback is not None:
            progress_callback()

        if checkpoint_path is not None and (i + 1) % checkpoint_every == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            logger.info("Checkpoint salvo: %d registros", len(results))

    # Final checkpoint
    if checkpoint_path is not None:
        pd.DataFrame(results).to_csv(checkpoint_path, index=False)

    return results


def compute_review_concordance(
    s2_labels: pd.Series,
    sabia_labels: pd.Series,
) -> dict:
    """Compute concordance statistics between S2 and Sabiá 4 labels.

    Ignores records where sabia_labels is "erro".
    """
    mask = sabia_labels != "erro"
    s2 = s2_labels[mask].reset_index(drop=True)
    sabia = sabia_labels[mask].reset_index(drop=True)

    total = len(s2)
    concordant = (s2 == sabia).sum()
    discordant = total - concordant

    # Transition counts
    outros_to_mercado = ((s2 == "outros") & (sabia == "mercado")).sum()
    mercado_to_outros = ((s2 == "mercado") & (sabia == "outros")).sum()

    kappa = cohen_kappa_score(s2, sabia) if total > 0 else 0.0

    return {
        "total": int(total),
        "erros_parsing": int((~mask).sum()),
        "concordant": int(concordant),
        "discordant": int(discordant),
        "concordance_rate": round(concordant / total, 4) if total > 0 else 0.0,
        "cohen_kappa": round(float(kappa), 4),
        "outros_to_mercado": int(outros_to_mercado),
        "mercado_to_outros": int(mercado_to_outros),
    }


# ---------------------------------------------------------------------------
# HuggingFace local backend (Qwen2.5, Llama-3.1, etc.)
# ---------------------------------------------------------------------------

LLM_REGISTRY: dict[str, str] = {
    # Apache 2.0, no gating; ~7.6B params, ~15 GB FP16 — fits Colab L4 (24 GB)
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    # Apache 2.0, no gating; ~7.2B params, ~14 GB FP16 — fits L4 comfortably.
    # Substitui Llama-3.1-8B enquanto acesso a meta-llama/* nao sai.
    "mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    # Llama 3.1 community license (gated on HF). Reativar trocando a chave usada
    # em LLMS no notebook 31 quando o acesso for aprovado.
    # "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
}


def load_hf_model(
    model_name: str,
    *,
    dtype: str = "float16",
    device_map: str = "auto",
):
    """Load tokenizer and causal LM from HuggingFace.

    Returns ``(tokenizer, model)`` ready for ``.generate``. Sets the model to
    eval mode and ensures ``pad_token`` is defined (required for batched
    generation; decoder-only models often default to ``pad_token=None``).

    Designed for chat-instruct checkpoints (Qwen2.5-Instruct, Llama-3.1-Instruct).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models must left-pad for batched generation to align outputs
    tokenizer.padding_side = "left"

    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    return tokenizer, model


def _hf_generate(tokenizer, model, prompts: list[str], *,
                 max_new_tokens: int, temperature: float,
                 max_input_length: int) -> list[str]:
    """Run ``model.generate`` on a batch of pre-templated prompts.

    Returns one string per prompt — only the generated tokens, with the input
    portion sliced off and special tokens stripped.
    """
    import torch

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    input_len = inputs.input_ids.shape[1]
    decoded: list[str] = []
    for output_ids in outputs:
        generated = output_ids[input_len:]
        decoded.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return decoded


def classify_single_hf(
    tokenizer,
    model,
    text: str,
    *,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    max_input_length: int = 2048,
    prompt_builder=None,
    parser=None,
) -> dict:
    """Classify a single text via local HF causal LM (zero-shot).

    ``prompt_builder`` defaults to :func:`build_review_prompt` (binary).
    ``parser`` defaults to :func:`parse_llm_response` (binary labels).
    For multiclass pass ``prompt_builder=build_review_prompt_multiclass``
    and ``parser=parse_llm_response_multiclass``.

    Returns ``{"label": str, "justificativa": str, "raw_response": str}``.
    On parse failure returns ``label="erro"``.
    """
    if prompt_builder is None:
        prompt_builder = build_review_prompt
    if parser is None:
        parser = parse_llm_response

    messages = prompt_builder(text)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    raw = _hf_generate(
        tokenizer, model, [prompt],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        max_input_length=max_input_length,
    )[0]
    parsed = parser(raw)
    parsed["raw_response"] = raw
    return parsed


def classify_batch_hf(
    tokenizer,
    model,
    texts: list[str],
    record_ids: list,
    *,
    method: str,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    batch_size: int = 8,
    max_input_length: int = 2048,
    checkpoint_path: "Path | None" = None,
    checkpoint_every: int = 50,
    progress_callback=None,
    prompt_builder=None,
    parser=None,
) -> list[dict]:
    """Classify a list of texts in batches via local HF causal LM.

    ``prompt_builder``/``parser`` default to the binary helpers; pass the
    multiclass variants for the top-7+other task.

    Each result dict contains ``label``, ``justificativa``, ``raw_response``,
    ``record_id`` and ``method``. ``method`` is the registry key (e.g.
    ``"qwen2.5-7b-instruct"``) and is propagated to the predictions CSV so
    multi-LLM runs can be concatenated without losing provenance.
    """
    if len(texts) != len(record_ids):
        raise ValueError(
            f"texts and record_ids length mismatch: {len(texts)} vs {len(record_ids)}"
        )

    if prompt_builder is None:
        prompt_builder = build_review_prompt
    if parser is None:
        parser = parse_llm_response

    # Defensive: enforce left padding for decoder-only generation
    tokenizer.padding_side = "left"

    results: list[dict] = []
    total = len(texts)

    for batch_start in range(0, total, batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]
        batch_ids = record_ids[batch_start:batch_start + batch_size]

        prompts = [
            tokenizer.apply_chat_template(
                prompt_builder(t), tokenize=False, add_generation_prompt=True,
            )
            for t in batch_texts
        ]

        try:
            raws = _hf_generate(
                tokenizer, model, prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                max_input_length=max_input_length,
            )
        except Exception as exc:  # noqa: BLE001
            # Per-batch failure (OOM, CUDA error, etc.) — record errors and continue
            logger.error("Batch %d-%d failed: %s", batch_start, batch_start + batch_size, exc)
            raws = ["" for _ in batch_texts]

        for j, raw in enumerate(raws):
            parsed = parser(raw) if raw else {
                "label": "erro", "justificativa": "Geracao falhou (batch error)",
            }
            parsed["raw_response"] = raw
            parsed["record_id"] = batch_ids[j]
            parsed["method"] = method
            results.append(parsed)

            if progress_callback is not None:
                progress_callback()

        # Free GPU cache between batches to avoid fragmentation
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

        # Checkpoint logic — save every checkpoint_every records
        if checkpoint_path is not None:
            n_done = len(results)
            if n_done % checkpoint_every < batch_size or n_done == total:
                pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                logger.info("Checkpoint salvo: %d/%d registros", n_done, total)

    if checkpoint_path is not None:
        pd.DataFrame(results).to_csv(checkpoint_path, index=False)

    return results


def hf_results_to_predictions(
    results: list[dict],
    *,
    label_to_id: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Convert raw binary HF classification results into the standard predictions CSV format.

    Output columns: ``index``, ``y_pred`` (0/1), ``y_score`` (0.0/1.0), ``method``,
    plus ``label`` and ``justificativa`` for traceability. Records with
    ``label="erro"`` are dropped (LLM failed to produce a parseable response).

    LLM scores are not natively calibrated probabilities — ``y_score`` is set to
    1.0 for "mercado" and 0.0 for "outros". To get true probabilities, sample
    multiple times with temperature>0 or use logit-based scoring (not implemented).
    """
    if label_to_id is None:
        label_to_id = {"outros": 0, "mercado": 1}

    rows = []
    for r in results:
        if r.get("label") not in label_to_id:
            continue  # skip "erro" rows
        y = label_to_id[r["label"]]
        rows.append({
            "index": r.get("record_id"),
            "y_pred": y,
            "y_score": float(y),
            "method": r.get("method", "hf-llm"),
            "label": r["label"],
            "justificativa": r.get("justificativa", ""),
        })
    return pd.DataFrame(rows)


def hf_results_to_multiclass_predictions(
    results: list[dict],
    *,
    valid_labels: tuple[str, ...] = VALID_MULTI_LABELS,
) -> pd.DataFrame:
    """Convert raw multiclass HF results into the standard multiclass predictions format.

    Output columns: ``index``, ``y_pred`` (string label), ``method``, plus
    ``label`` and ``justificativa`` for traceability. Records with
    ``label="erro"`` (unparseable response) or any label outside *valid_labels*
    are dropped. The output mirrors what ``train_tfidf_multiclass`` and
    ``train_bert_multiclass`` produce — no ``y_score`` column (multiclass
    models in this project don't expose calibrated per-class scores).
    """
    valid = set(valid_labels)
    rows = []
    for r in results:
        label = r.get("label")
        if label not in valid:
            continue  # drop "erro" and anything outside the valid set
        rows.append({
            "index": r.get("record_id"),
            "y_pred": label,
            "method": r.get("method", "hf-llm"),
            "label": label,
            "justificativa": r.get("justificativa", ""),
        })
    return pd.DataFrame(rows)
