"""LLM-based review of ensemble predictions using Maritaca AI Sabiá 4.

Provides prompt construction, API interaction with retry/checkpoint,
response parsing, and concordance analysis for auditing S2 predictions.
"""

from __future__ import annotations

import json
import logging
import re
import time

import pandas as pd
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Voce e um classificador binario de textos jornalisticos brasileiros.

Sua tarefa: dado um trecho de texto, classifique-o segundo a sua tematica \
central como "mercado" (texto de tematica economica em sentido amplo) ou \
"outros" (qualquer outra tematica). O rotulo "mercado" e usado por convencao \
da pipeline para denotar a classe positiva "economia".

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

Responda APENAS com um JSON no formato:
{"label": "mercado" ou "outros", "justificativa": "<1 frase curta>"}"""


def build_review_prompt(text: str) -> list[dict]:
    """Build the messages array for the Sabiá 4 review API call (zero-shot).

    Returns a list with a system message and the user query, compatible with
    the OpenAI chat completions API.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def parse_llm_response(raw: str) -> dict:
    """Extract label and justification from a raw LLM response string.

    Handles pure JSON, markdown-fenced JSON, and extra surrounding text.
    Returns ``{"label": ..., "justificativa": ...}`` on success, or
    ``{"label": "erro", "justificativa": "<reason>"}`` on failure.
    """
    text = raw.strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        if _valid_response(parsed):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if _valid_response(parsed):
                return parsed
        except json.JSONDecodeError:
            pass

    # Try finding first {...} block
    brace_match = re.search(r"\{[^{}]*\}", text)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group())
            if _valid_response(parsed):
                return parsed
        except json.JSONDecodeError:
            pass

    return {"label": "erro", "justificativa": f"Resposta nao parseavel: {text[:200]}"}


def _valid_response(parsed: dict) -> bool:
    return (
        isinstance(parsed, dict)
        and parsed.get("label") in ("mercado", "outros")
        and "justificativa" in parsed
    )


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
