"""Auditoria de ``result_card.json`` contra a Guarda Metodologica (CLAUDE.md).

Logica pura e testavel que valida cada result card emitido pelos runs
(in-domain e OOD) contra as invariantes da guarda metodologica, antes de os
numeros entrarem na tabela final do artigo. A CLI fina vive em
``scripts/audit_result_cards.py``.

Invariantes verificadas (codigo do issue entre parenteses):

* ``tfidf-missing-hp``  — modelo TF-IDF com ``hyperparameter_search=null``
  (anti-padrao "Pular a busca": sem o payload o TF-IDF nao entra na tabela).
* ``bad-scoring``       — ``hyperparameter_search.scoring`` incompativel com a
  tarefa (ex.: ``f1_macro`` em binario; ``f1`` em multiclasse).
* ``unexpected-hp``     — BERT/LLM com busca populada (excecao declarada: config
  fixa da literatura, ``hyperparameter_search`` deve ser ``null``).
* ``cv-no-variance``    — card ``cv_5fold`` sem a chave ``*_std`` da metrica
  primaria (anti-padrao "F1 unico sem variancia da CV").
* ``missing-primary``   — ausencia da metrica primaria (``f1`` / ``macro_f1``).
* ``missing-regime``    — modelo sem um dos regimes esperados (TF-IDF/BERT/
  ensemble => 3 regimes; LLM => apenas ``test_set``).
* ``llm-coverage``      — LLM com ``coverage<1.0`` (avaliacao em subconjunto;
  comparacao precisa alinhar indices).
* ``llm-auc``           — LLM reportando ``auc_roc`` (scores deterministicos;
  AUC deve ser marcado ``N/A`` no artigo).
* ``no-calibration``    — TF-IDF/BERT binario (regime pontual) sem Brier/ECE.
* ``no-predictions`` / ``no-eval-n`` — sidecar ``predictions.csv`` ausente ou
  ``n_eval_samples`` vazio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from economy_classifier.project import (
    VALID_REGIMES,
    VALID_TASKS,
    _VALID_SCORING_BY_TASK as VALID_SCORING_BY_TASK,
)

SEVERITY_ORDER = {"ERROR": 0, "WARN": 1, "INFO": 2}

# Quantos regimes cada metodo deve cobrir. LLM e avaliado apenas no test_set
# (sem fixed_split/cv_5fold) por design — ver CLAUDE.md, secao LLMs.
EXPECTED_REGIMES: dict[str, set[str]] = {
    "tfidf": set(VALID_REGIMES),
    "bert": set(VALID_REGIMES),
    "ensemble": set(VALID_REGIMES),
    "llm": {"test_set"},
}

_REQUIRED_FIELDS = ("model_id", "task", "regime", "metrics", "cost", "config")


@dataclass(frozen=True)
class Issue:
    """Uma violacao (ou aviso) encontrada em um card."""

    severity: str  # ERROR | WARN | INFO
    code: str
    message: str
    source: str = ""  # nome da pasta do run / identificador

    def __str__(self) -> str:  # pragma: no cover - formatacao trivial
        where = f" [{self.source}]" if self.source else ""
        return f"{self.severity:<5} {self.code}{where}: {self.message}"


def classify_method(model_id: str) -> str:
    """Mapeia ``model_id`` para a familia de metodo via prefixo."""
    mid = (model_id or "").lower()
    for prefix in ("tfidf", "bert", "llm", "ensemble"):
        if mid.startswith(prefix):
            return prefix
    return "unknown"


def is_ood_card(card: dict) -> bool:
    """Card de avaliacao out-of-domain (inferencia em base com shift).

    Cards OOD reusam um modelo ja treinado in-domain e o aplicam a outra base;
    por isso tem ``regime='test_set'`` e ``hyperparameter_search=null`` por
    construcao, e nao participam da cobertura de regimes (nao existem
    ``cv_5fold``/``fixed_split`` OOD). Sao identificados por
    ``config.evaluation_type`` comecando com ``out_of_domain``.
    """
    cfg = card.get("config") or {}
    et = cfg.get("evaluation_type")
    return isinstance(et, str) and et.startswith("out_of_domain")


def _metric_aliases(name: str) -> set[str]:
    """``macro_f1`` e ``f1_macro`` sao a mesma metrica em cards diferentes."""
    aliases = {name}
    if "macro_f1" in name:
        aliases.add(name.replace("macro_f1", "f1_macro"))
    if "f1_macro" in name:
        aliases.add(name.replace("f1_macro", "macro_f1"))
    return aliases


def _has_metric(metrics: dict, name: str) -> bool:
    return bool(_metric_aliases(name) & set(metrics))


def _primary_base(task: str) -> str:
    return "f1" if task == "binary" else "macro_f1"


def primary_metric_value(card: dict) -> dict | None:
    """Valor da metrica primaria do card (``f1`` binario / ``macro_f1`` multi).

    Retorna ``{'value', 'std', 'metric'}`` ou ``None`` se ausente. Em
    ``cv_5fold`` usa ``*_mean``/``*_std``; nos demais regimes (incl. OOD,
    ``regime='test_set'``) usa o ponto. Aceita o alias ``f1_macro``/``macro_f1``.
    """
    task = card.get("task")
    if task not in VALID_TASKS:
        return None
    metrics = card.get("metrics") or {}
    base = _primary_base(task)

    def _get(name):
        for alias in _metric_aliases(name):
            if alias in metrics:
                return metrics[alias]
        return None

    if card.get("regime") == "cv_5fold":
        value, std = _get(f"{base}_mean"), _get(f"{base}_std")
    else:
        value, std = _get(base), None
    if value is None:
        return None
    return {"value": value, "std": std, "metric": base}


def audit_card(
    card: dict,
    *,
    source: str = "",
    predictions_exists: bool | None = None,
) -> list[Issue]:
    """Aplica todas as checagens de guarda a um unico card.

    ``predictions_exists`` (quando informado) liga a checagem do sidecar
    ``predictions.csv``; ``None`` a desliga (util em testes de card puro).
    """
    issues: list[Issue] = []

    def add(severity: str, code: str, message: str) -> None:
        issues.append(Issue(severity, code, message, source))

    # --- estrutura minima -------------------------------------------------
    for key in _REQUIRED_FIELDS:
        if key not in card:
            add("ERROR", "missing-field", f"card sem campo obrigatorio {key!r}")
    if "task" not in card or "metrics" not in card:
        return issues  # sem isso nao da pra prosseguir

    task = card.get("task")
    regime = card.get("regime")
    metrics = card.get("metrics") or {}
    model_id = card.get("model_id", "")
    method = classify_method(model_id)

    if task not in VALID_TASKS:
        add("ERROR", "bad-task", f"task={task!r} fora de {sorted(VALID_TASKS)}")
        return issues

    ood = is_ood_card(card)
    if regime not in VALID_REGIMES and not ood:
        add(
            "ERROR",
            "bad-regime",
            f"regime={regime!r} fora de {sorted(VALID_REGIMES)} e card nao e OOD",
        )
    if ood:
        et = (card.get("config") or {}).get("evaluation_type")
        add(
            "INFO",
            "ood-card",
            f"card OOD (evaluation_type={et!r}); cobertura de regimes e busca "
            "de HP relaxadas — e inferencia reusando modelo treinado, nao treino",
        )

    # --- busca de hiperparametros -----------------------------------------
    # scoring sanity vale sempre que houver payload (qualquer metodo/contexto).
    hp = card.get("hyperparameter_search")
    if hp is not None:
        sc = hp.get("scoring")
        allowed = VALID_SCORING_BY_TASK.get(task, set())
        if sc is not None and sc not in allowed:
            add(
                "ERROR",
                "bad-scoring",
                f"hyperparameter_search.scoring={sc!r} incompativel com "
                f"task={task!r}; esperado {sorted(allowed)}",
            )
    # Expectativas por metodo so valem para runs in-domain (treino). Cards OOD
    # sao inferencia: reaproveitam o modelo in-domain, entao hp=null e correto.
    if not ood:
        if method == "tfidf" and hp is None:
            add(
                "ERROR",
                "tfidf-missing-hp",
                "TF-IDF com hyperparameter_search=null — ficaria fora da "
                "tabela final (guarda exige RandomizedSearchCV antes dos regimes)",
            )
        elif method in ("bert", "llm") and hp is not None:
            add(
                "WARN",
                "unexpected-hp",
                f"{method} com hyperparameter_search populado; guarda declara "
                f"config fixa da literatura (esperado null)",
            )

    # --- metrica primaria + variancia da CV -------------------------------
    base = _primary_base(task)
    if not ood and regime == "cv_5fold":
        if not _has_metric(metrics, f"{base}_mean"):
            add("ERROR", "missing-primary", f"cv_5fold sem {base}_mean")
        if not _has_metric(metrics, f"{base}_std"):
            add(
                "ERROR",
                "cv-no-variance",
                f"cv_5fold sem {base}_std — guarda exige media +/- desvio, "
                "nao estimativa pontual",
            )
    else:
        # regimes pontuais (fixed_split/test_set) e cards OOD
        if not (_has_metric(metrics, base) or _has_metric(metrics, f"{base}_mean")):
            sev = "WARN" if ood else "ERROR"
            add(sev, "missing-primary", f"sem metrica primaria {base!r} (task={task})")

    # --- especifico de LLM ------------------------------------------------
    if method == "llm":
        cov = metrics.get("coverage")
        if cov is not None and cov < 1.0:
            add(
                "WARN",
                "llm-coverage",
                f"coverage={cov:.4f} < 1.0 — avaliacao em subconjunto; alinhar "
                "TF-IDF/BERT aos mesmos indices na comparacao",
            )
        if "auc_roc" in metrics or "auc_roc_mean" in metrics:
            add(
                "WARN",
                "llm-auc",
                "LLM reporta auc_roc, mas scores sao deterministicos — marcar "
                "AUC como 'N/A — deterministic scores' no artigo",
            )

    # --- calibracao (informativo) ----------------------------------------
    if (
        method in ("tfidf", "bert")
        and task == "binary"
        and regime in ("fixed_split", "test_set")
        and not ({"brier", "ece"} & set(metrics))
    ):
        add(
            "WARN",
            "no-calibration",
            "sem Brier/ECE — guarda lista essas metricas como reportaveis "
            "para TF-IDF/BERT binario calibrado",
        )

    # --- sidecar e contagem ----------------------------------------------
    if predictions_exists is False:
        add("ERROR", "no-predictions", "predictions.csv ausente ou vazio ao lado do card")
    if card.get("n_eval_samples") in (None, 0):
        add("WARN", "no-eval-n", "n_eval_samples ausente ou zero")

    return issues


def audit_run_tree(root: str | Path) -> tuple[list[tuple[str, dict]], list[Issue]]:
    """Audita todos os ``*/result_card.json`` sob *root*.

    Retorna ``(records, issues)`` onde ``records`` e a lista de
    ``(folder_name, card)`` lidos com sucesso e ``issues`` agrega checagens
    por-card e a cobertura de regimes entre cards.
    """
    root = Path(root)
    records: list[tuple[str, dict]] = []
    issues: list[Issue] = []
    seen: dict[tuple[str, str], set[str]] = {}

    for cp in sorted(root.glob("*/result_card.json")):
        folder = cp.parent.name
        try:
            card = json.loads(cp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            issues.append(Issue("ERROR", "unreadable", f"falha ao ler card: {exc}", folder))
            continue
        preds = cp.parent / "predictions.csv"
        preds_ok = preds.exists() and preds.stat().st_size > 0
        issues.extend(audit_card(card, source=folder, predictions_exists=preds_ok))
        records.append((folder, card))
        # Cards OOD nao participam da cobertura de regimes (sao inferencia em
        # base com shift; nao existem cv_5fold/fixed_split OOD por construcao).
        if not is_ood_card(card):
            key = (card.get("model_id", ""), card.get("task", ""))
            seen.setdefault(key, set()).add(card.get("regime", ""))

    # --- cobertura de regimes (so para cards in-domain) -------------------
    for (model_id, task), regimes in sorted(seen.items()):
        std_present = regimes & set(VALID_REGIMES)
        if not std_present:
            continue  # sem nenhum regime padrao => OOD, nao se aplica
        method = classify_method(model_id)
        expected = EXPECTED_REGIMES.get(method, set(VALID_REGIMES))
        for missing in sorted(expected - regimes):
            sev = "ERROR" if missing == "cv_5fold" else "WARN"
            issues.append(
                Issue(
                    sev,
                    "missing-regime",
                    f"{model_id}/{task}: sem card {missing} "
                    f"(presentes: {sorted(std_present)})",
                    f"{model_id}_{task}",
                )
            )

    return records, issues


def summarize(issues: list[Issue]) -> dict[str, int]:
    """Conta issues por severidade."""
    counts = {"ERROR": 0, "WARN": 0, "INFO": 0}
    for it in issues:
        counts[it.severity] = counts.get(it.severity, 0) + 1
    return counts
