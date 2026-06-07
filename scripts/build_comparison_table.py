#!/usr/bin/env python3
"""Monta tabelas comparativas a partir dos result cards espelhados do Drive.

Le todos os ``result_card.json`` sob ``artifacts/drive_audit/{runs,ood_runs}``
e o inventario de pastas (``folder_inventory.json``) e produz pivots
**modelo x regime** (in-domain) e **modelo x dataset** (OOD), tornando
explicitas as ausencias e destacando o vencedor de cada coluna.

  **valor** (negrito) -> melhor metrica primaria da coluna (vencedor)
  valor               -> card presente (F1 binario / macro-F1 multiclasse;
                         cv_5fold mostra media±desvio, comparado pela media)
  ``∅``               -> a pasta do run existe no Drive mas SEM result_card
                         (lacuna real)
  ``—``               -> combinacao inexistente/nao planejada (ex.: LLM e
                         stacking so tem test_set)
  ``?``               -> card sem a metrica primaria

Saidas:
  docs/tabela_comparativa_resultados.md
  artifacts/analysis/comparison_indomain.csv
  artifacts/analysis/comparison_ood.csv

Uso:  uv run python scripts/build_comparison_table.py
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

from economy_classifier.card_audit import (
    classify_method,
    is_ood_card,
    primary_metric_value,
)

ROOT = Path("artifacts/drive_audit")
REGIMES = ["fixed_split", "cv_5fold", "test_set"]
EMPTY = "∅"   # pasta existe, sem card
NONE_ = "—"   # pasta nao existe
_EPS = 1e-9

_FAMILY_ORDER = {"tfidf": 0, "bert": 1, "ensemble": 2, "llm": 3, "unknown": 4}


def _load(group: str) -> list[tuple[str, dict]]:
    out = []
    for p in sorted(glob.glob(str(ROOT / group / "*" / "result_card.json"))):
        out.append((Path(p).parent.name, json.loads(Path(p).read_text(encoding="utf-8"))))
    return out


def _inventory() -> dict:
    p = ROOT / "folder_inventory.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"runs": {}, "ood_runs": {}}


def _regime_of(folder: str) -> str | None:
    for r in REGIMES:
        if folder.endswith("_" + r):
            return r
    return None


def _cell(pm: dict | None) -> dict:
    """Celula com display string ``s`` e valor numerico ``v`` (ou None)."""
    if pm is None:
        return {"s": "?", "v": None}
    if pm["std"] is not None:
        return {"s": f"{pm['value']:.3f}±{pm['std']:.3f}", "v": pm["value"]}
    return {"s": f"{pm['value']:.3f}", "v": pm["value"]}


def _absent(marker: str) -> dict:
    return {"s": marker, "v": None}


def _row_sort_key(label: str) -> tuple:
    return (_FAMILY_ORDER.get(classify_method(label), 9), label)


def _col_winners(rows: dict, cols: list[str]) -> dict:
    """Maior valor (vencedor) por coluna; higher-is-better p/ F1/macro-F1."""
    best = {}
    for col in cols:
        vals = [rows[r][col]["v"] for r in rows if rows[r][col]["v"] is not None]
        best[col] = max(vals) if vals else None
    return best


def _render_cell(cell: dict, best_v) -> str:
    s = cell["s"]
    if cell["v"] is not None and best_v is not None and abs(cell["v"] - best_v) < _EPS:
        return f"**{s}**"
    return s


# ---------------------------------------------------------------------------
# In-domain: modelo x regime, por tarefa
# ---------------------------------------------------------------------------
def build_indomain(cards, inv) -> dict:
    inv_runs = inv.get("runs", {})
    card_at: dict[tuple[str, str], dict] = {}
    tasks_of: dict[str, str] = {}
    for folder, card in cards:
        if is_ood_card(card):
            continue
        reg = card.get("regime")
        if reg not in REGIMES:
            continue
        row_key = folder[: -(len(reg) + 1)]
        card_at[(row_key, reg)] = card
        tasks_of[row_key] = card.get("task", "?")

    row_keys: set[str] = set(tasks_of)
    for folder in inv_runs:
        reg = _regime_of(folder)
        if reg is None:
            continue
        rk = folder[: -(len(reg) + 1)]
        row_keys.add(rk)
        tasks_of.setdefault(rk, "binary" if "_binary" in rk else "multiclass")

    out: dict[str, dict] = {"binary": {}, "multiclass": {}}
    for rk in row_keys:
        task = tasks_of.get(rk, "?")
        label = rk.replace(f"_{task}", "", 1)
        cells = {}
        for reg in REGIMES:
            folder = f"{rk}_{reg}"
            if (rk, reg) in card_at:
                cells[reg] = _cell(primary_metric_value(card_at[(rk, reg)]))
            elif folder in inv_runs:
                cells[reg] = _absent(EMPTY)
            else:
                cells[reg] = _absent(NONE_)
        out.setdefault(task, {})[label] = cells
    return out


# ---------------------------------------------------------------------------
# OOD: modelo x dataset, por tarefa
# ---------------------------------------------------------------------------
def build_ood(cards) -> dict:
    table: dict[str, dict] = {}
    datasets: dict[str, set] = {}
    all_models: set[str] = set()
    for folder, card in cards:
        if not is_ood_card(card):
            continue
        task = card.get("task", "?")
        model = card.get("model_id", "?")
        all_models.add(model)
        ds = folder
        pref = f"{model}_{task}_"
        if folder.startswith(pref):
            ds = folder[len(pref):]
        table.setdefault(task, {}).setdefault(model, {})[ds] = _cell(primary_metric_value(card))
        datasets.setdefault(task, set()).add(ds)
    for task in table:
        cols = sorted(datasets[task])
        for model in all_models:
            row = table[task].setdefault(model, {})
            for ds in cols:
                row.setdefault(ds, _absent(NONE_))
    return {t: (sorted(datasets[t]), table[t]) for t in table}


# ---------------------------------------------------------------------------
# Render markdown (com vencedores em negrito por coluna)
# ---------------------------------------------------------------------------
def _md_indomain(table: dict, task: str) -> str:
    rows = table.get(task, {})
    if not rows:
        return f"_(sem dados para {task})_\n"
    best = _col_winners(rows, REGIMES)
    lines = [
        "| Modelo | fixed_split | cv_5fold (média±dp) | test_set |",
        "|---|---|---|---|",
    ]
    for label in sorted(rows, key=_row_sort_key):
        c = rows[label]
        cells = " | ".join(_render_cell(c[r], best[r]) for r in REGIMES)
        lines.append(f"| `{label}` | {cells} |")
    return "\n".join(lines) + "\n"


def _md_ood(ood: dict, task: str) -> str:
    if task not in ood:
        return f"_(sem dados OOD para {task})_\n"
    datasets, rows = ood[task]
    best = _col_winners(rows, datasets)
    header = "| Modelo | " + " | ".join(datasets) + " |"
    sep = "|---|" + "|".join("---" for _ in datasets) + "|"
    lines = [header, sep]
    for model in sorted(rows, key=_row_sort_key):
        cells = " | ".join(_render_cell(rows[model][ds], best[ds]) for ds in datasets)
        lines.append(f"| `{model}` | {cells} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CSV (uma linha por card) + marca de vencedor
# ---------------------------------------------------------------------------
def _write_csv(path: Path, cards, ood: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["folder", "model_id", "task", "regime", "evaluation_type",
            "primary_metric", "value", "std", "accuracy", "n_eval_samples",
            "coverage", "has_hp_search"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for folder, card in sorted(cards):
            if is_ood_card(card) != ood:
                continue
            m = card.get("metrics") or {}
            pm = primary_metric_value(card)
            w.writerow({
                "folder": folder,
                "model_id": card.get("model_id"),
                "task": card.get("task"),
                "regime": card.get("regime"),
                "evaluation_type": (card.get("config") or {}).get("evaluation_type", ""),
                "primary_metric": pm["metric"] if pm else "",
                "value": pm["value"] if pm else "",
                "std": pm["std"] if pm and pm["std"] is not None else "",
                "accuracy": m.get("accuracy", m.get("accuracy_mean", "")),
                "n_eval_samples": card.get("n_eval_samples", ""),
                "coverage": m.get("coverage", ""),
                "has_hp_search": card.get("hyperparameter_search") is not None,
            })


def main():
    runs = _load("runs")
    ood_cards = _load("ood_runs")
    inv = _inventory()
    indomain = build_indomain(runs, inv)
    ood = build_ood(ood_cards)
    n_empty = sum(1 for g in inv.values() for v in g.values() if not v)

    md = []
    md.append("# Tabela comparativa de resultados (Drive)\n")
    md.append("**Data:** 2026-06-04 · **Fonte:** `artifacts/drive_audit/` "
              f"({len(runs)} cards in-domain + {len(ood_cards)} OOD).\n")
    md.append("**Métrica primária:** F1 (binário) / macro-F1 (multiclasse). "
              "Em `cv_5fold`, média±desvio dos 5 folds (comparado pela média).\n")
    md.append("**Destaque:** **negrito** = vencedor da coluna (maior métrica do "
              "regime/dataset). Empates são marcados juntos.\n")
    md.append("**Ausências:** "
              f"`{EMPTY}` = pasta existe no Drive **sem** `result_card.json` "
              f"(lacuna real); `{NONE_}` = combinação inexistente/não planejada "
              "(LLM e stacking só têm `test_set`); `?` = card sem a métrica.\n")
    md.append(f"> Pastas sem card no inventário: **{n_empty}** "
              "(ver `docs/auditoria_result_cards.md`).\n")

    md.append("\n## 1. In-domain — BINÁRIO (F1)\n")
    md.append(_md_indomain(indomain, "binary"))
    md.append("\n## 2. In-domain — MULTICLASSE (macro-F1)\n")
    md.append(_md_indomain(indomain, "multiclass"))

    md.append("\n## 3. OOD — BINÁRIO (F1, inferência por dataset)\n")
    md.append("> Cards OOD são inferência (`regime=test_set`, sem cv/fixed). "
              "Vencedor por dataset em **negrito**.\n")
    md.append(_md_ood(ood, "binary"))
    if "multiclass" in ood:
        md.append("\n## 4. OOD — MULTICLASSE (macro-F1)\n")
        md.append(_md_ood(ood, "multiclass"))

    md.append("\n## 5. Notas sobre ausências\n")
    md.append(
        "- **Por design (`—`):** LLM (4 variantes × tarefa) e `ensemble_stacking` "
        "só têm `test_set`. OOD é só inferência (`test_set`).\n"
        "- **Lacunas reais (`∅`):** `ensemble_agreement` (binário e multiclasse, "
        "6 pastas, 0 cards); `bert_norberto_base_nostop` multiclasse sem `test_set`; "
        "`bert_NorBERTo` (legado) `cv_5fold` vazio.\n"
        "- **Card legado:** `bert_NorBERTo` (binário) é resquício de nomenclatura — "
        "o modelo válido é `bert_norberto_base_nostop`. Recomendado remover.\n"
        "- **Sem linha (não executado):** `ensemble_weighted` foi rodado **só em "
        "binário**. `ensemble_agreement`, LLM e stacking não têm execução OOD.\n"
        "- **Detalhamento e plano de remediação:** `docs/auditoria_result_cards.md`.\n"
    )

    out_md = Path("docs/tabela_comparativa_resultados.md")
    out_md.write_text("\n".join(md), encoding="utf-8")
    _write_csv(Path("artifacts/analysis/comparison_indomain.csv"), runs + ood_cards, ood=False)
    _write_csv(Path("artifacts/analysis/comparison_ood.csv"), runs + ood_cards, ood=True)

    # resumo dos vencedores no stdout
    print(f"Markdown:  {out_md}")
    print("CSV:       artifacts/analysis/comparison_indomain.csv, comparison_ood.csv")
    for task in ("binary", "multiclass"):
        rows = indomain.get(task, {})
        best = _col_winners(rows, REGIMES)
        win = {r: next((lbl for lbl in rows if rows[lbl][r]["v"] == best[r]), None)
               for r in REGIMES if best[r] is not None}
        print(f"Vencedor in-domain {task}: " +
              ", ".join(f"{r}={win[r]}({best[r]:.3f})" for r in win))


if __name__ == "__main__":
    main()
