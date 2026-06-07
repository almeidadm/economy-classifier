#!/usr/bin/env python3
"""Audita os ``result_card.json`` contra a Guarda Metodologica (CLAUDE.md).

Roda as checagens de :mod:`economy_classifier.card_audit` sobre uma ou mais
arvores de runs e imprime um relatorio pass/fail (matriz de cobertura de
regimes + issues agrupados por severidade). Sai com codigo != 0 quando ha
issues no nivel de corte (``--fail-on``, default ``ERROR``), para uso em CI ou
como gate antes de montar a tabela final do artigo.

Uso:
    uv run python scripts/audit_result_cards.py
    uv run python scripts/audit_result_cards.py artifacts/runs artifacts/ood_runs
    uv run python scripts/audit_result_cards.py --json artifacts/analysis/audit.json
    uv run python scripts/audit_result_cards.py --fail-on warn

Como os cards completos (com _nostop, LLM e OOD) vivem no Google Drive, baixe a
pasta para um diretorio local e aponte os roots para ela.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from economy_classifier.card_audit import (
    SEVERITY_ORDER,
    VALID_REGIMES,
    audit_run_tree,
    classify_method,
    summarize,
)

DEFAULT_ROOTS = ["artifacts/runs", "artifacts/ood_runs"]
_STD_REGIMES = ["fixed_split", "cv_5fold", "test_set"]


def _coverage_matrix(records: list[tuple[str, dict]]) -> str:
    """Tabela (model_id, task) x regimes padrao com presenca marcada."""
    grid: dict[tuple[str, str], set[str]] = {}
    for _folder, card in records:
        key = (card.get("model_id", "?"), card.get("task", "?"))
        grid.setdefault(key, set()).add(card.get("regime", ""))

    rows = sorted(k for k in grid if grid[k] & set(VALID_REGIMES))
    if not rows:
        return "(nenhum card in-domain com regime padrao)"

    name_w = max(len(f"{m} [{t}]") for m, t in rows)
    header = "  ".join(f"{r:^11}" for r in _STD_REGIMES)
    lines = [f"{'model [task]':<{name_w}}  {header}"]
    for model_id, task in rows:
        present = grid[(model_id, task)]
        cells = "  ".join(
            f"{'  ok ' if r in present else '  -- ':^11}" for r in _STD_REGIMES
        )
        lines.append(f"{f'{model_id} [{task}]':<{name_w}}  {cells}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        default=DEFAULT_ROOTS,
        help=f"arvores de runs a auditar (default: {DEFAULT_ROOTS})",
    )
    parser.add_argument("--json", dest="json_out", help="grava o relatorio em JSON")
    parser.add_argument(
        "--fail-on",
        choices=["error", "warn"],
        default="error",
        help="nivel minimo que faz a CLI sair com codigo != 0 (default: error)",
    )
    args = parser.parse_args(argv)

    all_issues = []
    all_records: list[tuple[str, dict]] = []
    roots = [Path(r) for r in args.roots]

    for root in roots:
        if not root.exists():
            print(f"!! root inexistente, pulando: {root}", file=sys.stderr)
            continue
        records, issues = audit_run_tree(root)
        for it in issues:
            all_issues.append((str(root), it))
        all_records.extend(records)
        counts = summarize(issues)
        print(
            f"== {root}: {len(records)} cards "
            f"(ERROR={counts['ERROR']} WARN={counts['WARN']} INFO={counts['INFO']})"
        )

    if not all_records and not all_issues:
        print("\nNenhum result_card.json encontrado nos roots informados.")
        return 0

    # --- matriz de cobertura ---------------------------------------------
    print("\n--- Cobertura de regimes (in-domain) ---")
    print(_coverage_matrix(all_records))

    # --- issues agrupados por severidade ---------------------------------
    ordered = sorted(all_issues, key=lambda ri: (SEVERITY_ORDER[ri[1].severity], ri[1].code))
    totals = summarize([it for _r, it in all_issues])
    print(
        f"\n--- Issues (ERROR={totals['ERROR']} "
        f"WARN={totals['WARN']} INFO={totals['INFO']}) ---"
    )
    if not all_issues:
        print("  nenhum issue — todos os cards passam na guarda.")
    last_sev = None
    for _root, it in ordered:
        if it.severity != last_sev:
            print(f"\n[{it.severity}]")
            last_sev = it.severity
        print(f"  {it.code:<18} {it.source:<48} {it.message}")

    # --- saida JSON opcional ---------------------------------------------
    if args.json_out:
        payload = {
            "roots": [str(r) for r in roots],
            "n_cards": len(all_records),
            "summary": totals,
            "methods": sorted({classify_method(c.get("model_id", "")) for _f, c in all_records}),
            "issues": [
                {
                    "root": root,
                    "severity": it.severity,
                    "code": it.code,
                    "source": it.source,
                    "message": it.message,
                }
                for root, it in ordered
            ],
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nRelatorio JSON gravado em {out}")

    threshold = SEVERITY_ORDER["ERROR" if args.fail_on == "error" else "WARN"]
    blocking = [it for _r, it in all_issues if SEVERITY_ORDER[it.severity] <= threshold]
    if blocking:
        print(f"\nFALHA: {len(blocking)} issue(s) no nivel >= {args.fail_on.upper()}.")
        return 1
    print("\nOK: nenhum issue no nivel de corte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
