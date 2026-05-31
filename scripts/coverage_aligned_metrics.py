"""Re-restringe predicoes de TF-IDF/BERT/ensemble ao subconjunto coberto por cada LLM.

Risco #5 do docs/stil_strategy_note.md: quando LLMs tem coverage<1.0 (amostras
descartadas por output nao parseavel), comparar a metrica do LLM (computada
no subset coberto) com a metrica de TF-IDF/BERT no test set inteiro e
apples-to-oranges. Este script produz, para cada LLM run, a mesma metrica
recomputada para todos os modelos nao-LLM nos indices cobertos pelo LLM.

Usa as funcoes oficiais de src/economy_classifier/evaluation.py para garantir
que os numeros sejam comparaveis com os result_card.json originais.

Saidas:
    artifacts/analysis/coverage_aligned_binary.csv
    artifacts/analysis/coverage_aligned_multiclass.csv

Uso:
    uv run python scripts/coverage_aligned_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from economy_classifier.evaluation import (  # noqa: E402
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_roc_auc,
)

RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"
OUT_DIR = PROJECT_ROOT / "artifacts" / "analysis"
TEST_INDICES_PATH = PROJECT_ROOT / "artifacts" / "splits" / "test_indices.csv"

MULTICLASS_LABELS = [
    "poder", "colunas", "mercado", "esporte",
    "mundo", "cotidiano", "ilustrada", "outros",
]


def load_card(run_dir: Path) -> dict:
    return json.loads((run_dir / "result_card.json").read_text())


def load_preds(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "predictions.csv").set_index("index")


def discover_runs() -> tuple[list[Path], dict[str, list[Path]]]:
    """LLM test_set runs e demais test_set runs agrupados por task."""
    llm_runs: list[Path] = []
    non_llm: dict[str, list[Path]] = {"binary": [], "multiclass": []}
    for d in sorted(RUNS_DIR.iterdir()):
        if not (d / "result_card.json").exists():
            continue
        if not (d / "predictions.csv").exists():
            continue
        if not d.name.endswith("_test_set"):
            continue
        if d.name.startswith("llm_"):
            llm_runs.append(d)
        else:
            task = load_card(d).get("task")
            if task in non_llm:
                non_llm[task].append(d)
    return llm_runs, non_llm


def evaluate_subset(preds: pd.DataFrame, indices: pd.Index, task: str) -> dict:
    """Calcula metricas restringindo `preds` aos `indices` do LLM de referencia."""
    sub = preds.loc[preds.index.intersection(indices)]
    n = len(sub)
    if n == 0:
        return {"n_eval": 0}

    if task == "binary":
        m = compute_binary_metrics(sub["y_true"].values, sub["y_pred"].values)
        if "y_score" in sub.columns:
            try:
                m["auc_roc"] = round(compute_roc_auc(sub["y_true"].values, sub["y_score"].values), 4)
            except ValueError:
                m["auc_roc"] = None
        m["n_eval"] = n
        return m

    m = compute_multiclass_metrics(
        sub["y_true"].values, sub["y_pred"].values, labels=MULTICLASS_LABELS,
    )
    m.pop("per_class_f1", None)
    m["n_eval"] = n
    return m


def coverage_from_card(llm_card: dict, n_covered: int) -> float:
    cov = llm_card.get("metrics", {}).get("coverage")
    if cov is not None:
        return float(cov)
    n_attempted = llm_card.get("n_eval_samples") or 16629
    return n_covered / n_attempted


def build_rows(
    llm_runs: list[Path], non_llm: dict[str, list[Path]],
) -> tuple[list[dict], list[dict]]:
    rows_binary: list[dict] = []
    rows_multi: list[dict] = []

    for llm_dir in llm_runs:
        llm_card = load_card(llm_dir)
        task = llm_card["task"]
        llm_preds = load_preds(llm_dir)
        covered = llm_preds.index
        coverage = coverage_from_card(llm_card, len(covered))

        comparators = list(non_llm[task]) + [llm_dir]
        for comp_dir in comparators:
            comp_card = load_card(comp_dir)
            comp_preds = load_preds(comp_dir)

            sub_metrics = evaluate_subset(comp_preds, covered, task)

            if comp_dir is llm_dir:
                # Comparar a propria contribuicao do LLM no seu subset com nada
                # alem do que o card ja registra. full_* ficam vazios.
                full_metrics = {k: None for k in sub_metrics if k != "n_eval"}
                full_metrics["n_eval"] = None
            else:
                full_metrics = evaluate_subset(comp_preds, comp_preds.index, task)

            row: dict = {
                "reference_llm": llm_dir.name,
                "task": task,
                "n_covered": len(covered),
                "coverage": round(coverage, 4),
                "model_id": comp_card.get("model_id"),
                "comp_run": comp_dir.name,
                "is_self": comp_dir is llm_dir,
            }
            for k, v in sub_metrics.items():
                row[f"{k}_subset"] = v
            for k, v in full_metrics.items():
                row[f"{k}_full"] = v

            (rows_binary if task == "binary" else rows_multi).append(row)

    return rows_binary, rows_multi


def print_summary(df: pd.DataFrame, *, primary: str) -> None:
    """Tabela por LLM ranqueada pela metrica primaria no subset."""
    if df.empty:
        return
    full_col = f"{primary}_full"
    sub_col = f"{primary}_subset"
    for ref, group in df.groupby("reference_llm", sort=False):
        cov = group.iloc[0]["coverage"]
        n = group.iloc[0]["n_covered"]
        print(f"\n  ref={ref}  coverage={cov:.4f}  n_covered={n}")
        ranked = group.sort_values(sub_col, ascending=False)
        for _, r in ranked.iterrows():
            full_val = r[full_col]
            sub_val = r[sub_col]
            full_str = f"{full_val:.4f}" if pd.notna(full_val) else "  N/A "
            self_mark = " <- LLM ref" if r["is_self"] else ""
            delta = ""
            if pd.notna(full_val) and pd.notna(sub_val):
                d = sub_val - full_val
                delta = f"  delta={d:+.4f}"
            print(
                f"    {r['comp_run']:65s}  full={full_str}  subset={sub_val:.4f}{delta}{self_mark}",
            )


def sanity_check_against_cards(rows: list[dict], primary: str) -> None:
    """Confirma que a metrica recomputada do proprio LLM bate com o card."""
    sub_col = f"{primary}_subset"
    for row in rows:
        if not row["is_self"]:
            continue
        ref = row["reference_llm"]
        card = load_card(RUNS_DIR / ref)
        card_val = card.get("metrics", {}).get(primary)
        recomputed = row[sub_col]
        if card_val is None or recomputed is None:
            continue
        if abs(float(card_val) - float(recomputed)) > 1e-3:
            print(
                f"  WARN sanity: {ref} {primary} card={card_val} recomputed={recomputed}",
            )


def main() -> int:
    if not RUNS_DIR.exists():
        print(f"ERRO: {RUNS_DIR} nao existe.")
        return 1

    llm_runs, non_llm = discover_runs()
    if not llm_runs:
        print("Nenhum LLM run *_test_set com predictions encontrado.")
        return 1

    print(f"LLM runs encontrados: {len(llm_runs)}")
    print(f"Non-LLM binary runs:    {len(non_llm['binary'])}")
    print(f"Non-LLM multi runs:     {len(non_llm['multiclass'])}")

    rows_b, rows_m = build_rows(llm_runs, non_llm)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_b = pd.DataFrame(rows_b)
    df_m = pd.DataFrame(rows_m)
    out_b = OUT_DIR / "coverage_aligned_binary.csv"
    out_m = OUT_DIR / "coverage_aligned_multiclass.csv"
    df_b.to_csv(out_b, index=False)
    df_m.to_csv(out_m, index=False)

    print()
    print(f"Binary  rows: {len(df_b):4d}  -> {out_b.relative_to(PROJECT_ROOT)}")
    print(f"Multi   rows: {len(df_m):4d}  -> {out_m.relative_to(PROJECT_ROOT)}")

    print("\n=== Sanity: metrica recomputada do LLM no proprio subset vs card ===")
    sanity_check_against_cards(rows_b, "f1")
    sanity_check_against_cards(rows_m, "macro_f1")
    print("  (sem mensagens acima = todos baterem dentro de 1e-3)")

    print("\n=== Binario: F1 full vs subset por referencia LLM ===")
    print_summary(df_b, primary="f1")

    print("\n=== Multiclasse: macro_F1 full vs subset por referencia LLM ===")
    print_summary(df_m, primary="macro_f1")

    return 0


if __name__ == "__main__":
    sys.exit(main())
