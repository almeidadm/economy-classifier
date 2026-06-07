"""McNemar pareado (Bonferroni) entre uma familia de modelos no test_set.

Orquestrador fino sobre ``evaluation.compute_mcnemar_pairwise`` — carrega os
``predictions.csv`` de cada run, valida o alinhamento (mesmos indices, mesmo
``y_true``) e imprime a tabela de pares com ``p_value_adjusted`` e
``significant_after_correction``, como exige a guarda metodologica.

Uso:
    uv run python scripts/mcnemar_family.py \
        bert_norberto_base_nostop bert_bertimbau_nostop bert_finbert_ptbr_nostop \
        bert_deb3rta_base_nostop tfidf_logreg_nostop tfidf_linearsvc_nostop tfidf_nb_nostop \
        --task binary --out artifacts/analysis/mcnemar_nostop_binary.csv

Pre-requisito: ``<runs-dir>/<model>_<task>_test_set/predictions.csv`` reais
(nao stubs). Para predicoes que so existem no Drive, empacote com
``scripts/colab_pack_results.py`` e desempacote com
``scripts/colab_unpack_streaming.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from economy_classifier.evaluation import compute_mcnemar_pairwise  # noqa: E402


def load_predictions(runs_dir: Path, model_id: str, task: str, regime: str) -> pd.Series:
    path = runs_dir / f"{model_id}_{task}_{regime}" / "predictions.csv"
    if not path.exists():
        raise SystemExit(f"ERRO: {path} nao existe — sincronize do Drive primeiro.")
    if path.stat().st_size < 1000:
        raise SystemExit(f"ERRO: {path} parece stub ({path.stat().st_size} bytes) — sincronize o CSV real.")
    df = pd.read_csv(path)
    missing = {"index", "y_true", "y_pred"} - set(df.columns)
    if missing:
        raise SystemExit(f"ERRO: {path} sem colunas {sorted(missing)}.")
    df = df.set_index("index").sort_index()
    if df.index.duplicated().any():
        raise SystemExit(f"ERRO: {path} tem indices duplicados.")
    return df["y_true"], df["y_pred"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("models", nargs="+", help="model_ids da familia (>= 2)")
    parser.add_argument("--task", choices=["binary", "multiclass"], required=True)
    parser.add_argument("--regime", default="test_set")
    parser.add_argument("--runs-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=None, help="CSV de saida (opcional)")
    args = parser.parse_args()

    if len(args.models) < 2:
        raise SystemExit("ERRO: informe pelo menos 2 modelos.")

    y_true_ref: pd.Series | None = None
    ref_model = None
    preds: dict[str, pd.Series] = {}
    for model in args.models:
        y_true, y_pred = load_predictions(args.runs_dir, model, args.task, args.regime)
        if y_true_ref is None:
            y_true_ref, ref_model = y_true, model
        else:
            if not y_true.index.equals(y_true_ref.index):
                raise SystemExit(
                    f"ERRO: indices de {model} nao batem com {ref_model} "
                    f"({len(y_true)} vs {len(y_true_ref)} amostras). "
                    "Coverage divergente? Alinhe antes (scripts/coverage_aligned_metrics.py)."
                )
            if not (y_true.values == y_true_ref.values).all():
                raise SystemExit(f"ERRO: y_true de {model} difere de {ref_model} — arquivos corrompidos ou de splits distintos.")
        preds[model] = y_pred

    acc = {m: float((p.values == y_true_ref.values).mean()) for m, p in preds.items()}

    table = compute_mcnemar_pairwise(y_true_ref.values, {m: p.values for m, p in preds.items()}, alpha=args.alpha)

    n = len(y_true_ref)
    k = len(preds)
    print(f"\nFamilia: {k} modelos | task={args.task} regime={args.regime} | n={n} | "
          f"{k * (k - 1) // 2} pares (Bonferroni) | alpha={args.alpha}\n")
    print("Accuracy no test (contexto, nao metrica primaria):")
    for m, a in sorted(acc.items(), key=lambda kv: -kv[1]):
        print(f"  {m:42s} {a:.4f}")
    print()
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.to_string(index=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"\nSalvo em {args.out}")


if __name__ == "__main__":
    main()
