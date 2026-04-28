#!/usr/bin/env python3
"""Export a stacking ensemble for production use.

Usage:
    uv run python scripts/export_ensemble.py s2
    uv run python scripts/export_ensemble.py s3
    uv run python scripts/export_ensemble.py s2 s3

Trains the meta-classifier on validation predictions and serializes it
alongside sub-model references into ``artifacts/ensemble_<name>/``.

Prerequisites:
    Notebooks 01-06 executed and colab_bert_results.zip unpacked.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from economy_classifier.ensemble import (
    discover_runs,
    load_run_predictions,
    save_stacking_classifier,
    train_stacking_classifier,
)
from economy_classifier.project import RUNS_DIR

ENSEMBLE_CONFIGS = {
    "s2": {
        "methods": ["bertimbau", "linearsvc"],
        "description": "Stacking top-2 (BERTimbau + LinearSVC)",
    },
    "s3": {
        "methods": ["bertimbau", "linearsvc", "finbert_ptbr"],
        "description": "Stacking top-3 (BERTimbau + LinearSVC + FinBERT-PT-BR)",
    },
}


def export_ensemble(name: str, runs: dict) -> None:
    cfg = ENSEMBLE_CONFIGS[name]
    methods = cfg["methods"]
    output_dir = PROJECT_ROOT / "artifacts" / f"ensemble_{name}"

    print(f"\n{'=' * 60}")
    print(f"Exportando {name.upper()}: {cfg['description']}")
    print(f"{'=' * 60}")

    # Verify all required methods exist
    missing = [m for m in methods if m not in runs]
    if missing:
        print(f"ERRO: Metodos nao encontrados: {missing}")
        print("Execute os notebooks 01, 11-13, 21 e desempacote os resultados BERT.")
        sys.exit(1)

    # Load validation predictions
    val_scores = {}
    for method in methods:
        vp = load_run_predictions(runs[method]["run_dir"], split="val")
        if vp is None:
            print(f"ERRO: Predicoes de validacao nao encontradas para {method}")
            sys.exit(1)
        vp = vp.sort_values("index").reset_index(drop=True)
        val_scores[method] = vp["y_score"]
        print(f"  {method}: {len(vp):,} predicoes de validacao carregadas")

    y_true_val = vp["y_true"]

    # Train meta-classifier
    print("\nTreinando meta-classificador (LogReg, seed=2026)...")
    meta_model = train_stacking_classifier(val_scores, y_true_val, seed=2026)

    coefs = dict(zip(methods, meta_model.coef_[0]))
    print("Coeficientes:")
    for m, c in sorted(coefs.items(), key=lambda x: -x[1]):
        print(f"  {m:20s} {c:+.4f}")
    print(f"  {'intercept':20s} {meta_model.intercept_[0]:+.4f}")

    # Create output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save meta-classifier
    save_stacking_classifier(
        meta_model, output_dir, feature_names=methods,
    )

    # Save ensemble config with sub-model paths
    config = {
        "ensemble": name.upper(),
        "description": cfg["description"],
        "methods": methods,
        "model_paths": {},
        "seed": 42,
    }

    for method in methods:
        run_dir = runs[method]["run_dir"]
        model_dir = run_dir / "model"
        if not model_dir.exists():
            print(f"AVISO: Diretorio de modelo nao encontrado para {method}: {model_dir}")
            config["model_paths"][method] = str(run_dir)
        else:
            config["model_paths"][method] = str(model_dir)

    (output_dir / "ensemble_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
    )

    print(f"\nEnsemble exportado em: {output_dir}")
    print("Arquivos:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {f.relative_to(output_dir)} ({size_kb:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export stacking ensembles for production use.",
    )
    parser.add_argument(
        "ensembles", nargs="*", default=["s2"],
        choices=list(ENSEMBLE_CONFIGS.keys()),
        help="Ensemble(s) to export (default: s2)",
    )
    args = parser.parse_args()

    runs = discover_runs(RUNS_DIR)

    for name in args.ensembles:
        export_ensemble(name, runs)


if __name__ == "__main__":
    main()
