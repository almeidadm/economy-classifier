#!/usr/bin/env python3
"""Run S2 stacking ensemble predictions from the command line.

Usage:
    echo "Bolsa sobe 2% com alta do dolar" | uv run python scripts/predict.py
    uv run python scripts/predict.py --input texts.txt --output results.csv
    uv run python scripts/predict.py --input texts.csv --format csv

The default model directory is ``artifacts/ensemble_s2``. Use ``--model-dir``
to override.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "ensemble_s2"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify texts using the S2 stacking ensemble.",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
        help="Path to the exported ensemble directory",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Input file (.txt, .csv, or .jsonl). "
             "Reads from stdin if omitted.",
    )
    parser.add_argument(
        "--text-column", type=str, default="text",
        help="Name of the column/field containing the text (default: text)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output file. Writes to stdout if omitted.",
    )
    parser.add_argument(
        "--format", choices=["csv", "jsonl"], default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for BERT inference (default: 16)",
    )
    args = parser.parse_args()

    # Read input texts
    input_df = None  # original DataFrame for CSV/JSONL (preserved in output)
    if args.input is not None:
        input_path = args.input
        if not input_path.exists():
            print(f"ERRO: Arquivo nao encontrado: {input_path}", file=sys.stderr)
            sys.exit(1)

        import pandas as pd

        if input_path.suffix == ".jsonl":
            input_df = pd.read_json(input_path, lines=True)
            if args.text_column not in input_df.columns:
                print(
                    f"ERRO: Coluna '{args.text_column}' nao encontrada no JSONL. "
                    f"Colunas: {list(input_df.columns)}",
                    file=sys.stderr,
                )
                sys.exit(1)
            texts = input_df[args.text_column].fillna("").tolist()
        elif input_path.suffix == ".csv":
            input_df = pd.read_csv(input_path)
            if args.text_column not in input_df.columns:
                print(
                    f"ERRO: Coluna '{args.text_column}' nao encontrada no CSV. "
                    f"Colunas: {list(input_df.columns)}",
                    file=sys.stderr,
                )
                sys.exit(1)
            texts = input_df[args.text_column].fillna("").tolist()
        else:
            texts = [
                line.strip() for line in input_path.read_text().splitlines()
                if line.strip()
            ]
    else:
        texts = [line.strip() for line in sys.stdin if line.strip()]

    if not texts:
        print("ERRO: Nenhum texto fornecido", file=sys.stderr)
        sys.exit(1)

    print(f"Carregando ensemble de {args.model_dir}...", file=sys.stderr)
    from economy_classifier.predict import load_ensemble, predict

    ensemble = load_ensemble(args.model_dir)
    print(f"Classificando {len(texts)} textos...", file=sys.stderr)
    results = predict(ensemble, texts, batch_size=args.batch_size)

    # Merge original fields with predictions when input was CSV/JSONL
    if input_df is not None:
        import pandas as pd
        results = pd.concat(
            [input_df.reset_index(drop=True), results.reset_index(drop=True)],
            axis=1,
        )

    # Write output
    if args.format == "jsonl":
        output_lines = results.to_json(orient="records", lines=True, force_ascii=False)
    else:
        output_lines = results.to_csv(index=False)

    if args.output is not None:
        args.output.write_text(output_lines)
        print(f"Resultados salvos em {args.output}", file=sys.stderr)
    else:
        print(output_lines, end="")


if __name__ == "__main__":
    main()
