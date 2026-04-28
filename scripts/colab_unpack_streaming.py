#!/usr/bin/env python3
"""Streaming, selective unpacker for the multi-zip Colab runs download.

Google Drive splits large folder downloads into ``runs-<timestamp>-XXX.zip``
chunks. Each chunk holds a *disjoint* slice of the ``runs/`` tree, and the
union reconstructs the full result bundle. Naively running ``extractall`` on
all chunks would materialise tens of GB of files we never consume downstream
(BERT model weights, optimizer states, intermediate checkpoints).

This script:

1. Walks each zip's central directory without loading payloads into memory.
2. Applies a whitelist/blacklist matching what notebooks 43_ensemble and
   42_comparacao actually read (result_card.json, predictions.csv, etc.).
3. Streams allowed entries with ``shutil.copyfileobj`` in 1 MiB chunks so
   peak RAM stays under ~50 MB regardless of zip or file size.
4. Validates the zip (CRC + extraction count) before optionally deleting it
   to free disk pressure between chunks.

Usage:
    uv run python scripts/colab_unpack_streaming.py --dry-run
    uv run python scripts/colab_unpack_streaming.py --pattern 'runs-*-019.zip'
    uv run python scripts/colab_unpack_streaming.py --delete-after
"""

from __future__ import annotations

import argparse
import errno
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNS_PREFIX = "runs/"
COPY_CHUNK = 1024 * 1024  # 1 MiB

WHITELIST_BASENAMES = frozenset({
    "result_card.json",
    "predictions.csv",
    "confusion_matrix.csv",
    "run_metadata.json",
    "search_result.json",
    "examples_binary.json",
    "examples_multiclass.json",
    "tfidf_config.json",
    # Ensemble artifacts persisted by notebook 43.
    "meta_classifier.joblib",
    "meta_classifier_meta.json",
    "agreement_matrix.csv",
    "fleiss_kappa.json",
    "contingency_table.csv",
})

BLACKLIST_BASENAMES = frozenset({
    "predictions_checkpoint.csv",
})

BLACKLIST_PATH_FRAGMENTS = (
    "/checkpoints/",
    "/model/model.safetensors",
    "/model/tokenizer.json",
    "/model/tokenizer_config.json",
    "/model/vocab.txt",
    "/model/special_tokens_map.json",
    "/model/config.json",
    "/model/training_args.bin",
    "/model/added_tokens.json",
    "/model/sentencepiece.bpe.model",
    "/model/spiece.model",
)

JOBLIB_FRAGMENT = "/model/tfidf_pipeline.joblib"


@dataclass
class ExtractStats:
    zip_name: str
    extracted_count: int = 0
    skipped_count: int = 0
    bytes_written: int = 0
    result_cards: int = 0
    predictions: int = 0
    expected_filtered: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class Report:
    processed_zips: list[str] = field(default_factory=list)
    deleted_zips: list[str] = field(default_factory=list)
    failed_zips: list[str] = field(default_factory=list)
    total_extracted: int = 0
    total_bytes: int = 0
    total_result_cards: int = 0
    total_predictions: int = 0


def is_safe_path(name: str) -> bool:
    """Reject zip-slip attempts and entries outside ``runs/``."""
    if not name or name.startswith("/") or "\\" in name:
        return False
    if ".." in Path(name).parts:
        return False
    return name.startswith(RUNS_PREFIX)


def should_extract(name: str, *, keep_tfidf_joblib: bool) -> bool:
    """Decide if a zip entry should be extracted based on whitelist/blacklist."""
    if name.endswith("/"):
        return False
    if not is_safe_path(name):
        return False

    basename = name.rsplit("/", 1)[-1]

    if basename in BLACKLIST_BASENAMES:
        return False
    for fragment in BLACKLIST_PATH_FRAGMENTS:
        if fragment in name:
            return False

    if JOBLIB_FRAGMENT in name:
        return keep_tfidf_joblib

    if basename in WHITELIST_BASENAMES:
        return True

    return False


def _check_disk_space(dest_root: Path, min_free_gb: float) -> None:
    probe = dest_root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Espaco em disco insuficiente: {free_gb:.2f} GB livres "
            f"(minimo exigido: {min_free_gb:.2f} GB). Abortando.",
        )


def extract_zip_streaming(
    zip_path: Path,
    dest_root: Path,
    *,
    keep_tfidf_joblib: bool,
    min_free_gb: float,
    dry_run: bool,
) -> ExtractStats:
    """Extract whitelisted entries from a single zip in streaming mode."""
    stats = ExtractStats(zip_name=zip_path.name)

    if not dry_run:
        _check_disk_space(dest_root, min_free_gb)

    with zipfile.ZipFile(zip_path, "r") as zf:
        if not dry_run:
            bad = zf.testzip()
            if bad is not None:
                stats.errors.append(f"CRC invalido em {bad}")
                return stats

        for info in zf.infolist():
            name = info.filename
            if not should_extract(name, keep_tfidf_joblib=keep_tfidf_joblib):
                stats.skipped_count += 1
                continue

            stats.expected_filtered += 1
            basename = name.rsplit("/", 1)[-1]

            if dry_run:
                stats.extracted_count += 1
                stats.bytes_written += info.file_size
                if basename == "result_card.json":
                    stats.result_cards += 1
                elif basename == "predictions.csv":
                    stats.predictions += 1
                continue

            dst = dest_root / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(info) as src, dst.open("wb") as out:
                    shutil.copyfileobj(src, out, length=COPY_CHUNK)
            except OSError as exc:
                dst.unlink(missing_ok=True)
                if exc.errno == errno.ENOSPC:
                    raise RuntimeError(
                        f"Disco cheio ao extrair {name} de {zip_path.name}. "
                        f"Espaco livre: {shutil.disk_usage(dest_root).free / 1e9:.2f} GB.",
                    ) from exc
                stats.errors.append(f"{name}: {exc}")
                continue
            except Exception as exc:
                dst.unlink(missing_ok=True)
                stats.errors.append(f"{name}: {exc}")
                continue

            stats.extracted_count += 1
            stats.bytes_written += info.file_size
            if basename == "result_card.json":
                stats.result_cards += 1
            elif basename == "predictions.csv":
                stats.predictions += 1

    return stats


def validate_extraction(stats: ExtractStats) -> bool:
    """Pass if no errors and extracted count matches expected filtered count."""
    if stats.errors:
        return False
    return stats.extracted_count == stats.expected_filtered


def _format_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def process_all(
    zips_dir: Path,
    pattern: str,
    dest_root: Path,
    *,
    keep_tfidf_joblib: bool,
    delete_after: bool,
    dry_run: bool,
    min_free_gb: float,
    continue_on_error: bool,
) -> Report:
    report = Report()
    zip_paths = sorted(zips_dir.glob(pattern))

    if not zip_paths:
        print(f"Nenhum zip encontrado em {zips_dir} com padrao '{pattern}'.")
        return report

    print(f"Encontrados {len(zip_paths)} zip(s) em {zips_dir}")
    print(f"Destino: {dest_root}")
    print(f"Modo: {'DRY-RUN' if dry_run else 'extracao'}"
          f" | keep_tfidf_joblib={keep_tfidf_joblib}"
          f" | delete_after={delete_after}")
    print("-" * 72)

    if not dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)

    for zip_path in zip_paths:
        try:
            stats = extract_zip_streaming(
                zip_path,
                dest_root,
                keep_tfidf_joblib=keep_tfidf_joblib,
                min_free_gb=min_free_gb,
                dry_run=dry_run,
            )
        except Exception as exc:
            print(f"  FALHA {zip_path.name}: {exc}")
            report.failed_zips.append(zip_path.name)
            if continue_on_error:
                continue
            raise

        ok = validate_extraction(stats)
        status = "OK" if ok else "FALHA"
        print(
            f"  {status:5s} {zip_path.name}: "
            f"extraidos={stats.extracted_count} "
            f"(cards={stats.result_cards}, preds={stats.predictions}) "
            f"ignorados={stats.skipped_count} "
            f"tamanho={_format_size(stats.bytes_written)}",
        )
        for err in stats.errors:
            print(f"      ERRO: {err}")

        if not ok:
            report.failed_zips.append(zip_path.name)
            if not continue_on_error:
                break
            continue

        report.processed_zips.append(zip_path.name)
        report.total_extracted += stats.extracted_count
        report.total_bytes += stats.bytes_written
        report.total_result_cards += stats.result_cards
        report.total_predictions += stats.predictions

        if delete_after and not dry_run:
            zip_path.unlink()
            report.deleted_zips.append(zip_path.name)
            print(f"      apagado {zip_path.name}")

    print("-" * 72)
    print(f"Zips processados com sucesso: {len(report.processed_zips)}")
    print(f"Zips apagados: {len(report.deleted_zips)}")
    print(f"Zips com falha: {len(report.failed_zips)}")
    print(f"Total extraido: {report.total_extracted} arquivos, "
          f"{_format_size(report.total_bytes)}")
    print(f"Total result_card.json: {report.total_result_cards}")
    print(f"Total predictions.csv:  {report.total_predictions}")

    if not dry_run:
        runs_dir = dest_root / "runs"
        if runs_dir.exists():
            run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir())
            complete = [
                d for d in run_dirs
                if (d / "result_card.json").exists() and (d / "predictions.csv").exists()
            ]
            partial = [d for d in run_dirs if d not in complete]
            print(f"Run dirs em artifacts/runs/: {len(run_dirs)}")
            print(f"  completos (card + preds): {len(complete)}")
            if partial:
                print(f"  incompletos: {len(partial)}")
                for d in partial[:10]:
                    have_card = (d / "result_card.json").exists()
                    have_pred = (d / "predictions.csv").exists()
                    flags = []
                    if not have_card:
                        flags.append("sem card")
                    if not have_pred:
                        flags.append("sem preds")
                    print(f"    - {d.name} ({', '.join(flags)})")
                if len(partial) > 10:
                    print(f"    ... e mais {len(partial) - 10}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zips-dir",
        type=Path,
        default=Path.home() / "Downloads",
        help="Diretorio com os arquivos zip (default: ~/Downloads).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="runs-20260427T103250Z-3-*.zip",
        help="Glob pattern para os zips.",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Raiz onde 'runs/...' sera escrito (default: artifacts/).",
    )
    parser.add_argument(
        "--keep-tfidf-joblib",
        action="store_true",
        help="Manter tfidf_pipeline.joblib (~400 MB total). Padrao: descartar.",
    )
    parser.add_argument(
        "--delete-after",
        action="store_true",
        help="Apagar cada zip apos extracao validada com sucesso.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Listar o que seria extraido sem escrever em disco.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=5.0,
        help="Espaco minimo livre exigido antes de cada zip (default: 5 GB).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continuar para o proximo zip mesmo se um falhar.",
    )
    args = parser.parse_args()

    if not args.zips_dir.exists():
        print(f"ERRO: diretorio de zips nao existe: {args.zips_dir}")
        return 1

    report = process_all(
        zips_dir=args.zips_dir,
        pattern=args.pattern,
        dest_root=args.dest_root,
        keep_tfidf_joblib=args.keep_tfidf_joblib,
        delete_after=args.delete_after,
        dry_run=args.dry_run,
        min_free_gb=args.min_free_gb,
        continue_on_error=args.continue_on_error,
    )

    return 0 if not report.failed_zips else 2


if __name__ == "__main__":
    sys.exit(main())
