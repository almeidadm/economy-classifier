#!/usr/bin/env python3
"""Avaliacao out-of-domain (OOD) sobre o RecognaSumm (Paiola et al., PROPOR 2024).

Quarta base externa, complementar a:
- ``evaluate_fake_br.py`` (Fake.Br, fake news, label proxy via veracidade)
- ``evaluate_fake_recogna.py`` (FakeRecogna, fake news, label proxy via URL)
- ``evaluate_portuguese_news.py`` (WikiNoticias, topico explicito, registro enciclopedico)

Diferencial do RecognaSumm: corpus de **noticias jornalisticas** (multi-veiculo:
G1, CNN Brasil, UOL, Folha entre outros) com **categoria editorial explicita**
incluindo *Economia*. Comparado ao PortugueseNewsDataset (WikiNoticias), aqui
o registro e jornalistico-comercial — proximo do treino Folha — mas com
multi-publisher (controla efeito unico-veiculo da Folha).

Fonte: https://huggingface.co/datasets/recogna-nlp/recognasumm
Paper: Paiola et al., "RecognaSumm: A Novel Brazilian Summarization Dataset",
PROPOR 2024. License MIT.

Niveis (apenas binario nesta versao, mesmo escopo dos outros 3 OODs):

1. **Binario OOD na particao test** (default) — positivo = ``Categoria ==
   "Economia"`` (2.515/27.055 = 9.30%). Prevalencia proxima a Folha train
   (12.5% mercado). Comparable to nivel 1 dos outros scripts.

Observacoes:
- Texto de entrada: ``Titulo + ". " + Noticia`` (mesma estrategia que
  ``evaluate_portuguese_news.py``). ``Subtitulo`` ignorado deliberadamente
  para nao introduzir uma 3a estrategia de concat entre os OOD scripts.
- Particao default: ``test`` (27.055 docs). Sem leakage porque nenhum modelo
  treinou em RecognaSumm. As particoes train/validation existem no upstream
  mas sao desnecessarias para OOD (consumo extra de disco e GPU).
- Categoria tem ruido lexical no upstream (e.g., 'Saude'+'saude',
  'Politica'+'politica'). Para binario isso nao afeta — so a string exata
  'Economia' (n=2.515) entra como positivo. Categorias minoritarias (<2%)
  ficam todas em negativo, igual ao tratamento original.

Cards usam ``regime='test_set'`` + ``config.domain='recognasumm_propor2024'``
para que o notebook de agregacao os separe dos cards in-domain e dos
outros 3 OODs.

Uso:

    uv run python scripts/evaluate_recognasumm.py \\
        --model-dir /content/drive/MyDrive/economy-classifier/runs/bert_bertimbau_binary_test_set/model \\
        --model-id bert_bertimbau

    uv run python scripts/evaluate_recognasumm.py \\
        --model-dir /content/drive/MyDrive/economy-classifier/runs/tfidf_logreg_binary_test_set/model \\
        --model-id tfidf_logreg --partition test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from economy_classifier.evaluation import (  # noqa: E402
    compute_binary_metrics,
    compute_brier_score,
    compute_ece,
    compute_roc_auc,
)
from economy_classifier.project import (  # noqa: E402
    RUNS_DIR,
    build_result_card,
    compute_artifact_size_mb,
    persist_result_card,
)

DEFAULT_RECOGNASUMM_ROOT = Path(
    "/home/diacrono/Documentos/repositorios/economy-classifier/data/recognasumm"
)
DEFAULT_HARDWARE = "local-CPU"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 128
POSITIVE_CATEGORY = "Economia"
POSITIVE_BINARY = 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _build_input_text(titulo: Any, noticia: Any) -> str:
    """Concatena Titulo + '. ' + Noticia, ambos texto cru."""
    t = str(titulo).strip() if pd.notna(titulo) else ""
    n = str(noticia).strip() if pd.notna(noticia) else ""
    if n:
        return f"{t}. {n}".strip()
    return t


def load_recognasumm(root: Path, partition: str = "test") -> pd.DataFrame:
    """Carrega o RecognaSumm de uma das particoes JSONL.

    Args:
        root: diretorio com ``test.jsonl`` (e opcionalmente train/validation).
        partition: ``test`` (default, 27.055 docs), ``train``, ``validation``
            ou ``full`` (concatena train+validation+test, ~150k docs).

    Retorna colunas: ``document_id, text, category, autor, url, sumario,
    y_true_binary``.
    """
    parts = ["train", "validation", "test"] if partition == "full" else [partition]
    missing = [p for p in parts if not (root / f"{p}.jsonl").exists()]
    if missing:
        urls = "\n".join(
            f"  https://huggingface.co/datasets/recogna-nlp/recognasumm/resolve/main/{p}.jsonl"
            for p in missing
        )
        raise FileNotFoundError(
            f"Arquivo(s) ausente(s) em {root}: {missing}. Baixe via:\n{urls}"
        )

    rows: list[dict[str, Any]] = []
    for part in parts:
        path = root / f"{part}.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                rows.append({
                    "document_id": f"{part}:{obj.get('index', '')}" if partition == "full" else str(obj.get("index", "")),
                    "titulo": obj.get("Titulo", "") or "",
                    "subtitulo": obj.get("Subtitulo", "") or "",
                    "noticia": obj.get("Noticia", "") or "",
                    "category": (obj.get("Categoria", "") or "").strip(),
                    "autor": obj.get("Autor_corrigido") or obj.get("Autor", "") or "",
                    "url": obj.get("URL", "") or "",
                    "sumario": obj.get("Sumario", "") or "",
                })
    df = pd.DataFrame(rows)
    df["text"] = [_build_input_text(t, n) for t, n in zip(df["titulo"], df["noticia"], strict=True)]
    df = df[df["text"].str.len() > 0].copy()
    df["y_true_binary"] = (df["category"] == POSITIVE_CATEGORY).astype(int)
    return df.sort_values("document_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model loading / inference (mesmas convencoes dos outros 3 OOD scripts)
# ---------------------------------------------------------------------------
def detect_model_type(model_dir: Path) -> str:
    if (model_dir / "tfidf_pipeline.joblib").exists():
        return "tfidf"
    bert_markers = ("config.json", "pytorch_model.bin", "model.safetensors")
    if any((model_dir / fname).exists() for fname in bert_markers):
        return "bert"
    raise FileNotFoundError(
        f"Tipo de modelo nao identificado em {model_dir}. Esperado "
        "tfidf_pipeline.joblib (TF-IDF) ou config.json + pesos HF (BERT)."
    )


def predict_bert(
    texts: list[str], model_dir: Path,
    *, batch_size: int, max_length: int,
) -> tuple[np.ndarray, dict[int, str], float, dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    n_params = int(sum(p.numel() for p in model.parameters()))

    all_probs: list[np.ndarray] = []
    start = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", truncation=True,
            padding=True, max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    inference_seconds = time.perf_counter() - start
    info = {"n_parameters": n_params, "hardware": f"local-{device.type}"}
    return np.concatenate(all_probs, axis=0), id2label, inference_seconds, info


def predict_tfidf(
    texts: list[str], model_dir: Path,
) -> tuple[np.ndarray, list[Any], float, dict[str, Any]]:
    import joblib

    pipeline = joblib.load(model_dir / "tfidf_pipeline.joblib")
    start = time.perf_counter()
    probs = pipeline.predict_proba(texts)
    inference_seconds = time.perf_counter() - start
    info = {"n_parameters": None, "hardware": DEFAULT_HARDWARE}
    return probs, list(pipeline.classes_), inference_seconds, info


def _binary_positive_index(classes: Any, model_type: str) -> int:
    if model_type == "bert":
        label_to_idx = {v: k for k, v in classes.items()}
        if "mercado" in label_to_idx:
            return label_to_idx["mercado"]
        return label_to_idx.get("1", 1)
    classes_list = list(classes)
    if POSITIVE_BINARY in classes_list:
        return classes_list.index(POSITIVE_BINARY)
    return 1


def _build_cost(
    *, inference_seconds: float, n_samples: int,
    model_size_mb: float, n_parameters: int | None, hardware: str,
) -> dict[str, Any]:
    throughput = (
        round(n_samples / inference_seconds, 2) if inference_seconds > 0 else None
    )
    return {
        "train_seconds_mean": 0.0,
        "train_seconds_std": 0.0,
        "inference_seconds_mean": round(inference_seconds, 4),
        "inference_seconds_std": 0.0,
        "throughput_samples_per_second": throughput,
        "model_size_mb": model_size_mb,
        "n_parameters": n_parameters,
        "hardware": hardware,
    }


# ---------------------------------------------------------------------------
# Nivel 1: binario OOD na particao informada
# ---------------------------------------------------------------------------
def evaluate_level1_binary(
    *, df: pd.DataFrame, probs: np.ndarray, classes: Any,
    model_id: str, model_type: str, output_root: Path,
    inference_seconds: float, model_size_mb: float,
    max_length: int, n_parameters: int | None, hardware: str,
    partition: str,
) -> Path:
    pos_idx = _binary_positive_index(classes, model_type)
    y_score = probs[:, pos_idx]
    y_pred = (y_score >= 0.5).astype(int)
    y_true = df["y_true_binary"].to_numpy()

    metrics = compute_binary_metrics(y_true, y_pred)
    metrics["auc_roc"] = round(compute_roc_auc(y_true, y_score), 4)
    metrics["brier"] = compute_brier_score(y_true, y_score)
    metrics["ece"] = compute_ece(y_true, y_score)
    metrics["positive_prevalence"] = round(float(y_true.mean()), 4)

    n = len(df)
    out_dir = output_root / f"{model_id}_binary_recognasumm_ood"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({
        "index": df.index,
        "document_id": df["document_id"].values,
        "category": df["category"].values,
        "autor": df["autor"].values,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": np.round(y_score, 4),
        "method": model_id,
    })
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    card = build_result_card(
        model_id=model_id,
        task="binary",
        regime="test_set",
        metrics=metrics,
        cost=_build_cost(
            inference_seconds=inference_seconds, n_samples=n,
            model_size_mb=model_size_mb, n_parameters=n_parameters,
            hardware=hardware,
        ),
        config={
            "domain": "recognasumm_propor2024",
            "evaluation_type": "out_of_domain",
            "level": "1_binary_full_corpus",
            "partition": partition,
            "max_length": max_length,
            "decision_threshold": 0.5,
            "positive_class_mapping": (
                "mercado (Folha) <- Categoria == 'Economia' (RecognaSumm)"
            ),
            "input_text_strategy": "Titulo + '. ' + Noticia (raw)",
            "source_dataset": "recogna-nlp/recognasumm (PROPOR 2024)",
            "source_dataset_url": "https://huggingface.co/datasets/recogna-nlp/recognasumm",
        },
        n_train_samples=None,
        n_eval_samples=n,
        predictions_path=str(out_dir / "predictions.csv"),
        notes=(
            "Avaliacao OOD no RecognaSumm. Positivo = Categoria == 'Economia' "
            "(2.515/27.055 = 9.30% na particao test; ~9% no corpus full de "
            "135k). Corpus multi-veiculo (G1, CNN Brasil, UOL, Folha e outros) "
            "com categoria editorial explicita. Diferentemente dos OODs "
            "anteriores: registro jornalistico-comercial (proximo do treino "
            "Folha) + multi-publisher (controla efeito unico-veiculo). "
            "Categoria tem ruido lexical no upstream (e.g., Saude vs saude); "
            "para binario apenas a string exata 'Economia' (n=2.515) entra "
            "como positivo. Texto de entrada: Titulo + '. ' + Noticia."
        ),
        hyperparameter_search=None,
    )
    persist_result_card(card, out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Avaliacao OOD do classificador binario `mercado` da Folha no "
            "RecognaSumm (PROPOR 2024, categoria 'Economia')."
        ),
    )
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Diretorio com pesos HF (BERT) ou tfidf_pipeline.joblib (TF-IDF).")
    parser.add_argument("--model-id", type=str, required=True,
                        help="Identificador do card (ex.: bert_bertimbau, tfidf_logreg).")
    parser.add_argument("--recognasumm-root", type=Path,
                        default=DEFAULT_RECOGNASUMM_ROOT,
                        help="Raiz do RecognaSumm com test.jsonl (e opcionais train/validation).")
    parser.add_argument("--partition", choices=["test", "train", "validation", "full"],
                        default="test",
                        help="Particao a avaliar (default: test, 27.055 docs; "
                             "full concatena train+validation+test).")
    parser.add_argument("--output-root", type=Path, default=RUNS_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_dir.exists():
        print(f"ERRO: --model-dir nao encontrado: {args.model_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.recognasumm_root.exists():
        print(
            f"ERRO: --recognasumm-root nao encontrado: {args.recognasumm_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Carregando RecognaSumm de {args.recognasumm_root} "
        f"(partition={args.partition})...",
        file=sys.stderr,
    )
    df = load_recognasumm(args.recognasumm_root, partition=args.partition)
    n_pos = int((df["y_true_binary"] == 1).sum())
    print(
        f"  {len(df)} docs validos | 'Economia'={n_pos} "
        f"(prevalencia={n_pos/len(df):.4f})",
        file=sys.stderr,
    )

    model_type = detect_model_type(args.model_dir)
    model_size_mb = round(compute_artifact_size_mb(args.model_dir), 3)
    print(f"Modelo: type={model_type}, size={model_size_mb} MB", file=sys.stderr)

    texts = df["text"].tolist()
    if model_type == "bert":
        probs, classes, inf_s, info = predict_bert(
            texts, args.model_dir,
            batch_size=args.batch_size, max_length=args.max_length,
        )
    else:
        probs, classes, inf_s, info = predict_tfidf(texts, args.model_dir)
    print(
        f"Inferencia: {len(texts)} amostras em {inf_s:.2f}s "
        f"({probs.shape[1]} classes de saida)",
        file=sys.stderr,
    )

    n_classes = probs.shape[1]
    if n_classes != 2:
        print(
            f"ERRO: esperado modelo binario (2 classes); recebido {n_classes}. "
            "Este script avalia somente a tarefa binaria.",
            file=sys.stderr,
        )
        sys.exit(1)

    args.output_root.mkdir(parents=True, exist_ok=True)
    out1 = evaluate_level1_binary(
        df=df, probs=probs, classes=classes,
        model_id=args.model_id, model_type=model_type,
        output_root=args.output_root, inference_seconds=inf_s,
        model_size_mb=model_size_mb, max_length=args.max_length,
        n_parameters=info["n_parameters"], hardware=info["hardware"],
        partition=args.partition,
    )
    print(f"  nivel 1 (binario OOD) -> {out1}", file=sys.stderr)


if __name__ == "__main__":
    main()
