#!/usr/bin/env python3
"""Avaliacao out-of-domain (OOD) sobre o PortugueseNewsDataset (Klaifer/WikiNoticias).

Terceira base externa, complementar a ``evaluate_fake_br.py`` e
``evaluate_fake_recogna.py``. Diferentemente das duas anteriores, o
PortugueseNewsDataset tem **label de topico explicita** alinhada ao construto
`mercado` da Folha: a categoria *"Economia e negocios"* e diretamente
comparavel — nao precisa do mapeamento-proxy via portal (UOL) ou veracidade
(Fake.Br) que enfraqueceu a leitura nos cenarios anteriores.

Fonte: WikiNoticias (PT-BR collaborative news), dump Wikimedia 2022-04-01,
reconstrucao via scripts do repo https://github.com/Klaifer/PortugueseNewsDataset
(PLOS ONE 2024). 5 classes (Crime+Direito+Justica, Desporto, Economia e
negocios, Politica, Saude), 9.135 docs apos filtro single-category.

Niveis (apenas binario nesta versao, por escolha de escopo):

1. **Binario OOD no corpus completo** (default) — positivo = categoria
   *"Economia e negocios"* (1.151/9.135 = 12.60%). Prevalencia quase identica
   a Folha train (12.5% mercado) — primeiro cenario OOD em que a base-rate
   nao precisa ser corrigida via re-amostragem. Comparable to nivel 1 dos
   outros dois scripts.

Observacoes:
- Texto de entrada: campo ``text`` ja vem como ``title + '. ' + body``
  (concatenado por ``train_split.py`` do repo upstream). Sem
  pre-processamento adicional (raw + tokenizer do HF, igual ao treino Folha).
- Partition default: ``full`` (corpus inteiro reconstruido). Sem leakage
  porque nenhum modelo treinou em WikiNoticias. Para reproduzir o split
  canonico do paper Klaifer 2024, use ``--partition test`` (n=914).

Cards usam ``regime='test_set'`` + ``config.domain='portuguese_news_wikinotices'``
para que o notebook de agregacao os separe dos cards in-domain e OOD anteriores.

Uso:

    uv run python scripts/evaluate_portuguese_news.py \\
        --model-dir /content/drive/MyDrive/economy-classifier/models/bertimbau_binary \\
        --model-id bert_bertimbau

    uv run python scripts/evaluate_portuguese_news.py \\
        --model-dir /content/drive/MyDrive/economy-classifier/models/tfidf_logreg_binary \\
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

DEFAULT_PORTUGUESE_NEWS_ROOT = Path(
    "/home/diacrono/Documentos/repositorios/economy-classifier/data/portuguese_news_wikinotices"
)
DEFAULT_HARDWARE = "local-CPU"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 500  # uniforme com o treino in-domain (NB 21); antes 128
POSITIVE_CATEGORY = "Economia e negócios"
POSITIVE_BINARY = 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_portuguese_news(root: Path, partition: str = "full") -> pd.DataFrame:
    """Carrega o PortugueseNewsDataset reconstruido (Klaifer 2024).

    Args:
        root: diretorio com ``wikinews_categories.json`` (full corpus),
            ``wikinews_train.json`` e ``wikinews_test.json`` (split canonico).
        partition: ``full`` (default), ``train`` ou ``test``.

    Retorna colunas: ``document_id, text, category, label_index, y_true_binary``.

    ``y_true_binary``: 1 se categoria == 'Economia e negocios', 0 caso contrario.
    """
    if partition == "full":
        path = root / "wikinews_categories.json"
        with open(path, "r") as f:
            raw = json.load(f)
        df = pd.DataFrame([
            {
                "document_id": str(d["pageid"]),
                "text": ". ".join([d["title"], d["body"]]).strip(),
                "category": d["category"],
            }
            for d in raw
        ])
    elif partition in ("train", "test"):
        path = root / f"wikinews_{partition}.json"
        with open(path, "r") as f:
            payload = json.load(f)
        labels = payload["labels"]
        df = pd.DataFrame([
            {
                "document_id": str(d["pageid"]),
                "text": d["text"].strip(),
                "category": labels[d["label"]],
                "label_index": int(d["label"]),
            }
            for d in payload["data"]
        ])
    else:
        raise ValueError(f"partition deve ser 'full', 'train' ou 'test'; got {partition!r}")

    df = df[df["text"].str.len() > 0].copy()
    df["y_true_binary"] = (df["category"] == POSITIVE_CATEGORY).astype(int)
    return df.sort_values("document_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model loading / inference (mesmas convencoes de evaluate_fake_recogna.py)
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
# Nivel 1: binario OOD no corpus completo
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
    out_dir = output_root / f"{model_id}_binary_portuguese_news_wikinotices_ood"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({
        "index": df.index,
        "document_id": df["document_id"].values,
        "category": df["category"].values,
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
            "domain": "portuguese_news_wikinotices",
            "evaluation_type": "out_of_domain",
            "level": "1_binary_full_corpus",
            "partition": partition,
            "max_length": max_length,
            "decision_threshold": 0.5,
            "positive_class_mapping": (
                "mercado (Folha) <- category == 'Economia e negócios' (WikiNoticias)"
            ),
            "input_text_strategy": "title + '. ' + body (raw, conforme train_split.py do repo Klaifer)",
            "source_dataset": "Klaifer/PortugueseNewsDataset (PLOS ONE 2024)",
            "source_dump": "ptwikinews-20220401-pages-meta-current.xml",
        },
        n_train_samples=None,
        n_eval_samples=n,
        predictions_path=str(out_dir / "predictions.csv"),
        notes=(
            "Avaliacao OOD no PortugueseNewsDataset (WikiNoticias). Positivo = "
            f"categoria '{POSITIVE_CATEGORY}' (1.151/9.135 = 12.60% no full "
            "corpus, prevalencia quase identica ao treino Folha de 12.5% "
            "mercado). Diferentemente de Fake.Br e FakeRecogna, a label aqui "
            "e topico-explicito (nao proxy via veracidade ou portal), entao "
            "interpretacao do F1 e direta. Shift de dominio principal: registro "
            "encyclopedico/colaborativo (WikiNoticias) vs jornalismo comercial "
            "(Folha). Texto de entrada e title + '. ' + body, igual ao paper "
            "Klaifer 2024."
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
            "PortugueseNewsDataset (WikiNoticias, categoria 'Economia e negocios')."
        ),
    )
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Diretorio com pesos HF (BERT) ou tfidf_pipeline.joblib (TF-IDF).")
    parser.add_argument("--model-id", type=str, required=True,
                        help="Identificador do card (ex.: bert_bertimbau, tfidf_logreg).")
    parser.add_argument("--portuguese-news-root", type=Path,
                        default=DEFAULT_PORTUGUESE_NEWS_ROOT,
                        help="Raiz do PortugueseNewsDataset reconstruido.")
    parser.add_argument("--partition", choices=["full", "train", "test"],
                        default="full",
                        help="Particao a avaliar. 'full' (9135 docs) por default; "
                             "'test' (914) reproduz o split canonico do paper.")
    parser.add_argument("--output-root", type=Path, default=RUNS_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_dir.exists():
        print(f"ERRO: --model-dir nao encontrado: {args.model_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.portuguese_news_root.exists():
        print(
            f"ERRO: --portuguese-news-root nao encontrado: {args.portuguese_news_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Carregando PortugueseNewsDataset de {args.portuguese_news_root} "
        f"(partition={args.partition})...",
        file=sys.stderr,
    )
    df = load_portuguese_news(args.portuguese_news_root, partition=args.partition)
    n_pos = int((df["y_true_binary"] == 1).sum())
    print(
        f"  {len(df)} docs validos | 'Economia e negocios'={n_pos} "
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
