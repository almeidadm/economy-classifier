#!/usr/bin/env python3
"""Avaliacao out-of-domain (OOD) sobre o corpus Fake.Br.

Executa os tres niveis de leitura definidos no plano metodologico:

1. **Binario, corpus completo (7200 docs)** — positivo = `metadata_category=='economia'`,
   negativo = qualquer outra categoria. Recria a tarefa `mercado vs outros` em
   dominio externo, com prevalencia ~0.6% (44/7200) — mais agressiva que os
   12.5% da Folha.
2. **Multiclasse mapeado** — projeta as categorias do Fake.Br no esquema
   `7+other` via ``FAKE_BR_TO_TOPK``. Classes `colunas/esporte/mundo` nao tem
   analogo no Fake.Br: aparecem com suporte zero. Reporta `macro_f1` (schema
   completo, para paridade com o card da Folha) e `macro_f1_present_only`
   (so labels com suporte real).
3. **Subgrupo por veracidade (nivel binario)** — emite cards separados para
   `veracity=='fake'` e `veracity=='true'`. Diferencas substanciais entre
   esses subgrupos sao evidencia de que a retorica de desinformacao descola
   das pistas lexicas vistas no treino.

Cada nivel grava ``predictions.csv`` + ``result_card.json`` em
``<output-root>/<model_id>_<task>_fake_br_<level>/`` reutilizando o schema
de ``project.build_result_card``. Os cards usam ``regime='test_set'`` e
declaram o contexto OOD em ``config.domain`` e ``notes`` (a comparacao com
os cards in-domain da Folha precisa filtrar por ``config.domain``).

Limitacoes declaradas (refletidas em ``notes``):
- AUC-ROC e Brier/ECE sob domain shift nao sao apples-to-apples com os cards
  in-domain — interpretar como estimativa OOD, nao como nivel absoluto de
  calibracao.
- Truncacao em ``max_length=128`` (config BERT do projeto) descarta a cauda
  longa de muitos artigos do Fake.Br — manter para paridade com o treino.

Uso tipico (rodar uma vez por modelo, no Colab onde os pesos vivem):

    uv run python scripts/evaluate_fake_br.py \\
        --model-dir /content/drive/MyDrive/economy-classifier/models/bertimbau_binary \\
        --model-id bert_bertimbau --task binary

    uv run python scripts/evaluate_fake_br.py \\
        --model-dir /content/drive/MyDrive/economy-classifier/models/bertimbau_multi \\
        --model-id bert_bertimbau --task multiclass

    uv run python scripts/evaluate_fake_br.py \\
        --model-dir artifacts/runs/tfidf_logreg_binary_test_set/model \\
        --model-id tfidf_logreg --task binary
"""

from __future__ import annotations

import argparse
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
    compute_confusion_matrix,
    compute_ece,
    compute_multiclass_metrics,
    compute_roc_auc,
)
from economy_classifier.project import (  # noqa: E402
    RUNS_DIR,
    build_result_card,
    compute_artifact_size_mb,
    persist_result_card,
)

DEFAULT_FAKE_BR_ROOT = Path(
    "/home/diacrono/Documentos/repositorios/fn-dataset-eda/data/raw/fake_br"
)
DEFAULT_HARDWARE = "local-CPU"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 128
POSITIVE_BINARY = 1

# Mapping Fake.Br categories -> 7+other multiclass schema.
# - economia -> mercado e o pareamento alvo do estudo.
# - politica -> poder, tv_celebridades -> ilustrada, sociedade_cotidiano -> cotidiano
#   sao mapeamentos editoriais aproximados (rubricas equivalentes na Folha).
# - ciencia_tecnologia e religiao folham para 'outros' por falta de analogo
#   no esquema da Folha (nao existem secoes equivalentes no 7+other puro).
# - Classes ausentes no Fake.Br: colunas, esporte, mundo (suporte zero).
FAKE_BR_TO_TOPK: dict[str, str] = {
    "economia": "mercado",
    "politica": "poder",
    "tv_celebridades": "ilustrada",
    "sociedade_cotidiano": "cotidiano",
    "ciencia_tecnologia": "outros",
    "religiao": "outros",
}
TOPK_LABELS: list[str] = [
    "poder", "colunas", "mercado", "esporte",
    "mundo", "cotidiano", "ilustrada", "outros",
]


# ---------------------------------------------------------------------------
# Data loading (self-contained: nao depende do pacote fn_dataset_eda)
# ---------------------------------------------------------------------------
def load_fake_br(root: Path) -> pd.DataFrame:
    """Carrega Fake.Br/full_texts emparelhado com a categoria do metadata.

    Retorna colunas: ``document_id, veracity, text, fake_br_category,
    y_true_binary, y_true_multi``.
    """
    rows: list[dict[str, Any]] = []
    for veracity in ("fake", "true"):
        text_dir = root / "full_texts" / veracity
        meta_dir = root / "full_texts" / f"{veracity}-meta-information"
        if not text_dir.is_dir():
            raise FileNotFoundError(f"Diretorio de textos ausente: {text_dir}")
        if not meta_dir.is_dir():
            raise FileNotFoundError(f"Diretorio de metadata ausente: {meta_dir}")
        for text_file in sorted(text_dir.iterdir()):
            if not text_file.is_file() or text_file.suffix != ".txt":
                continue
            doc_id = text_file.stem
            meta_path = meta_dir / f"{doc_id}-meta.txt"
            if not meta_path.exists():
                continue
            with meta_path.open("r", encoding="utf-8") as handle:
                meta_lines = [line.strip() for line in handle.readlines()]
            # Linha 3 do metadata = categoria editorial (vide formato Fake.Br upstream).
            category = meta_lines[2] if len(meta_lines) >= 3 else ""
            rows.append({
                "document_id": doc_id,
                "veracity": veracity,
                "text": text_file.read_text(encoding="utf-8"),
                "fake_br_category": category,
                "y_true_binary": 1 if category == "economia" else 0,
                "y_true_multi": FAKE_BR_TO_TOPK.get(category, "outros"),
            })
    if not rows:
        raise RuntimeError(f"Nenhum documento carregado de {root}")
    return (
        pd.DataFrame(rows)
        .sort_values(["veracity", "document_id"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Model loading / inference
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
    texts: list[str],
    model_dir: Path,
    *,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, dict[int, str], float, dict[str, Any]]:
    """Retorna ``(probs[n, n_classes], id2label, inference_seconds, info)``."""
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
    texts: list[str],
    model_dir: Path,
) -> tuple[np.ndarray, list[Any], float, dict[str, Any]]:
    """Retorna ``(probs[n, n_classes], classes_, inference_seconds, info)``."""
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
    # tfidf: classes_ é uma lista (e.g. [0, 1])
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
    out_dir = output_root / f"{model_id}_binary_fake_br_ood"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({
        "index": df.index,
        "document_id": df["document_id"].values,
        "veracity": df["veracity"].values,
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
            "domain": "fake_br_full_texts",
            "evaluation_type": "out_of_domain",
            "level": "1_binary_full_corpus",
            "max_length": max_length,
            "decision_threshold": 0.5,
            "positive_class_mapping": "mercado (Folha) <- economia (Fake.Br)",
        },
        n_train_samples=None,
        n_eval_samples=n,
        predictions_path=str(out_dir / "predictions.csv"),
        notes=(
            "Avaliacao out-of-domain no Fake.Br/full_texts. Positivo = "
            "metadata_category=='economia' (44/7200, ~0.6% — mais agressivo "
            "que os 12.5% da Folha). Modelo treinado em Folha (mercado vs "
            "outros) e aplicado sem fine-tuning. Brier/ECE refletem "
            "calibracao OOD — quedas vs card in-domain (test_set Folha) sao "
            "esperadas e nao indicam bug. AUC-ROC depende de y_score "
            "calibrado: para LLMs deterministicos, marcar N/A."
        ),
        hyperparameter_search=None,
    )
    persist_result_card(card, out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Nivel 2: multiclasse OOD com mapeamento de categorias
# ---------------------------------------------------------------------------
def evaluate_level2_multiclass(
    *, df: pd.DataFrame, probs: np.ndarray, classes: Any,
    model_id: str, model_type: str, output_root: Path,
    inference_seconds: float, model_size_mb: float,
    max_length: int, n_parameters: int | None, hardware: str,
) -> Path:
    if model_type == "bert":
        idx_to_label = dict(classes)
    else:
        idx_to_label = {i: c for i, c in enumerate(classes)}

    pred_idx = probs.argmax(axis=1)
    y_pred = np.array([idx_to_label[int(i)] for i in pred_idx])
    y_true = df["y_true_multi"].to_numpy()

    metrics_full = compute_multiclass_metrics(y_true, y_pred, labels=TOPK_LABELS)
    present_labels = sorted(set(y_true.tolist()))
    metrics_present = compute_multiclass_metrics(y_true, y_pred, labels=present_labels)
    metrics: dict[str, Any] = {
        "macro_f1": metrics_full["macro_f1"],
        "weighted_f1": metrics_full["weighted_f1"],
        "accuracy": metrics_full["accuracy"],
        "per_class_f1": metrics_full["per_class_f1"],
        "macro_f1_present_only": metrics_present["macro_f1"],
        "present_labels": present_labels,
        "label_distribution": {
            label: int((y_true == label).sum()) for label in TOPK_LABELS
        },
    }

    n = len(df)
    out_dir = output_root / f"{model_id}_multiclass_fake_br_ood"
    out_dir.mkdir(parents=True, exist_ok=True)

    proba_cols = {
        f"y_proba_{idx_to_label[i]}": np.round(probs[:, i], 4)
        for i in range(probs.shape[1])
    }
    pred_df = pd.DataFrame({
        "index": df.index,
        "document_id": df["document_id"].values,
        "veracity": df["veracity"].values,
        "fake_br_category": df["fake_br_category"].values,
        "y_true": y_true,
        "y_pred": y_pred,
        "method": model_id,
        **proba_cols,
    })
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    cm = compute_confusion_matrix(y_true, y_pred, labels=TOPK_LABELS, normalize="true")
    cm.to_csv(out_dir / "confusion_matrix.csv")

    card = build_result_card(
        model_id=model_id,
        task="multiclass",
        regime="test_set",
        metrics=metrics,
        cost=_build_cost(
            inference_seconds=inference_seconds, n_samples=n,
            model_size_mb=model_size_mb, n_parameters=n_parameters,
            hardware=hardware,
        ),
        config={
            "domain": "fake_br_full_texts",
            "evaluation_type": "out_of_domain",
            "level": "2_multiclass_mapped",
            "max_length": max_length,
            "label_mapping": FAKE_BR_TO_TOPK,
            "label_schema": TOPK_LABELS,
            "absent_classes": [c for c in TOPK_LABELS if c not in present_labels],
        },
        n_train_samples=None,
        n_eval_samples=n,
        predictions_path=str(out_dir / "predictions.csv"),
        notes=(
            "Avaliacao multiclasse OOD no Fake.Br/full_texts. Mapeamento "
            "Fake.Br -> 7+other em config.label_mapping. Classes "
            "colunas/esporte/mundo nao aparecem no Fake.Br (suporte zero): "
            "macro_f1 (schema completo) inclui essas classes com F1=0; "
            "macro_f1_present_only e o macro honesto sobre labels com "
            "suporte real. ciencia_tecnologia e religiao folham para "
            "'outros' por falta de analogo editorial — predicoes nessas "
            "amostras nao devem ser interpretadas como erro de classe."
        ),
        hyperparameter_search=None,
    )
    persist_result_card(card, out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Nivel 3: subgrupo por veracidade (binario apenas)
# ---------------------------------------------------------------------------
def evaluate_level3_subgroup(
    *, df: pd.DataFrame, probs: np.ndarray, classes: Any,
    model_id: str, model_type: str, output_root: Path,
    inference_seconds: float, model_size_mb: float,
    max_length: int, n_parameters: int | None, hardware: str,
) -> list[Path]:
    pos_idx = _binary_positive_index(classes, model_type)
    y_score_all = probs[:, pos_idx]
    y_pred_all = (y_score_all >= 0.5).astype(int)
    y_true_all = df["y_true_binary"].to_numpy()

    out_dirs: list[Path] = []
    for veracity in ("fake", "true"):
        mask = (df["veracity"] == veracity).to_numpy()
        n_sub = int(mask.sum())
        if n_sub == 0:
            continue
        y_true = y_true_all[mask]
        y_pred = y_pred_all[mask]
        y_score = y_score_all[mask]

        metrics = compute_binary_metrics(y_true, y_pred)
        metrics["auc_roc"] = (
            round(compute_roc_auc(y_true, y_score), 4)
            if len(set(y_true.tolist())) == 2 else None
        )
        metrics["brier"] = compute_brier_score(y_true, y_score)
        metrics["ece"] = compute_ece(y_true, y_score)
        metrics["positive_prevalence"] = round(float(y_true.mean()), 4)

        out_dir = output_root / f"{model_id}_binary_fake_br_subgroup_{veracity}"
        out_dir.mkdir(parents=True, exist_ok=True)

        pred_df = pd.DataFrame({
            "index": df.index[mask],
            "document_id": df.loc[mask, "document_id"].values,
            "veracity": veracity,
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
            # Custo e compartilhado: o run de inferencia ocorre uma vez sobre os
            # 7200 documentos. Reportamos proporcional ao subset para preservar
            # ordens de grandeza, mas o reader deve combinar os dois subgrupos.
            cost=_build_cost(
                inference_seconds=inference_seconds * (n_sub / len(df)),
                n_samples=n_sub, model_size_mb=model_size_mb,
                n_parameters=n_parameters, hardware=hardware,
            ),
            config={
                "domain": "fake_br_full_texts",
                "evaluation_type": "out_of_domain_subgroup",
                "level": f"3_subgroup_{veracity}",
                "max_length": max_length,
                "decision_threshold": 0.5,
                "subgroup": veracity,
            },
            n_train_samples=None,
            n_eval_samples=n_sub,
            predictions_path=str(out_dir / "predictions.csv"),
            notes=(
                f"Subgrupo Fake.Br restrito a veracity=='{veracity}'. "
                "Comparar com o card do outro subgrupo para testar se o "
                "classificador `mercado` degrada de forma diferente em "
                "desinformacao vs texto verificado. Custo de inferencia "
                "reportado proporcional ao tamanho do subset; o run real "
                "cobre os 7200 documentos. AUC pode ser None caso o "
                "subgrupo colapse para uma unica classe."
            ),
            hyperparameter_search=None,
        )
        persist_result_card(card, out_dir)
        out_dirs.append(out_dir)
    return out_dirs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avaliacao OOD do classificador de Folha no corpus Fake.Br.",
    )
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Diretorio com pesos HF (BERT) ou tfidf_pipeline.joblib (TF-IDF).")
    parser.add_argument("--model-id", type=str, required=True,
                        help="Identificador do card (ex.: bert_bertimbau, tfidf_logreg).")
    parser.add_argument("--task", choices=["binary", "multiclass"], required=True,
                        help="Tarefa que o modelo executa (deve casar com num_labels).")
    parser.add_argument("--fake-br-root", type=Path, default=DEFAULT_FAKE_BR_ROOT,
                        help="Raiz do Fake.Br (contem full_texts/{fake,true,*-meta-information}).")
    parser.add_argument("--output-root", type=Path, default=RUNS_DIR,
                        help="Onde gravar os cards (default: artifacts/runs).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                        help="Truncacao do tokenizer BERT (default casa com o treino).")
    parser.add_argument("--skip-subgroup", action="store_true",
                        help="Pula o nivel 3 (subgrupos por veracidade).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_dir.exists():
        print(f"ERRO: --model-dir nao encontrado: {args.model_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.fake_br_root.exists():
        print(f"ERRO: --fake-br-root nao encontrado: {args.fake_br_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Carregando Fake.Br de {args.fake_br_root}...", file=sys.stderr)
    df = load_fake_br(args.fake_br_root)
    n_pos = int((df["y_true_binary"] == 1).sum())
    cats = sorted(df["fake_br_category"].unique().tolist())
    print(f"  {len(df)} docs | economia={n_pos} | categorias={cats}", file=sys.stderr)

    model_type = detect_model_type(args.model_dir)
    model_size_mb = round(compute_artifact_size_mb(args.model_dir), 3)
    print(f"Modelo: type={model_type}, size={model_size_mb} MB", file=sys.stderr)

    texts = df["text"].fillna("").tolist()
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
    if args.task == "binary" and n_classes != 2:
        print(
            f"ERRO: --task=binary mas o modelo emite {n_classes} classes. "
            "Verifique se o checkpoint correto foi passado.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.task == "multiclass" and n_classes != len(TOPK_LABELS):
        print(
            f"ERRO: --task=multiclass mas o modelo emite {n_classes} classes; "
            f"esperado {len(TOPK_LABELS)} (esquema 7+other).",
            file=sys.stderr,
        )
        sys.exit(1)

    args.output_root.mkdir(parents=True, exist_ok=True)
    shared = dict(
        df=df, probs=probs, classes=classes,
        model_id=args.model_id, model_type=model_type,
        output_root=args.output_root, inference_seconds=inf_s,
        model_size_mb=model_size_mb, max_length=args.max_length,
        n_parameters=info["n_parameters"], hardware=info["hardware"],
    )

    if args.task == "binary":
        out1 = evaluate_level1_binary(**shared)
        print(f"  nivel 1 (binario OOD)        -> {out1}", file=sys.stderr)
        if not args.skip_subgroup:
            for out in evaluate_level3_subgroup(**shared):
                print(f"  nivel 3 (subgrupo)           -> {out}", file=sys.stderr)
    else:
        out2 = evaluate_level2_multiclass(**shared)
        print(f"  nivel 2 (multiclasse mapeado) -> {out2}", file=sys.stderr)


if __name__ == "__main__":
    main()
