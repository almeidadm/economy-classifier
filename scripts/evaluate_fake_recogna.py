#!/usr/bin/env python3
"""Avaliacao out-of-domain (OOD) sobre o portal `economia.uol.com.br` no FakeRecogna.

Complementa ``evaluate_fake_br.py`` com uma segunda base externa. O foco e o
sub-corpus do portal `economia.uol.com.br` (312/11902 docs), o unico indicador
*confiavel* de topico no FakeRecogna — a coluna ``category`` mistura criterios
editoriais e e fortemente correlacionada com ``label_name`` por artefatos de
curadoria (entretenimento 100% fake, mundo 99% fake, brasil 92% fake).

Niveis (parametrizaveis pelo CLI):

1. **Binario OOD no corpus completo** (default) — positivo = URL do dominio
   ``economia.uol.com.br`` (~2.6% de prevalencia, entre os 0.6% do Fake.Br e
   12.5% da Folha). Mesma natureza do nivel 1 do ``evaluate_fake_br.py``.
2. **Multiclasse OOD no corpus completo** — mapeia ``category`` para o esquema
   7+other via ``FAKE_RECOGNA_CAT_TO_TOPK`` e *sobrescreve* o gold para
   ``mercado`` quando a URL pertence ao economia.uol (URL > category como
   sinal de topico, conforme verificado: dos 312 docs do portal, 225 estao
   etiquetados como `saude` e 87 como `politica`).
3. **Balanceado intra-UOL (substitui o nivel 3 veracidade)** — junta os 312
   positivos do economia.uol com 312 negativos amostrados de outros portais
   ``*.uol.com.br`` (educacao/entretenimento/noticias/tab/tvefamosos/www),
   todos com ``label_name=='real'`` e ``seed=2026``. Como nao ha fake no
   economia.uol, o nivel 3 do evaluate_fake_br.py (subgrupo por veracidade)
   degenera; este corte controla simultaneamente *publisher* (so UOL) e
   *veracidade* (so real), isolando o sinal topical. Agora precision/recall/
   F1/AUC/Brier/ECE ficam todos bem-definidos sobre as 624 amostras
   (~50% de prevalencia positiva).

Texto enviado ao modelo: ``title + " " + sub_title`` (ambos texto cru). A
coluna ``news`` do FakeRecogna foi lematizada/lowercased no upstream e nao
casa com o pre-processamento do treino na Folha (raw + tokenizer do HF).

Cards usam ``regime='test_set'`` + ``config.domain='fake_recogna_economia_uol'``
para que o notebook de agregacao consiga separa-los dos cards in-domain.

Uso:

    uv run python scripts/evaluate_fake_recogna.py \\
        --model-dir /content/drive/MyDrive/economy-classifier/models/bertimbau_binary \\
        --model-id bert_bertimbau --task binary

    uv run python scripts/evaluate_fake_recogna.py \\
        --model-dir /content/drive/MyDrive/economy-classifier/models/bertimbau_multi \\
        --model-id bert_bertimbau --task multiclass
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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

DEFAULT_FAKE_RECOGNA_ROOT = Path(
    "/home/diacrono/Documentos/repositorios/fn-dataset-eda/data/raw/fake_recogna"
)
DEFAULT_HARDWARE = "local-CPU"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 128
ECONOMIA_UOL_NETLOC = "economia.uol.com.br"
POSITIVE_BINARY = 1

# Mapeamento aproximado das categorias FakeRecogna -> 7+other da Folha.
# - politica -> poder e mundo -> mundo sao matches diretos.
# - entretenimento -> ilustrada (secao Folha que cobre cultura/entretenimento).
# - brasil -> cotidiano (noticias nacionais cotidianas).
# - saude e ciencia folham para 'outros' por falta de analogo no schema 7+other.
# Aviso: FakeRecogna apresenta forte correlacao categoria<->label (e.g.
# entretenimento=100% fake, mundo=99% fake), entao essas classes estao
# enviesadas por construcao no corpus.
FAKE_RECOGNA_CAT_TO_TOPK: dict[str, str] = {
    "política": "poder",
    "politica": "poder",
    "mundo": "mundo",
    "entretenimento": "ilustrada",
    "brasil": "cotidiano",
    "saúde": "outros",
    "saude": "outros",
    "ciência": "outros",
    "ciencia": "outros",
}
TOPK_LABELS: list[str] = [
    "poder", "colunas", "mercado", "esporte",
    "mundo", "cotidiano", "ilustrada", "outros",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _build_input_text(title: Any, sub_title: Any) -> str:
    """Concatena title + sub_title (ambos cru). Fallback para title se sub_title null."""
    title_str = str(title).strip() if pd.notna(title) else ""
    sub_str = str(sub_title).strip() if pd.notna(sub_title) else ""
    if sub_str:
        return f"{title_str} {sub_str}".strip()
    return title_str


def load_fake_recogna(root: Path) -> pd.DataFrame:
    """Carrega FakeRecogna do .xlsx upstream e prepara o gold OOD.

    Retorna colunas: ``document_id, url, netloc, fake_recogna_category, label,
    label_name, text, is_economia_uol, y_true_binary, y_true_multi``.

    - ``y_true_binary``: 1 se netloc == 'economia.uol.com.br', 0 caso contrario.
    - ``y_true_multi``: category mapeada via FAKE_RECOGNA_CAT_TO_TOPK,
      *sobrescrito* para 'mercado' quando a URL pertence ao economia.uol.
    """
    candidates = sorted(root.glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError(f"Nenhum .xlsx encontrado em {root}")
    workbook = candidates[0]
    df = pd.read_excel(workbook)

    # O xlsx upstream usa nomes PT (Titulo/Subtitulo/...); normalizamos para os
    # nomes EN que o resto do projeto espera (mesma convencao do EDA repo).
    upstream_rename = {
        "Titulo": "title",
        "Subtitulo": "sub_title",
        "Noticia": "news",
        "Categoria": "category",
        "Data": "date",
        "Autor": "author",
        "URL": "url",
        "Classe": "label",
    }
    df = df.rename(columns={k: v for k, v in upstream_rename.items() if k in df.columns})
    if "label" in df.columns and "label_name" not in df.columns:
        df["label_name"] = df["label"].map({0: "fake", 1: "real"}).fillna("unknown")

    required = {"title", "url", "category", "label_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Colunas obrigatorias ausentes apos normalizacao: {sorted(missing)}. "
            f"Colunas disponiveis: {df.columns.tolist()}"
        )
    if "sub_title" not in df.columns:
        df["sub_title"] = pd.NA

    # documento_id sintetico (xlsx nao tem id estavel; usamos o indice da linha)
    df = df.reset_index().rename(columns={"index": "document_id"})
    df["document_id"] = df["document_id"].astype(str)
    df["url"] = df["url"].fillna("")
    df["netloc"] = df["url"].apply(lambda u: urlparse(u).netloc if u else "")
    df["fake_recogna_category"] = df["category"].fillna("").astype(str).str.strip().str.lower()
    df["is_economia_uol"] = df["netloc"] == ECONOMIA_UOL_NETLOC

    df["text"] = [
        _build_input_text(title, sub_title)
        for title, sub_title in zip(df["title"], df["sub_title"], strict=True)
    ]
    df = df[df["text"].str.len() > 0].copy()

    df["y_true_binary"] = df["is_economia_uol"].astype(int)
    df["y_true_multi"] = [
        "mercado" if is_econ else FAKE_RECOGNA_CAT_TO_TOPK.get(cat, "outros")
        for is_econ, cat in zip(df["is_economia_uol"], df["fake_recogna_category"], strict=True)
    ]
    return df.sort_values(["netloc", "document_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model loading / inference (mesmas convencoes de evaluate_fake_br.py)
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
    out_dir = output_root / f"{model_id}_binary_fake_recogna_economia_uol_ood"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({
        "index": df.index,
        "document_id": df["document_id"].values,
        "netloc": df["netloc"].values,
        "label_name": df["label_name"].values,
        "is_economia_uol": df["is_economia_uol"].values,
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
            "domain": "fake_recogna_economia_uol",
            "evaluation_type": "out_of_domain",
            "level": "1_binary_full_corpus",
            "max_length": max_length,
            "decision_threshold": 0.5,
            "positive_class_mapping": (
                "mercado (Folha) <- URL.netloc == 'economia.uol.com.br'"
            ),
            "input_text_strategy": "title + ' ' + sub_title (raw, ignora coluna news pre-processada)",
        },
        n_train_samples=None,
        n_eval_samples=n,
        predictions_path=str(out_dir / "predictions.csv"),
        notes=(
            "Avaliacao OOD no FakeRecogna. Positivo = URL pertence ao portal "
            f"'{ECONOMIA_UOL_NETLOC}' (312/11902, ~2.6%). Todos os positivos "
            "sao label_name='real' (nao ha fake economia.uol). Brier/ECE "
            "interpretar sob domain shift. URL e o sinal de topico mais "
            "confiavel: a coluna 'category' do FakeRecogna nesses 312 docs "
            "lista 225 como 'saude' e 87 como 'politica' — divergencia "
            "evidente entre rubrica do dataset e portal."
        ),
        hyperparameter_search=None,
    )
    persist_result_card(card, out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Nivel 2: multiclasse OOD no corpus completo
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
    out_dir = output_root / f"{model_id}_multiclass_fake_recogna_economia_uol_ood"
    out_dir.mkdir(parents=True, exist_ok=True)

    proba_cols = {
        f"y_proba_{idx_to_label[i]}": np.round(probs[:, i], 4)
        for i in range(probs.shape[1])
    }
    pred_df = pd.DataFrame({
        "index": df.index,
        "document_id": df["document_id"].values,
        "netloc": df["netloc"].values,
        "fake_recogna_category": df["fake_recogna_category"].values,
        "label_name": df["label_name"].values,
        "is_economia_uol": df["is_economia_uol"].values,
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
            "domain": "fake_recogna_economia_uol",
            "evaluation_type": "out_of_domain",
            "level": "2_multiclass_mapped",
            "max_length": max_length,
            "category_mapping": FAKE_RECOGNA_CAT_TO_TOPK,
            "url_override": (
                f"netloc == '{ECONOMIA_UOL_NETLOC}' sobrescreve gold para 'mercado'"
            ),
            "label_schema": TOPK_LABELS,
            "absent_classes": [c for c in TOPK_LABELS if c not in present_labels],
            "input_text_strategy": "title + ' ' + sub_title (raw)",
        },
        n_train_samples=None,
        n_eval_samples=n,
        predictions_path=str(out_dir / "predictions.csv"),
        notes=(
            "Multiclasse OOD no FakeRecogna. Gold = category mapeada via "
            "FAKE_RECOGNA_CAT_TO_TOPK, com override para 'mercado' nos 312 "
            "docs do economia.uol (URL > category, vide nivel 1). Aviso: "
            "FakeRecogna tem correlacao forte category<->label_name por "
            "curadoria (entretenimento=100% fake, mundo=99% fake, brasil=92% "
            "fake) — per_class_f1 nessas classes contem vies de curadoria do "
            "dataset, nao so erro do modelo. Classes colunas/esporte sem "
            "suporte (zero documentos) — F1=0 no schema completo."
        ),
        hyperparameter_search=None,
    )
    persist_result_card(card, out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Nivel 3: corte balanceado intra-UOL (positivos economia.uol + negativos
# amostrados de outros portais *.uol.com.br, todos `real`).
# ---------------------------------------------------------------------------
LEVEL3_SAMPLE_SEED = 2026


def evaluate_level3_uol_balanced(
    *, df: pd.DataFrame, probs: np.ndarray, classes: Any,
    model_id: str, model_type: str, output_root: Path,
    inference_seconds: float, model_size_mb: float,
    max_length: int, n_parameters: int | None, hardware: str,
    task: str,
) -> Path | None:
    """Combina os 312 positivos do economia.uol com 312 negativos amostrados
    de outros portais ``*.uol.com.br`` (todos com ``label_name=='real'``).

    Controla simultaneamente shift de publisher (toda a amostra vem do UOL)
    e shift de veracidade (so real), isolando o sinal topical. Com
    ~50% de prevalencia positiva, precision/recall/F1/AUC/Brier/ECE ficam
    todos bem-definidos — diferente do antigo "portal-only" que so reportava
    recall.
    """
    pos_positions = np.where(df["is_economia_uol"].to_numpy())[0]
    n_pos = len(pos_positions)
    if n_pos == 0:
        print("  nivel 3 ignorado: nenhum doc do economia.uol no corpus.", file=sys.stderr)
        return None

    other_uol_mask = (
        df["netloc"].str.endswith("uol.com.br", na=False)
        & ~df["is_economia_uol"]
        & (df["label_name"] == "real")
    ).to_numpy()
    other_positions = np.where(other_uol_mask)[0]
    if len(other_positions) == 0:
        print(
            "  nivel 3 ignorado: pool de outros portais UOL (real) vazio.",
            file=sys.stderr,
        )
        return None
    n_target = min(n_pos, len(other_positions))
    if n_target < n_pos:
        print(
            f"  AVISO: pool de outros UOL real tem apenas {len(other_positions)} "
            f"docs (< {n_pos} positivos); amostrando o pool inteiro.",
            file=sys.stderr,
        )

    rng = np.random.default_rng(LEVEL3_SAMPLE_SEED)
    sampled_neg = rng.choice(other_positions, size=n_target, replace=False)

    selected = np.sort(np.concatenate([pos_positions, sampled_neg]))
    sub_df = df.iloc[selected].reset_index(drop=True)
    sub_probs = probs[selected]
    n_sub = len(selected)

    neg_netloc_dist = (
        sub_df.loc[~sub_df["is_economia_uol"], "netloc"]
        .value_counts().to_dict()
    )
    sampling_meta = {
        "positive_n": int(n_pos),
        "negative_n": int(n_target),
        "negative_pool_size": int(len(other_positions)),
        "sample_seed": LEVEL3_SAMPLE_SEED,
        "negative_netloc_distribution": {
            k: int(v) for k, v in neg_netloc_dist.items()
        },
    }

    out_dir = output_root / f"{model_id}_{task}_fake_recogna_uol_balanced"
    out_dir.mkdir(parents=True, exist_ok=True)

    if task == "binary":
        pos_idx = _binary_positive_index(classes, model_type)
        y_score = sub_probs[:, pos_idx]
        y_pred = (y_score >= 0.5).astype(int)
        y_true = sub_df["y_true_binary"].to_numpy()

        metrics: dict[str, Any] = compute_binary_metrics(y_true, y_pred)
        metrics["auc_roc"] = (
            round(compute_roc_auc(y_true, y_score), 4)
            if len(set(y_true.tolist())) == 2 else None
        )
        metrics["brier"] = compute_brier_score(y_true, y_score)
        metrics["ece"] = compute_ece(y_true, y_score)
        metrics["positive_prevalence"] = round(float(y_true.mean()), 4)
        metrics["sampling"] = sampling_meta

        pred_df = pd.DataFrame({
            "index": sub_df.index,
            "document_id": sub_df["document_id"].values,
            "netloc": sub_df["netloc"].values,
            "is_economia_uol": sub_df["is_economia_uol"].values,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": np.round(y_score, 4),
            "method": model_id,
        })
        pred_df.to_csv(out_dir / "predictions.csv", index=False)

    else:  # multiclass
        if model_type == "bert":
            idx_to_label = dict(classes)
        else:
            idx_to_label = {i: c for i, c in enumerate(classes)}

        pred_idx = sub_probs.argmax(axis=1)
        y_pred = np.array([idx_to_label[int(i)] for i in pred_idx])
        y_true = sub_df["y_true_multi"].to_numpy()

        metrics_full = compute_multiclass_metrics(y_true, y_pred, labels=TOPK_LABELS)
        present_labels = sorted(set(y_true.tolist()))
        metrics_present = compute_multiclass_metrics(y_true, y_pred, labels=present_labels)
        metrics = {
            "macro_f1": metrics_full["macro_f1"],
            "weighted_f1": metrics_full["weighted_f1"],
            "accuracy": metrics_full["accuracy"],
            "per_class_f1": metrics_full["per_class_f1"],
            "macro_f1_present_only": metrics_present["macro_f1"],
            "present_labels": present_labels,
            "label_distribution": {
                label: int((y_true == label).sum()) for label in TOPK_LABELS
            },
            "sampling": sampling_meta,
        }

        proba_cols = {
            f"y_proba_{idx_to_label[i]}": np.round(sub_probs[:, i], 4)
            for i in range(sub_probs.shape[1])
        }
        pred_df = pd.DataFrame({
            "index": sub_df.index,
            "document_id": sub_df["document_id"].values,
            "netloc": sub_df["netloc"].values,
            "fake_recogna_category": sub_df["fake_recogna_category"].values,
            "is_economia_uol": sub_df["is_economia_uol"].values,
            "y_true": y_true,
            "y_pred": y_pred,
            "method": model_id,
            **proba_cols,
        })
        pred_df.to_csv(out_dir / "predictions.csv", index=False)

        cm = compute_confusion_matrix(
            y_true, y_pred, labels=TOPK_LABELS, normalize="true",
        )
        cm.to_csv(out_dir / "confusion_matrix.csv")

    card = build_result_card(
        model_id=model_id,
        task=task,
        regime="test_set",
        metrics=metrics,
        cost=_build_cost(
            inference_seconds=inference_seconds * (n_sub / len(df)),
            n_samples=n_sub, model_size_mb=model_size_mb,
            n_parameters=n_parameters, hardware=hardware,
        ),
        config={
            "domain": "fake_recogna_economia_uol",
            "evaluation_type": "out_of_domain_uol_balanced",
            "level": f"3_uol_balanced_{task}",
            "max_length": max_length,
            "netloc_positive": ECONOMIA_UOL_NETLOC,
            "netloc_negative_filter": "*.uol.com.br excluindo economia.uol.com.br",
            "label_name_filter": "real",
            "sample_seed": LEVEL3_SAMPLE_SEED,
            "input_text_strategy": "title + ' ' + sub_title (raw)",
        },
        n_train_samples=None,
        n_eval_samples=n_sub,
        predictions_path=str(out_dir / "predictions.csv"),
        notes=(
            f"Corte balanceado intra-UOL: {n_pos} positivos do "
            f"'{ECONOMIA_UOL_NETLOC}' + {n_target} negativos amostrados de "
            "outros portais *.uol.com.br (educacao, entretenimento, noticias, "
            "tab, tvefamosos, www), todos label_name=='real', seed=2026. "
            "Controla simultaneamente publisher (so UOL) e veracidade (so "
            "real), isolando o sinal topical. Em multiclasse o gold dos "
            "negativos vem de category mapeada via FAKE_RECOGNA_CAT_TO_TOPK "
            "(no UOL, isso e essencialmente politica->poder e saude->outros). "
            "Comparar este card com o nivel 1 para separar contribuicao do "
            "shift topical vs shift de registro fake."
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
        description="Avaliacao OOD do classificador da Folha no portal economia.uol (FakeRecogna).",
    )
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Diretorio com pesos HF (BERT) ou tfidf_pipeline.joblib (TF-IDF).")
    parser.add_argument("--model-id", type=str, required=True,
                        help="Identificador do card (ex.: bert_bertimbau, tfidf_logreg).")
    parser.add_argument("--task", choices=["binary", "multiclass"], required=True)
    parser.add_argument("--fake-recogna-root", type=Path, default=DEFAULT_FAKE_RECOGNA_ROOT,
                        help="Raiz do FakeRecogna (contem o .xlsx upstream).")
    parser.add_argument("--output-root", type=Path, default=RUNS_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--skip-uol-balanced", action="store_true",
                        help="Pula o nivel 3 (corte balanceado intra-UOL: economia.uol vs outros UOL real).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_dir.exists():
        print(f"ERRO: --model-dir nao encontrado: {args.model_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.fake_recogna_root.exists():
        print(f"ERRO: --fake-recogna-root nao encontrado: {args.fake_recogna_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Carregando FakeRecogna de {args.fake_recogna_root}...", file=sys.stderr)
    df = load_fake_recogna(args.fake_recogna_root)
    n_pos = int((df["y_true_binary"] == 1).sum())
    print(
        f"  {len(df)} docs validos | economia.uol={n_pos} "
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
    if args.task == "binary" and n_classes != 2:
        print(
            f"ERRO: --task=binary mas o modelo emite {n_classes} classes.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.task == "multiclass" and n_classes != len(TOPK_LABELS):
        print(
            f"ERRO: --task=multiclass mas o modelo emite {n_classes} classes; "
            f"esperado {len(TOPK_LABELS)}.",
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
        print(f"  nivel 1 (binario OOD)         -> {out1}", file=sys.stderr)
    else:
        out2 = evaluate_level2_multiclass(**shared)
        print(f"  nivel 2 (multiclasse mapeado) -> {out2}", file=sys.stderr)

    if not args.skip_uol_balanced:
        out3 = evaluate_level3_uol_balanced(**shared, task=args.task)
        if out3 is not None:
            print(f"  nivel 3 (UOL balanceado)      -> {out3}", file=sys.stderr)


if __name__ == "__main__":
    main()
