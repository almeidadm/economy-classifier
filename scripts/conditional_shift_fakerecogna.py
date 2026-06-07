#!/usr/bin/env python3
"""Decomposicao condicional do shift Folha -> FakeRecogna (primeiro tijolo da regua nova).

Computa, com BERTimbau fixo (sem fine-tune) e o KTS/MMD² do projeto:

    d⁺ = MMD²(Folha mercado,      FakeRecogna economia.uol)     -> governa o RECALL
    d⁻ = MMD²(Folha nao-mercado,  FakeRecogna nao-economia.uol) -> governa o FALSO-POSITIVO
    d_marginal = MMD²(Folha todos, FakeRecogna todos)           -> o que o NB 45 ja media

Positivo do FakeRecogna = portal `economia.uol.com.br` (312 docs; sinal de URL,
NAO a coluna Categoria) — identico a evaluate_fake_recogna.py. Texto do alvo =
Titulo + Subtitulo (raw); a coluna Noticia esta lematizada e e descartada.

CONTROLE DE COMPRIMENTO: o alvo so tem texto raw em nivel de manchete (~75 chars),
enquanto a Folha tem corpo (~2000). Para nao confundir deslocamento de dominio com
de comprimento, rodamos a decomposicao em DUAS granularidades da referencia:

    body   : Folha = Titulo + corpo   (comparavel ao MMD² marginal do NB 45)
    title  : Folha = Titulo apenas     (granularidade casada com a manchete do alvo)

A variante `title` e a distancia de dominio "limpa"; a `body` mostra o quanto a
distancia marginal do NB 45 esta inflada por comprimento no caso do FakeRecogna.

Uso:
    uv run python scripts/conditional_shift_fakerecogna.py
    uv run python scripts/conditional_shift_fakerecogna.py --folha-pool 800 --n-per-side 800 --n-perm 200  # run rapido
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from economy_classifier.shift_analysis import conditional_shift  # noqa: E402

ENCODER_ID = "neuralmind/bert-base-portuguese-cased"
EMB_MAX_LEN = 500  # uniforme com o classificador e o NB 45 (decisao max_length=500); antes 128
EMB_BATCH = 32
ECONOMIA_UOL_NETLOC = "economia.uol.com.br"
SEED = 2026

TRAIN_PARQUET = PROJECT_ROOT / "artifacts" / "splits" / "train.parquet"
FAKERECOGNA_DIR = PROJECT_ROOT / "ood_data" / "fake_recogna"


def embed_texts(texts: list[str], tokenizer, encoder, device, *,
                batch_size: int = EMB_BATCH, max_length: int = EMB_MAX_LEN) -> np.ndarray:
    """Mean-pool do last_hidden_state com mascara — identico ao NB 45 §3.5."""
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = [t if t else "[PAD]" for t in texts[i:i + batch_size]]
            enc = tokenizer(chunk, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
            h = encoder(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.cpu().numpy())
    return np.vstack(out)


def load_folha(pool: int, rng: np.random.Generator) -> dict[str, list[str]]:
    """Amostra `pool` positivos (mercado) e `pool` negativos da Folha train.

    Devolve titulo e corpo separados para as duas granularidades.
    """
    df = pd.read_parquet(TRAIN_PARQUET, columns=["title", "text", "label"])
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    out = {}
    for name, lab in [("pos", 1), ("neg", 0)]:
        sub = df[df["label"] == lab]
        take = min(pool, len(sub))
        idx = rng.choice(len(sub), size=take, replace=False)
        chosen = sub.iloc[idx]
        out[f"{name}_title"] = chosen["title"].tolist()
        out[f"{name}_body"] = [f"{t}. {x}"[:2000] for t, x in zip(chosen["title"], chosen["text"])]
    return out


def load_fakerecogna(neg_pool: int, rng: np.random.Generator) -> dict[str, list[str]]:
    """Positivos = economia.uol (todos os 312); negativos = resto (amostra `neg_pool`).

    Texto = Titulo + ' ' + Subtitulo (raw). Coluna Noticia (lematizada) descartada.
    """
    cand = sorted(FAKERECOGNA_DIR.glob("*.xlsx"))
    if not cand:
        raise FileNotFoundError(f"Nenhum .xlsx em {FAKERECOGNA_DIR}")
    df = pd.read_excel(cand[0]).rename(columns={"Titulo": "title", "Subtitulo": "sub_title", "URL": "url"})
    df["url"] = df["url"].fillna("")
    df["netloc"] = df["url"].apply(lambda u: urlparse(str(u)).netloc if u else "")

    def mk_text(row) -> str:
        t = str(row["title"]).strip() if pd.notna(row["title"]) else ""
        s = str(row.get("sub_title")).strip() if pd.notna(row.get("sub_title")) else ""
        return f"{t} {s}".strip()[:2000]

    df["txt"] = df.apply(mk_text, axis=1)
    df = df[df["txt"].str.len() > 0]
    pos = df[df["netloc"] == ECONOMIA_UOL_NETLOC]["txt"].tolist()
    neg_all = df[df["netloc"] != ECONOMIA_UOL_NETLOC]["txt"].tolist()
    take = min(neg_pool, len(neg_all))
    neg = [neg_all[int(i)] for i in rng.choice(len(neg_all), size=take, replace=False)]
    return {"pos": pos, "neg": neg, "n_pos": len(pos), "n_neg_total": len(neg_all)}


def run(args: argparse.Namespace) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer

    rng = np.random.default_rng(SEED)
    print("Carregando Folha train e FakeRecogna...", file=sys.stderr)
    folha = load_folha(args.folha_pool, rng)
    frec = load_fakerecogna(args.frec_neg_pool, rng)
    print(f"  Folha: {len(folha['pos_title'])} pos / {len(folha['neg_title'])} neg | "
          f"FakeRecogna: {frec['n_pos']} pos (economia.uol) / {len(frec['neg'])} de "
          f"{frec['n_neg_total']} neg", file=sys.stderr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding BERTimbau (device={device})...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(ENCODER_ID)
    enc = AutoModel.from_pretrained(ENCODER_ID).to(device).eval()

    def emb(texts):
        return embed_texts(texts, tok, enc, device)

    t0 = time.perf_counter()
    E = {
        "folha_pos_title": emb(folha["pos_title"]), "folha_neg_title": emb(folha["neg_title"]),
        "folha_pos_body": emb(folha["pos_body"]), "folha_neg_body": emb(folha["neg_body"]),
        "frec_pos": emb(frec["pos"]), "frec_neg": emb(frec["neg"]),
    }
    print(f"  embeddings em {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    ood_emb = np.vstack([E["frec_pos"], E["frec_neg"]])
    ood_lab = np.array([1] * len(E["frec_pos"]) + [0] * len(E["frec_neg"]))

    results = {}
    for gran in ("title", "body"):
        ref_emb = np.vstack([E[f"folha_pos_{gran}"], E[f"folha_neg_{gran}"]])
        ref_lab = np.array([1] * len(E[f"folha_pos_{gran}"]) + [0] * len(E[f"folha_neg_{gran}"]))
        results[gran] = conditional_shift(
            ref_emb, ref_lab, ood_emb, ood_lab,
            n_per_side=args.n_per_side, n_replicates=args.n_replicates,
            n_perm=args.n_perm, seed=SEED,
        )
    return {"results": results, "meta": {
        "encoder": ENCODER_ID, "seed": SEED, "n_per_side": args.n_per_side,
        "n_replicates": args.n_replicates, "n_perm": args.n_perm,
        "folha_pool": args.folha_pool, "frec_pos": frec["n_pos"],
        "target_text": "Titulo + Subtitulo (raw)",
        "note": "title=granularidade casada (dominio limpo); body=comparavel ao MMD² marginal do NB45 (length-confounded)",
    }}


def print_table(payload: dict) -> None:
    for gran, res in payload["results"].items():
        print(f"\n=== Granularidade ref = {gran.upper()} ===")
        print(f"{'componente':<10} {'MMD²_mean':>10} {'±std':>8} {'p_med':>7} {'n_rej':>6} {'n/lado':>7}")
        for comp in ("marginal", "positive", "negative"):
            r = res[comp]
            print(f"{comp:<10} {r['mmd2_mean']:>10.4f} {r['mmd2_std']:>8.4f} "
                  f"{r['p_median']:>7.3f} {r['n_reject']:>6d} {r['n_per_side']:>7d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folha-pool", type=int, default=1500, help="docs/classe da Folha a embeddar")
    ap.add_argument("--frec-neg-pool", type=int, default=1500, help="negativos do FakeRecogna a embeddar")
    ap.add_argument("--n-per-side", type=int, default=1500, help="subamostra por lado no KTS (cap p/ positivo=312)")
    ap.add_argument("--n-replicates", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "conditional_shift_fakerecogna.json")
    args = ap.parse_args()

    for p in (TRAIN_PARQUET, FAKERECOGNA_DIR):
        if not p.exists():
            print(f"ERRO: faltando {p}", file=sys.stderr); sys.exit(1)

    payload = run(args)
    print_table(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSalvo: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
