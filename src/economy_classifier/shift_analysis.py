"""Caracterizacao de distribution shift OOD: MMD²/KTS e a *decomposicao completa*.

Este modulo formaliza, fora do notebook, a maquinaria de shift representacional que
estava inline no ``45_ood_evaluation.ipynb`` (secao 3.5–3.7) e a estende para a
**decomposicao condicional** que sustenta a reformulacao metodologica do artigo.

Duas camadas:

1. **MMD² / KTS** (Kernel Two-Sample Test) — estimador biased de MMD² com kernel
   RBF (largura pela heuristica da mediana) e p-valor por permutacao. Identico ao
   que o NB 45 ja reportava como ``shift_mmd2_kts_summary.csv`` — agora testavel.

2. **Decomposicao do drift por classe** — em vez de uma unica distancia marginal
   ``d(P_fonte(x), P_alvo(x))``, mede separadamente:

   - ``d⁺ = MMD²(P_fonte(x|y=+),  P_alvo(x|y=+))``  — drift do POSITIVO (mercado)
   - ``d⁻ = MMD²(P_fonte(x|y=−),  P_alvo(x|y=−))``  — drift do NEGATIVO
   - ``d_marginal``                                 — o que o NB 45 ja media

   Motivacao (ver ``docs/analise_shift_mmd2_e_cards_ood.md`` §3): a distancia
   marginal sozinha NAO prediz a queda de F1 OOD; ela e dominada por confounds de
   superficie e e cega a prevalencia. A degradacao binaria obedece a

       F1 = 2·R·π / [π·(1+R) + φ·(1−π)]

   onde ``R`` (recall nos positivos) e governado por ``d⁺``, ``φ`` (taxa de falso-
   positivo nos negativos) por ``d⁻``, e ``π`` (prevalencia) e conhecida. ``predict_f1``
   implementa essa forma fechada; ``decompose_predictions`` recupera ``(R, φ, π)`` de
   um ``predictions.csv`` e confirma a identidade. O resíduo entre F1 previsto (a
   partir de ``d⁺, d⁻`` modelados) e F1 observado estima o concept shift.

Embeddings NAO sao responsabilidade deste modulo (dependeriam de torch/transformers
e quebrariam a testabilidade). O chamador embeda com encoder fixo (BERTimbau, sem
fine-tune) e passa arrays ``np.ndarray``; ver ``scripts/conditional_shift_fakerecogna.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel

DEFAULT_SEED = 2026
DEFAULT_N_PER_SIDE = 1500
DEFAULT_N_REPLICATES = 5
DEFAULT_N_PERMUTATIONS = 500
ALPHA = 0.05


# ---------------------------------------------------------------------------
# MMD² / KTS — extraidos verbatim (em comportamento) do NB 45 §3.6
# ---------------------------------------------------------------------------
def median_heuristic_sigma(Z: np.ndarray, rng: np.random.Generator, n_probe: int = 1000) -> float:
    """Largura sigma do RBF pela heuristica da mediana sobre uma subamostra de ``Z``."""
    sub = Z[rng.choice(len(Z), size=min(n_probe, len(Z)), replace=False)]
    d = pdist(sub)
    return float(np.sqrt(np.median(d ** 2) / 2))


def mmd2_biased(K: np.ndarray, n: int) -> float:
    """Estimador *biased* de MMD² a partir da matriz de kernel ``K`` do bloco [X; Y].

    ``n = |X|``; ``m = |Y| = K.shape[0] - n``.
    """
    m = K.shape[0] - n
    K_xx = K[:n, :n].sum() / (n * n)
    K_yy = K[n:, n:].sum() / (m * m)
    K_xy = K[:n, n:].sum() / (n * m)
    return float(K_xx + K_yy - 2 * K_xy)


def kts_permutation(
    X: np.ndarray, Y: np.ndarray, n_perm: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Kernel Two-Sample Test via permutacao.

    Retorna ``(mmd2_observado, p_valor, sigma)``. ``p`` e a fracao das permutacoes
    (embaralhamento dos rotulos de origem, reusando a matriz de kernel) em que
    ``MMD²_permutado >= MMD²_observado``.
    """
    Z = np.vstack([X, Y])
    n = len(X)
    sigma = median_heuristic_sigma(Z, rng)
    K = rbf_kernel(Z, gamma=1.0 / (2 * sigma ** 2))
    observed = mmd2_biased(K, n)
    idx_base = np.arange(len(Z))
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(idx_base)
        Kp = K[np.ix_(perm, perm)]
        null[b] = mmd2_biased(Kp, n)
    p = float((null >= observed).mean())
    return observed, p, sigma


def replicated_kts(
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    *,
    n_per_side: int = DEFAULT_N_PER_SIDE,
    n_replicates: int = DEFAULT_N_REPLICATES,
    n_perm: int = DEFAULT_N_PERMUTATIONS,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Roda ``n_replicates`` KTS independentes, subamostrando ``n_per_side`` por lado.

    Retorna o agregado no mesmo formato de ``shift_mmd2_kts_summary.csv``:
    ``mmd2_mean, mmd2_std, p_median, n_reject, n_per_side, n_replicates``.
    ``n_per_side`` efetivo e limitado pelo menor pool (relevante p/ o positivo, que
    em bases como o FakeRecogna tem poucas centenas de docs).
    """
    n = int(min(n_per_side, len(X_pool), len(Y_pool)))
    if n < 2:
        raise ValueError(f"pool insuficiente para KTS: n_efetivo={n} (X={len(X_pool)}, Y={len(Y_pool)})")
    mmd2s, pvals = [], []
    for _ in range(n_replicates):
        ix = rng.choice(len(X_pool), size=n, replace=False)
        iy = rng.choice(len(Y_pool), size=n, replace=False)
        mmd2, p, _sigma = kts_permutation(X_pool[ix], Y_pool[iy], n_perm, rng)
        mmd2s.append(mmd2)
        pvals.append(p)
    mmd2s_arr = np.asarray(mmd2s)
    pvals_arr = np.asarray(pvals)
    return {
        "mmd2_mean": round(float(mmd2s_arr.mean()), 6),
        "mmd2_std": round(float(mmd2s_arr.std()), 6),
        "p_median": round(float(np.median(pvals_arr)), 4),
        "n_reject": int((pvals_arr < ALPHA).sum()),
        "n_per_side": n,
        "n_replicates": n_replicates,
    }


# ---------------------------------------------------------------------------
# Decomposicao condicional: d⁺, d⁻, d_marginal
# ---------------------------------------------------------------------------
def conditional_shift(
    ref_emb: np.ndarray,
    ref_labels: np.ndarray,
    ood_emb: np.ndarray,
    ood_labels: np.ndarray,
    *,
    positive_label: int = 1,
    n_per_side: int = DEFAULT_N_PER_SIDE,
    n_replicates: int = DEFAULT_N_REPLICATES,
    n_perm: int = DEFAULT_N_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, Any]]:
    """Decompoe o shift fonte→alvo em marginal, positivo-condicional e negativo-condicional.

    ``ref_*`` e o pool in-domain (Folha train); ``ood_*`` e a base OOD. ``*_labels``
    sao binarios (``positive_label`` = classe mercado). Cada chave do dict de saida
    (``marginal``, ``positive``, ``negative``) e o agregado de ``replicated_kts``.

    Usa UM unico ``rng`` semeado para que as tres medidas compartilhem o mesmo fluxo
    de aleatoriedade (reprodutivel; ``seed=2026`` por padrao, coerente com o projeto).
    """
    ref_labels = np.asarray(ref_labels)
    ood_labels = np.asarray(ood_labels)
    if len(ref_labels) != len(ref_emb) or len(ood_labels) != len(ood_emb):
        raise ValueError("labels e embeddings desalinhados (tamanhos diferentes).")

    rng = np.random.default_rng(seed)
    ref_pos, ref_neg = ref_emb[ref_labels == positive_label], ref_emb[ref_labels != positive_label]
    ood_pos, ood_neg = ood_emb[ood_labels == positive_label], ood_emb[ood_labels != positive_label]

    common = dict(n_per_side=n_per_side, n_replicates=n_replicates, n_perm=n_perm, rng=rng)
    return {
        "marginal": replicated_kts(ref_emb, ood_emb, **common),
        "positive": replicated_kts(ref_pos, ood_pos, **common),
        "negative": replicated_kts(ref_neg, ood_neg, **common),
    }


# ---------------------------------------------------------------------------
# Forma fechada F1(R, φ, π) e recuperacao empirica de (R, φ, π)
# ---------------------------------------------------------------------------
def predict_f1(recall: float, fp_rate: float, prevalence: float) -> float:
    """F1 binario em funcao de recall ``R``, taxa de falso-positivo ``φ`` e prevalencia ``π``.

    ``F1 = 2·R·π / [π·(1+R) + φ·(1−π)]`` — identidade algebrica derivada de
    ``TP=Rπ N, FN=(1−R)π N, FP=φ(1−π) N`` substituidos em ``2TP/(2TP+FP+FN)``.
    """
    denom = prevalence * (1.0 + recall) + fp_rate * (1.0 - prevalence)
    if denom <= 0.0:
        return 0.0
    return float(2.0 * recall * prevalence / denom)


def decompose_predictions(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> dict[str, Any]:
    """Recupera ``(recall, fp_rate, prevalence)`` de predicoes binarias e confere a forma fechada.

    Retorna tambem ``f1_observed`` (direto da matriz de confusao) e
    ``f1_reconstructed`` (via ``predict_f1``). Os dois devem coincidir ate
    erro de ponto flutuante — e o teste de sanidade da identidade. A
    *predicao* cientifica (F1 a partir de ``d⁺, d⁻``) so aparece quando ``recall``
    e ``fp_rate`` sao MODELADOS a partir das distancias, nao medidos.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos = y_true == positive_label
    neg = ~pos
    tp = int((y_pred[pos] == positive_label).sum())
    fn = int(pos.sum() - tp)
    fp = int((y_pred[neg] == positive_label).sum())
    tn = int(neg.sum() - fp)

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    prevalence = (tp + fn) / len(y_true) if len(y_true) else 0.0
    f1_observed = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    return {
        "recall": round(recall, 6),
        "fp_rate": round(fp_rate, 6),
        "prevalence": round(prevalence, 6),
        "f1_observed": round(f1_observed, 6),
        "f1_reconstructed": round(predict_f1(recall, fp_rate, prevalence), 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
