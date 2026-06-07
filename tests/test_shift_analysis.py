"""Testes da maquinaria de shift OOD: MMD²/KTS, decomposicao condicional e F1(R,φ,π)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from economy_classifier.shift_analysis import (
    conditional_shift,
    decompose_predictions,
    kts_permutation,
    mmd2_biased,
    predict_f1,
    replicated_kts,
)


# ---------------------------------------------------------------------------
# MMD² / KTS
# ---------------------------------------------------------------------------
def test_mmd2_zero_for_identical_blocks():
    """MMD² da mesma amostra contra si mesma e ~0 (kernel = 1 na diagonal cancela)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8))
    from sklearn.metrics.pairwise import rbf_kernel

    K = rbf_kernel(np.vstack([X, X]), gamma=0.5)
    assert abs(mmd2_biased(K, len(X))) < 1e-9


def test_kts_does_not_reject_same_distribution():
    """Duas subamostras da MESMA gaussiana -> MMD² pequeno, p alto (nao rejeita H0)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 10))
    Y = rng.normal(size=(120, 10))
    mmd2, p, sigma = kts_permutation(X, Y, n_perm=200, rng=rng)
    assert sigma > 0
    assert mmd2 < 0.05
    assert p > 0.05


def test_kts_rejects_shifted_distribution():
    """Gaussianas bem separadas -> MMD² grande, p ~ 0 (rejeita H0)."""
    rng = np.random.default_rng(2)
    X = rng.normal(loc=0.0, size=(120, 10))
    Y = rng.normal(loc=3.0, size=(120, 10))
    mmd2, p, _ = kts_permutation(X, Y, n_perm=200, rng=rng)
    assert mmd2 > 0.1
    assert p < 0.05


def test_replicated_kts_caps_n_per_side_to_smallest_pool():
    rng = np.random.default_rng(3)
    big = rng.normal(size=(500, 6))
    small = rng.normal(size=(37, 6))
    out = replicated_kts(big, small, n_per_side=1500, n_replicates=3, n_perm=50, rng=rng)
    assert out["n_per_side"] == 37
    assert out["n_replicates"] == 3
    assert {"mmd2_mean", "mmd2_std", "p_median", "n_reject"} <= out.keys()


def test_replicated_kts_raises_on_tiny_pool():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError):
        replicated_kts(np.zeros((1, 4)), np.zeros((5, 4)), n_per_side=10,
                       n_replicates=2, n_perm=10, rng=rng)


# ---------------------------------------------------------------------------
# Decomposicao condicional
# ---------------------------------------------------------------------------
def test_conditional_shift_isolates_negative_drift():
    """Positivos identicos fonte/alvo, negativos deslocados -> d⁻ >> d⁺.

    Demonstra que a decomposicao localiza ONDE esta o shift: a marginal mistura
    os dois, mas as condicionais separam.
    """
    rng = np.random.default_rng(5)
    # Fonte: positivos ~N(0), negativos ~N(0) em subespaco distinto (dims 5-9)
    n = 600
    ref = rng.normal(size=(n, 10))
    ref_lab = np.array([1] * (n // 2) + [0] * (n // 2))
    # Alvo: positivos IGUAIS (mesma gaussiana); negativos DESLOCADOS (+4 nas dims 5-9)
    ood = rng.normal(size=(n, 10))
    ood_lab = np.array([1] * (n // 2) + [0] * (n // 2))
    ood[ood_lab == 0, 5:] += 4.0

    out = conditional_shift(
        ref, ref_lab, ood, ood_lab,
        n_per_side=200, n_replicates=3, n_perm=100, seed=2026,
    )
    assert {"marginal", "positive", "negative"} == out.keys()
    assert out["negative"]["mmd2_mean"] > out["positive"]["mmd2_mean"]
    assert out["negative"]["n_reject"] == 3          # negativo sempre rejeita
    assert out["positive"]["p_median"] > 0.05        # positivo nao difere


def test_conditional_shift_rejects_misaligned_labels():
    rng = np.random.default_rng(6)
    emb = rng.normal(size=(10, 4))
    with pytest.raises(ValueError):
        conditional_shift(emb, np.array([1, 0]), emb, np.ones(10))


# ---------------------------------------------------------------------------
# Forma fechada F1(R, φ, π)
# ---------------------------------------------------------------------------
def test_predict_f1_matches_sklearn_identity():
    """A forma fechada reproduz exatamente o F1 da matriz de confusao."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = rng.integers(200, 2000)
        prevalence = float(rng.uniform(0.02, 0.6))
        n_pos = max(2, int(round(n * prevalence)))
        n_neg = n - n_pos
        recall = float(rng.uniform(0.1, 0.95))
        fp_rate = float(rng.uniform(0.0, 0.3))
        tp = int(round(recall * n_pos))
        fp = int(round(fp_rate * n_neg))
        y_true = np.array([1] * n_pos + [0] * n_neg)
        y_pred = np.array([1] * tp + [0] * (n_pos - tp) + [1] * fp + [0] * (n_neg - fp))

        d = decompose_predictions(y_true, y_pred)
        # forma fechada com (R, φ, π) recuperados == F1 observado == sklearn
        assert d["f1_observed"] == pytest.approx(d["f1_reconstructed"], abs=1e-6)
        assert d["f1_observed"] == pytest.approx(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0), abs=1e-6
        )


def test_predict_f1_prevalence_drives_collapse():
    """Mesmo R e φ, so muda π: F1 desaba quando a prevalencia cai (efeito FakeRecogna full vs balanced)."""
    R, phi = 0.55, 0.03
    f1_balanced = predict_f1(R, phi, prevalence=0.50)
    f1_full = predict_f1(R, phi, prevalence=0.026)
    assert f1_balanced > 0.6
    assert f1_full < 0.45
    assert f1_balanced - f1_full > 0.2


def test_predict_f1_handles_degenerate_zero():
    assert predict_f1(recall=0.0, fp_rate=0.0, prevalence=0.0) == 0.0
