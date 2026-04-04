"""Tests for economy_classifier.datasets — splits and balancing."""

from economy_classifier.datasets import (
    build_balanced_training_frame,
    build_train_val_test_split,
)


def test_split_disjunction(synthetic_corpus):
    train, val, test = build_train_val_test_split(synthetic_corpus, seed=42)
    train_idx = set(train.index)
    val_idx = set(val.index)
    test_idx = set(test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_split_coverage(synthetic_corpus):
    train, val, test = build_train_val_test_split(synthetic_corpus, seed=42)
    assert len(train) + len(val) + len(test) == len(synthetic_corpus)


def test_split_stratification(synthetic_corpus):
    train, val, test = build_train_val_test_split(synthetic_corpus, seed=42)
    for split in [train, val, test]:
        pct = split["label"].mean() * 100
        assert 8.0 <= pct <= 18.0, f"mercado pct {pct:.1f}% outside tolerance"


def test_split_determinism(synthetic_corpus):
    t1, v1, te1 = build_train_val_test_split(synthetic_corpus, seed=42)
    t2, v2, te2 = build_train_val_test_split(synthetic_corpus, seed=42)
    assert list(t1.index) == list(t2.index)
    assert list(v1.index) == list(v2.index)
    assert list(te1.index) == list(te2.index)


def test_split_proportions(synthetic_corpus):
    train, val, test = build_train_val_test_split(synthetic_corpus, seed=42)
    n = len(synthetic_corpus)
    assert abs(len(test) / n - 0.20) < 0.05
    assert abs(len(val) / n - 0.16) < 0.05
    assert abs(len(train) / n - 0.64) < 0.05


def test_balanced_train_50_50(synthetic_corpus):
    train, _, _ = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)
    counts = balanced["label"].value_counts()
    assert counts[0] == counts[1]


def test_balanced_train_size(synthetic_corpus):
    train, _, _ = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)
    n_positive = (train["label"] == 1).sum()
    assert len(balanced) == 2 * n_positive


def test_balanced_preserves_val_test(synthetic_corpus):
    train, val, test = build_train_val_test_split(synthetic_corpus, seed=42)
    balanced = build_balanced_training_frame(train, seed=42)
    # val and test must not be affected (they are separate objects)
    assert len(val) > 0
    assert len(test) > 0
    # balanced indices are a subset of train indices
    assert set(balanced.index).issubset(set(train.index))


def test_split_preserves_original_index(synthetic_corpus):
    train, val, test = build_train_val_test_split(synthetic_corpus, seed=42)
    all_indices = sorted(list(train.index) + list(val.index) + list(test.index))
    assert all_indices == list(range(len(synthetic_corpus)))


def test_build_train_val_test_split_accepts_label_column(synthetic_corpus):
    df = synthetic_corpus.rename(columns={"label": "target"})
    train, val, test = build_train_val_test_split(df, label_column="target", seed=42)
    assert len(train) + len(val) + len(test) == len(df)
