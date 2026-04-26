"""Tests for economy_classifier.datasets — splits and balancing."""

import pandas as pd

from economy_classifier.datasets import (
    MULTICLASS_TOP7,
    OTHER_LABEL,
    attach_multiclass_label,
    build_balanced_training_frame,
    build_cv_folds,
    build_train_val_test_split,
    map_to_multiclass,
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
    assert abs(len(test) / n - 0.10) < 0.02
    assert abs(len(val) / n - 0.10) < 0.02
    assert abs(len(train) / n - 0.80) < 0.02


def test_split_rejects_oversized_holdouts(synthetic_corpus):
    try:
        build_train_val_test_split(synthetic_corpus, test_size=0.5, val_size=0.5)
    except ValueError:
        return
    raise AssertionError("expected ValueError when test_size + val_size >= 1.0")


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


def test_map_to_multiclass_keeps_top7():
    series = pd.Series(["mercado", "tec", "poder", "bbc", "esporte", "opiniao"])
    mapped = map_to_multiclass(series)
    assert mapped.tolist() == [
        "mercado", OTHER_LABEL, "poder", OTHER_LABEL, "esporte", OTHER_LABEL,
    ]


def test_map_to_multiclass_label_set_size():
    series = pd.Series(list(MULTICLASS_TOP7) + ["a", "b", "c", "d"])
    mapped = map_to_multiclass(series)
    assert set(mapped.unique()).issubset(set(MULTICLASS_TOP7) | {OTHER_LABEL})
    assert len(set(mapped.unique())) <= 8


def test_attach_multiclass_preserves_index():
    df = pd.DataFrame(
        {"text": ["a", "b", "c"], "category": ["mercado", "tec", "poder"]},
        index=[10, 20, 30],
    )
    out = attach_multiclass_label(df)
    assert list(out.index) == [10, 20, 30]
    assert len(out) == len(df)
    assert "label_multi" in out.columns
    assert out["label_multi"].tolist() == ["mercado", OTHER_LABEL, "poder"]


def test_cv_folds_count_and_keys(synthetic_corpus):
    folds = build_cv_folds(synthetic_corpus, n_folds=5, seed=42)
    assert len(folds) == 5
    for fold in folds:
        assert {"train_indices", "val_indices"} == set(fold.keys())


def test_cv_folds_default_is_5(synthetic_corpus):
    folds = build_cv_folds(synthetic_corpus, seed=42)
    assert len(folds) == 5


def test_cv_folds_partition_is_disjoint_within_fold(synthetic_corpus):
    folds = build_cv_folds(synthetic_corpus, n_folds=5, seed=42)
    for fold in folds:
        train_idx = set(fold["train_indices"])
        val_idx = set(fold["val_indices"])
        assert train_idx.isdisjoint(val_idx)
        assert train_idx | val_idx == set(synthetic_corpus.index)


def test_cv_folds_val_unions_to_full_corpus(synthetic_corpus):
    folds = build_cv_folds(synthetic_corpus, n_folds=5, seed=42)
    all_val: list[int] = []
    for fold in folds:
        all_val.extend(fold["val_indices"])
    assert sorted(all_val) == sorted(synthetic_corpus.index.tolist())


def test_cv_folds_stratified_preserves_class_balance(synthetic_corpus):
    folds = build_cv_folds(synthetic_corpus, n_folds=5, seed=42)
    overall_pct = synthetic_corpus["label"].mean()
    for fold in folds:
        val_subset = synthetic_corpus.loc[fold["val_indices"]]
        fold_pct = val_subset["label"].mean()
        assert abs(fold_pct - overall_pct) < 0.05, (
            f"fold val class balance {fold_pct:.3f} drifts from overall {overall_pct:.3f}"
        )


def test_cv_folds_serializable_to_json(synthetic_corpus):
    import json
    folds = build_cv_folds(synthetic_corpus, n_folds=5, seed=42)
    blob = json.dumps(folds)
    loaded = json.loads(blob)
    assert len(loaded) == 5
    assert all(isinstance(i, int) for i in loaded[0]["train_indices"])


def test_cv_folds_determinism(synthetic_corpus):
    a = build_cv_folds(synthetic_corpus, n_folds=5, seed=42)
    b = build_cv_folds(synthetic_corpus, n_folds=5, seed=42)
    assert a == b


def test_cv_folds_rejects_n_folds_below_2(synthetic_corpus):
    try:
        build_cv_folds(synthetic_corpus, n_folds=1, seed=42)
    except ValueError:
        return
    raise AssertionError("expected ValueError when n_folds < 2")


def test_attach_multiclass_raises_when_category_missing():
    df = pd.DataFrame({"text": ["a"], "label": [1]})
    try:
        attach_multiclass_label(df)
    except KeyError as exc:
        assert "category" in str(exc)
    else:
        raise AssertionError("expected KeyError")
