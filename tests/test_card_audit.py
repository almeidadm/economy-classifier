"""Testes da auditoria de result_cards contra a Guarda Metodologica."""

from __future__ import annotations

from economy_classifier.card_audit import (
    audit_card,
    audit_run_tree,
    classify_method,
    is_ood_card,
    primary_metric_value,
    summarize,
)


def _codes(issues, severity=None):
    return {i.code for i in issues if severity is None or i.severity == severity}


# ---------------------------------------------------------------------------
# classificacao de metodo
# ---------------------------------------------------------------------------
def test_classify_method_by_prefix():
    assert classify_method("tfidf_logreg") == "tfidf"
    assert classify_method("bert_bertimbau") == "bert"
    assert classify_method("llm_qwen2_5_7b_instruct") == "llm"
    assert classify_method("ensemble_stacking") == "ensemble"
    assert classify_method("mistery_model") == "unknown"


# ---------------------------------------------------------------------------
# cards de referencia (validos)
# ---------------------------------------------------------------------------
def _tfidf_binary_test_set():
    return {
        "model_id": "tfidf_linearsvc",
        "task": "binary",
        "regime": "test_set",
        "metrics": {"precision": 0.9, "recall": 0.83, "f1": 0.86, "accuracy": 0.97,
                    "auc_roc": 0.98, "brier": 0.02, "ece": 0.01},
        "cost": {},
        "config": {},
        "n_eval_samples": 16629,
        "hyperparameter_search": {"scoring": "f1", "n_trials": 60, "best_score": 0.85},
    }


def test_valid_tfidf_card_has_no_errors():
    issues = audit_card(_tfidf_binary_test_set())
    assert _codes(issues, "ERROR") == set()


def test_valid_bert_cv_card_has_no_errors():
    card = {
        "model_id": "bert_bertimbau",
        "task": "multiclass",
        "regime": "cv_5fold",
        "metrics": {"macro_f1_mean": 0.7, "macro_f1_std": 0.01, "accuracy_mean": 0.8,
                    "accuracy_std": 0.005},
        "cost": {}, "config": {}, "n_eval_samples": 1000,
        "hyperparameter_search": None,
    }
    assert _codes(audit_card(card), "ERROR") == set()


# ---------------------------------------------------------------------------
# violacoes que devem virar ERROR
# ---------------------------------------------------------------------------
def test_tfidf_without_hp_search_is_error():
    card = _tfidf_binary_test_set()
    card["hyperparameter_search"] = None
    assert "tfidf-missing-hp" in _codes(audit_card(card), "ERROR")


def test_binary_card_with_f1_macro_scoring_is_error():
    card = _tfidf_binary_test_set()
    card["hyperparameter_search"]["scoring"] = "f1_macro"
    assert "bad-scoring" in _codes(audit_card(card), "ERROR")


def test_cv_card_without_std_is_error():
    card = {
        "model_id": "bert_bertimbau",
        "task": "binary",
        "regime": "cv_5fold",
        "metrics": {"f1_mean": 0.86, "accuracy_mean": 0.96},  # falta f1_std
        "cost": {}, "config": {}, "n_eval_samples": 1000,
        "hyperparameter_search": None,
    }
    assert "cv-no-variance" in _codes(audit_card(card), "ERROR")


def test_missing_primary_metric_is_error():
    card = _tfidf_binary_test_set()
    card["metrics"] = {"precision": 0.9, "recall": 0.83}  # sem f1
    assert "missing-primary" in _codes(audit_card(card), "ERROR")


def test_missing_required_field_is_error():
    card = _tfidf_binary_test_set()
    del card["cost"]
    assert "missing-field" in _codes(audit_card(card), "ERROR")


def test_missing_predictions_is_error():
    issues = audit_card(_tfidf_binary_test_set(), predictions_exists=False)
    assert "no-predictions" in _codes(issues, "ERROR")


# ---------------------------------------------------------------------------
# avisos (WARN)
# ---------------------------------------------------------------------------
def test_bert_with_populated_hp_is_warn():
    card = {
        "model_id": "bert_finbert_ptbr",
        "task": "binary",
        "regime": "test_set",
        "metrics": {"f1": 0.8, "brier": 0.02, "ece": 0.01},
        "cost": {}, "config": {}, "n_eval_samples": 100,
        "hyperparameter_search": {"scoring": "f1", "n_trials": 25},
    }
    assert "unexpected-hp" in _codes(audit_card(card), "WARN")


def test_llm_coverage_below_one_and_auc_are_warns():
    card = {
        "model_id": "llm_qwen2_5_7b_instruct",
        "task": "binary",
        "regime": "test_set",
        "metrics": {"f1": 0.7, "auc_roc": 0.7, "coverage": 0.99},
        "cost": {}, "config": {}, "n_eval_samples": 100,
        "hyperparameter_search": None,
    }
    warns = _codes(audit_card(card), "WARN")
    assert {"llm-coverage", "llm-auc"} <= warns


def test_calibrated_method_without_brier_ece_is_warn():
    card = _tfidf_binary_test_set()
    card["metrics"].pop("brier")
    card["metrics"].pop("ece")
    assert "no-calibration" in _codes(audit_card(card), "WARN")


# ---------------------------------------------------------------------------
# arvore: cobertura de regimes
# ---------------------------------------------------------------------------
def _write_card(tmp_path, folder, card):
    import json

    d = tmp_path / folder
    d.mkdir()
    (d / "result_card.json").write_text(json.dumps(card), encoding="utf-8")
    (d / "predictions.csv").write_text("y_true,y_pred\n1,1\n", encoding="utf-8")


def test_missing_cv_regime_is_error_for_bert(tmp_path):
    base = {
        "model_id": "bert_bertimbau", "task": "binary",
        "metrics": {"f1": 0.86, "brier": 0.02}, "cost": {}, "config": {},
        "n_eval_samples": 100, "hyperparameter_search": None,
    }
    _write_card(tmp_path, "bert_bertimbau_binary_fixed_split", {**base, "regime": "fixed_split"})
    _write_card(tmp_path, "bert_bertimbau_binary_test_set", {**base, "regime": "test_set"})
    # falta cv_5fold de proposito
    _records, issues = audit_run_tree(tmp_path)
    miss = [i for i in issues if i.code == "missing-regime" and i.severity == "ERROR"]
    assert any("cv_5fold" in i.message for i in miss)


def test_llm_test_set_only_does_not_trigger_missing_regime(tmp_path):
    card = {
        "model_id": "llm_qwen2_5_7b_instruct", "task": "binary", "regime": "test_set",
        "metrics": {"f1": 0.7, "coverage": 1.0}, "cost": {}, "config": {},
        "n_eval_samples": 100, "hyperparameter_search": None,
    }
    _write_card(tmp_path, "llm_qwen_binary_zero_shot_test_set", card)
    _records, issues = audit_run_tree(tmp_path)
    assert "missing-regime" not in {i.code for i in issues}


# ---------------------------------------------------------------------------
# cards OOD (inferencia: regime=test_set, hp=null, sem cobertura de regimes)
# ---------------------------------------------------------------------------
def _ood_tfidf_card():
    return {
        "model_id": "tfidf_nb",
        "task": "binary",
        "regime": "test_set",
        "metrics": {"f1": 0.4, "precision": 0.3, "recall": 0.6, "brier": 0.2, "ece": 0.1},
        "cost": {},
        "config": {"evaluation_type": "out_of_domain", "domain": "fake_br", "level": 1},
        "n_eval_samples": 7200,
        "hyperparameter_search": None,
    }


def test_is_ood_card_detects_evaluation_type():
    assert is_ood_card(_ood_tfidf_card()) is True
    assert is_ood_card(_tfidf_binary_test_set()) is False


def test_ood_tfidf_null_hp_is_not_error():
    # hp=null e correto em OOD (inferencia reusa o modelo in-domain)
    assert "tfidf-missing-hp" not in _codes(audit_card(_ood_tfidf_card()), "ERROR")


def test_ood_cards_excluded_from_regime_coverage(tmp_path):
    # tres datasets OOD do mesmo modelo, todos regime=test_set: NAO deve gerar
    # missing-regime (cv_5fold/fixed_split nao existem em OOD).
    for ds in ("fake_br", "recognasumm", "portuguese_news"):
        card = _ood_tfidf_card()
        card["config"] = {"evaluation_type": "out_of_domain", "domain": ds, "level": 1}
        _write_card(tmp_path, f"tfidf_nb_binary_{ds}_ood", card)
    _records, issues = audit_run_tree(tmp_path)
    assert "missing-regime" not in {i.code for i in issues}


def test_ood_bad_scoring_still_flagged():
    # se um payload de busca vazar num card OOD com scoring errado, ainda pega
    card = _ood_tfidf_card()
    card["hyperparameter_search"] = {"scoring": "f1_macro", "n_trials": 60}
    assert "bad-scoring" in _codes(audit_card(card), "ERROR")


# ---------------------------------------------------------------------------
# extracao da metrica primaria (para a tabela comparativa)
# ---------------------------------------------------------------------------
def test_primary_metric_binary_point():
    pm = primary_metric_value(_tfidf_binary_test_set())
    assert pm["metric"] == "f1" and pm["value"] == 0.86 and pm["std"] is None


def test_primary_metric_cv_uses_mean_and_std():
    card = {
        "model_id": "bert_bertimbau", "task": "multiclass", "regime": "cv_5fold",
        "metrics": {"macro_f1_mean": 0.71, "macro_f1_std": 0.02},
        "cost": {}, "config": {},
    }
    pm = primary_metric_value(card)
    assert pm["metric"] == "macro_f1" and pm["value"] == 0.71 and pm["std"] == 0.02


def test_primary_metric_accepts_f1_macro_alias():
    card = {"model_id": "tfidf_nb", "task": "multiclass", "regime": "test_set",
            "metrics": {"f1_macro": 0.5}, "cost": {}, "config": {}}
    assert primary_metric_value(card)["value"] == 0.5


def test_primary_metric_none_when_absent():
    card = {"model_id": "x", "task": "binary", "regime": "test_set",
            "metrics": {"precision": 0.9}, "cost": {}, "config": {}}
    assert primary_metric_value(card) is None


def test_summarize_counts_by_severity():
    card = _tfidf_binary_test_set()
    card["hyperparameter_search"] = None  # 1 ERROR
    counts = summarize(audit_card(card))
    assert counts["ERROR"] >= 1
