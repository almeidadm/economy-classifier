# Tabela comparativa de resultados (Drive)

**Data:** 2026-06-04 · **Fonte:** `artifacts/drive_audit/` (97 cards in-domain + 117 OOD).

**Métrica primária:** F1 (binário) / macro-F1 (multiclasse). Em `cv_5fold`, média±desvio dos 5 folds (comparado pela média).

**Destaque:** **negrito** = vencedor da coluna (maior métrica do regime/dataset). Empates são marcados juntos.

**Ausências:** `∅` = pasta existe no Drive **sem** `result_card.json` (lacuna real); `—` = combinação inexistente/não planejada (LLM e stacking só têm `test_set`); `?` = card sem a métrica.

> Pastas sem card no inventário: **9** (ver `docs/auditoria_result_cards.md`).


## 1. In-domain — BINÁRIO (F1)

| Modelo | fixed_split | cv_5fold (média±dp) | test_set |
|---|---|---|---|
| `tfidf_linearsvc` | 0.857 | 0.856±0.004 | 0.861 |
| `tfidf_linearsvc_nostop` | 0.863 | 0.859±0.004 | 0.861 |
| `tfidf_logreg` | 0.858 | 0.851±0.003 | 0.860 |
| `tfidf_logreg_nostop` | 0.857 | 0.856±0.006 | 0.861 |
| `tfidf_nb` | 0.755 | 0.762±0.002 | 0.761 |
| `tfidf_nb_nostop` | 0.771 | 0.775±0.002 | 0.782 |
| `bert_NorBERTo` | **0.908** | ∅ | — |
| `bert_bertimbau` | 0.899 | 0.899±0.002 | 0.903 |
| `bert_bertimbau_nostop` | 0.902 | **0.904±0.003** | 0.907 |
| `bert_deb3rta_base` | 0.802 | 0.775±0.011 | 0.773 |
| `bert_deb3rta_base_nostop` | 0.822 | 0.830±0.003 | 0.834 |
| `bert_finbert_ptbr` | 0.890 | 0.893±0.002 | 0.895 |
| `bert_finbert_ptbr_nostop` | 0.901 | 0.902±0.004 | 0.906 |
| `bert_norberto_base_nostop` | 0.901 | 0.899±0.002 | **0.907** |
| `ensemble_agreement` | ∅ | ∅ | ∅ |
| `ensemble_majority` | 0.863 | 0.860±0.005 | 0.864 |
| `ensemble_stacking` | — | — | 0.888 |
| `ensemble_weighted` | 0.873 | 0.873±0.003 | 0.883 |
| `llm_mistral_7b_instruct_v0_3_few_shot` | — | — | 0.640 |
| `llm_mistral_7b_instruct_v0_3_zero_shot` | — | — | 0.579 |
| `llm_qwen2_5_7b_instruct_few_shot` | — | — | 0.632 |
| `llm_qwen2_5_7b_instruct_zero_shot` | — | — | 0.634 |


## 2. In-domain — MULTICLASSE (macro-F1)

| Modelo | fixed_split | cv_5fold (média±dp) | test_set |
|---|---|---|---|
| `tfidf_linearsvc` | 0.894 | 0.891±0.002 | 0.900 |
| `tfidf_linearsvc_nostop` | 0.890 | 0.890±0.002 | 0.899 |
| `tfidf_logreg` | 0.888 | 0.890±0.002 | 0.898 |
| `tfidf_logreg_nostop` | 0.887 | 0.888±0.003 | 0.896 |
| `tfidf_nb` | 0.789 | 0.788±0.003 | 0.788 |
| `tfidf_nb_nostop` | 0.801 | 0.801±0.003 | 0.804 |
| `bert_bertimbau` | 0.917 | 0.886±0.003 | 0.894 |
| `bert_bertimbau_nostop` | **0.927** | **0.928±0.002** | **0.932** |
| `bert_deb3rta_base` | 0.793 | 0.790±0.003 | 0.795 |
| `bert_deb3rta_base_nostop` | 0.864 | 0.862±0.003 | 0.870 |
| `bert_finbert_ptbr` | 0.912 | 0.883±0.003 | 0.890 |
| `bert_finbert_ptbr_nostop` | 0.925 | 0.926±0.002 | 0.930 |
| `bert_norberto_base_nostop` | 0.926 | 0.926±0.001 | ∅ |
| `ensemble_agreement` | ∅ | ∅ | ∅ |
| `ensemble_majority` | 0.896 | 0.894±0.002 | 0.902 |
| `ensemble_stacking` | — | — | 0.923 |
| `llm_mistral_7b_instruct_v0_3_few_shot` | — | — | 0.436 |
| `llm_mistral_7b_instruct_v0_3_zero_shot` | — | — | 0.385 |
| `llm_qwen2_5_7b_instruct_few_shot` | — | — | 0.572 |
| `llm_qwen2_5_7b_instruct_zero_shot` | — | — | 0.482 |


## 3. OOD — BINÁRIO (F1, inferência por dataset)

> Cards OOD são inferência (`regime=test_set`, sem cv/fixed). Vencedor por dataset em **negrito**.

| Modelo | fake_br_ood | fake_br_subgroup_fake | fake_br_subgroup_true | fake_recogna_economia_uol_ood | fake_recogna_uol_balanced | portuguese_news_wikinotices_ood | recognasumm_ood |
|---|---|---|---|---|---|---|---|
| `tfidf_linearsvc` | 0.134 | 0.282 | 0.081 | 0.199 | 0.308 | 0.512 | 0.586 |
| `tfidf_linearsvc_nostop` | 0.135 | 0.286 | 0.082 | 0.195 | 0.276 | 0.498 | 0.585 |
| `tfidf_logreg` | 0.115 | 0.213 | 0.073 | 0.178 | 0.311 | 0.513 | 0.574 |
| `tfidf_logreg_nostop` | 0.116 | 0.220 | 0.071 | 0.185 | 0.467 | 0.578 | 0.581 |
| `tfidf_nb` | 0.146 | 0.280 | 0.086 | 0.351 | 0.576 | **0.659** | 0.617 |
| `tfidf_nb_nostop` | **0.152** | **0.296** | 0.092 | 0.368 | 0.556 | 0.641 | **0.626** |
| `bert_bertimbau` | 0.131 | 0.241 | 0.086 | 0.398 | 0.567 | 0.571 | 0.598 |
| `bert_bertimbau_nostop` | 0.136 | 0.282 | 0.083 | **0.420** | 0.651 | 0.572 | 0.593 |
| `bert_deb3rta_base` | 0.111 | 0.209 | 0.071 | 0.171 | 0.303 | 0.557 | 0.542 |
| `bert_deb3rta_base_nostop` | 0.151 | 0.286 | **0.095** | 0.127 | 0.143 | 0.170 | 0.306 |
| `bert_finbert_ptbr` | 0.119 | 0.261 | 0.069 | 0.354 | 0.661 | 0.627 | 0.603 |
| `bert_finbert_ptbr_nostop` | 0.108 | 0.239 | 0.061 | 0.409 | **0.714** | 0.646 | 0.604 |


## 4. OOD — MULTICLASSE (macro-F1)

| Modelo | fake_br_ood | fake_recogna_economia_uol_ood | fake_recogna_uol_balanced |
|---|---|---|---|
| `tfidf_linearsvc` | 0.169 | 0.187 | 0.165 |
| `tfidf_linearsvc_nostop` | 0.168 | 0.168 | 0.143 |
| `tfidf_logreg` | 0.163 | 0.191 | 0.160 |
| `tfidf_logreg_nostop` | 0.162 | 0.160 | 0.128 |
| `tfidf_nb` | **0.205** | **0.260** | 0.200 |
| `tfidf_nb_nostop` | 0.200 | 0.259 | 0.200 |
| `bert_bertimbau` | 0.141 | 0.192 | 0.186 |
| `bert_bertimbau_nostop` | 0.130 | 0.133 | 0.083 |
| `bert_deb3rta_base` | 0.139 | 0.180 | 0.097 |
| `bert_deb3rta_base_nostop` | — | — | — |
| `bert_finbert_ptbr` | 0.142 | 0.241 | **0.205** |
| `bert_finbert_ptbr_nostop` | 0.147 | 0.186 | 0.136 |


## 5. Notas sobre ausências

- **Por design (`—`):** LLM (4 variantes × tarefa) e `ensemble_stacking` só têm `test_set`. OOD é só inferência (`test_set`).
- **Lacunas reais (`∅`):** `ensemble_agreement` (binário e multiclasse, 6 pastas, 0 cards); `bert_norberto_base_nostop` multiclasse sem `test_set`; `bert_NorBERTo` (legado) `cv_5fold` vazio.
- **Card legado:** `bert_NorBERTo` (binário) é resquício de nomenclatura — o modelo válido é `bert_norberto_base_nostop`. Recomendado remover.
- **Sem linha (não executado):** `ensemble_weighted` foi rodado **só em binário**. `ensemble_agreement`, LLM e stacking não têm execução OOD.
- **Detalhamento e plano de remediação:** `docs/auditoria_result_cards.md`.
