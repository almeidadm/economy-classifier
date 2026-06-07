# Auditoria dos `result_card.json` — Guarda Metodológica

**Data:** 2026-06-04
**Escopo:** 214 result cards espelhados do Google Drive (`runs/` + `ood_runs/`) para `artifacts/drive_audit/`.
**Ferramenta:** `src/economy_classifier/card_audit.py` (lógica pura, 19 testes) + CLI `scripts/audit_result_cards.py`.
**Relatório-máquina:** `artifacts/analysis/audit_drive.json`.
**Reproduzir:**

```bash
uv run python scripts/audit_result_cards.py \
    artifacts/drive_audit/runs artifacts/drive_audit/ood_runs \
    --json artifacts/analysis/audit_drive.json
```

A CLI sai com código `!= 0` enquanto houver `ERROR` — serve como *gate* antes de montar a tabela final do artigo.

---

## 1. Resumo executivo

| Severidade | Total | Onde |
|---|---:|---|
| **ERROR** | 3 | in-domain (`runs/`) |
| **WARN** | 43 | in-domain (`runs/`) |
| **INFO** | 117 | OOD (`ood_runs/`) — apenas marcação informativa |

- **214 cards lidos, 0 JSON inválido.** Cobertura: 97 in-domain + 117 OOD.
- **Trilha OOD: limpa.** Nenhuma violação de guarda. Cards OOD são *inferência* (modelo treinado in-domain aplicado a base com shift): `regime="test_set"`, `hyperparameter_search=null` e contexto em `config.evaluation_type` (`out_of_domain*`) — tudo por construção, não é erro.
- **Trilha in-domain:** 3 erros bloqueantes + 2 padrões sistemáticos de aviso (calibração e cobertura de LLM).
- **Achado paralelo (fora do auditor):** 9 pastas de run existem no Drive **sem nenhum `result_card.json`** — ver §4.

Além disso, foi corrigido um **falso-positivo do próprio auditor** sobre cards OOD (ver §6) — descoberto justamente por rodar sobre os dados reais.

---

## 2. O que cada checagem verifica (ligação com a guarda)

| Código | Severidade | Invariante da guarda (CLAUDE.md) |
|---|---|---|
| `missing-regime` (cv_5fold) | ERROR | "F1 único sem variância da CV" — sem o card `cv_5fold` não há média ± desvio |
| `missing-regime` (fixed_split/test_set) | WARN | Cobertura incompleta dos 3 regimes |
| `tfidf-missing-hp` | ERROR | "Pular a busca e treinar com defaults" — TF-IDF sem `hyperparameter_search` não entra na tabela |
| `bad-scoring` | ERROR | "TF-IDF binário com `scoring=f1_macro`" — binário deve usar `f1`; multiclasse `f1_macro` |
| `unexpected-hp` | WARN | BERT/LLM têm config fixa da literatura → `hyperparameter_search=null` esperado |
| `cv-no-variance` | ERROR | Card `cv_5fold` sem chave `*_std` |
| `llm-coverage` | WARN | "Comparar LLM sem alinhar coverage" — `coverage<1.0` ⇒ subconjunto |
| `llm-auc` | WARN | "AUC-ROC de LLM como evidência" — scores determinísticos ⇒ AUC = `N/A` |
| `no-calibration` | WARN | Brier/ECE são reportáveis para TF-IDF/BERT calibrado |

Cards OOD são detectados por `config.evaluation_type` começando com `out_of_domain` e têm as regras de regime/HP **relaxadas** (são inferência, não treino).

---

## 3. ERROS bloqueantes (3)

> Estes impedem a entrada da linha correspondente na tabela final enquanto não resolvidos. A CLI sai com código 1.

### E1 — `ensemble_stacking` sem CV nem fixed_split (binário **e** multiclasse)

- **Cards presentes:** apenas `ensemble_stacking_binary_test_set` e `ensemble_stacking_multiclass_test_set`.
- **Faltam:** `cv_5fold` (⇒ ERROR) e `fixed_split` (⇒ WARN) em ambas as tarefas.
- **Contraste:** os demais ensembles têm os 3 regimes —
  `ensemble_majority` (binário e multiclasse) e `ensemble_weighted` (binário): `['cv_5fold', 'fixed_split', 'test_set']`.
- **Consequência metodológica:** o stacking só tem um ponto único de `test_set`, **sem variância de CV**. Reportá-lo lado a lado com modelos que têm média ± desvio é o anti-padrão "F1 único sem variância da CV".
- **Causa provável (a confirmar):** o meta-classificador do stacking é treinado na validação (para evitar *leakage*); aplicar isso dentro de cada fold da CV exige um esquema aninhado que pode não ter sido implementado.
- **Decisão necessária:** (a) rodar os regimes `cv_5fold`/`fixed_split` do stacking com CV aninhada; **ou** (b) declarar explicitamente na metodologia que o stacking é reportado só no `test_set` (ponto único) e mantê-lo fora de qualquer comparação que exija variância.

### E2 — `bert_NorBERTo_binary` legado, só `fixed_split`

- **Card presente:** apenas `bert_NorBERTo_binary_fixed_split`.
- **Faltam:** `cv_5fold` (⇒ ERROR) e `test_set` (⇒ WARN).
- **Diagnóstico:** é um **resquício de nomenclatura antiga**. O modelo NorBERTo definitivo é `bert_norberto_base_nostop`, que tem cobertura binária completa (`fixed_split` + `cv_5fold` + `test_set`, todos presentes e válidos).
- **Decisão necessária:** **remover** a pasta legada `bert_NorBERTo_binary_fixed_split` (e a vazia `bert_NorBERTo_binary_cv_5fold`, §4) do conjunto de resultados. Não é caso de "rodar mais": é card órfão que polui a auditoria.

---

## 4. Pastas de run **sem `result_card.json`** (9)

> Estas não aparecem no auditor (sem card = nada a auditar), mas são lacunas reais detectadas ao cruzar a árvore de pastas do Drive com os cards baixados.

| Pasta (em `runs/`) | Interpretação |
|---|---|
| `ensemble_agreement_binary_{cv_5fold,fixed_split,test_set}` | **Ensemble `agreement` não produziu nenhum card** (3 regimes vazios) |
| `ensemble_agreement_multiclass_{cv_5fold,fixed_split,test_set}` | idem, multiclasse (3 regimes vazios) |
| `bert_norberto_base_nostop_multiclass_test_set` | **Falta o `test_set` do NorBERTo multiclasse** (tem cv+fixed) — ver W4 |
| `bert_NorBERTo_binary_cv_5fold` | Pasta legada vazia (par do E2) — remover |
| `llm_few_shot_examples` | **Não é run** — é o holder dos exemplos few-shot; ausência de card é esperada |

**Implicações:**
- O ensemble **`agreement` está totalmente ausente dos resultados** (6 pastas, 0 cards). Decidir se entra no artigo (e então rodá-lo) ou se sai de escopo.
- O **NorBERTo multiclasse não tem número de teste** — o `test_set` precisa ser gerado se o modelo for reportado nessa tarefa.

---

## 5. AVISOS sistemáticos (43)

### W1 — Calibração ausente in-domain (27 cards `no-calibration`)

- **Fato:** **0 de 97** cards in-domain têm `brier`/`ece`. O auditor sinaliza ativamente os 27 cards binários em regime pontual (`fixed_split`/`test_set`) de TF-IDF e BERT; a ausência, porém, é **universal** (também nos multiclasse e nos `cv_5fold`).
- **Contraste gritante:** **todos os 117 cards OOD têm `brier` e `ece`.** Ou seja, o código de calibração existe e rodou para OOD, mas os cards in-domain **nunca foram reemitidos** com essas métricas.
- **Guarda:** Brier e ECE são listados como reportáveis sempre que `y_score` for probabilidade calibrada (TF-IDF + `CalibratedClassifierCV`, BERT softmax).
- **Decisão necessária:** *backfill* — recomputar Brier/ECE a partir dos `predictions.csv` + `y_score` já existentes (barato, não exige re-treino) **ou** declarar calibração fora de escopo in-domain. A inconsistência in-domain × OOD não pode ficar no artigo sem justificativa.

Cards sinalizados:

```
tfidf_{logreg,linearsvc,nb}[_nostop]_binary_{fixed_split,test_set}     (12)
bert_{bertimbau,deb3rta_base,finbert_ptbr}[_nostop]_binary_{fixed_split,test_set}  (...)
bert_norberto_base_nostop_binary_{fixed_split,test_set}
bert_NorBERTo_binary_fixed_split   (legado — será removido, ver E2)
```

### W2 — LLM com `coverage < 1.0` (8 cards `llm-coverage`)

Quando o LLM gera resposta não-parseável, a amostra é descartada. `n_eval_samples = 16.629` (test completo) em todos; o subconjunto efetivamente avaliado é `coverage × 16.629`.

| Card | coverage | avaliadas ≈ | descartadas ≈ |
|---|---:|---:|---:|
| `mistral_v0_3_multiclass_few_shot` | **0,7149** | 11.888 | **4.741** |
| `qwen2_5_multiclass_few_shot` | 0,8835 | 14.692 | 1.937 |
| `mistral_v0_3_binary_few_shot` | 0,9518 | 15.827 | 802 |
| `mistral_v0_3_multiclass_zero_shot` | 0,9518 | 15.827 | 802 |
| `qwen2_5_binary_few_shot` | 0,9764 | 16.237 | 392 |
| `mistral_v0_3_binary_zero_shot` | 0,9835 | 16.355 | 274 |
| `qwen2_5_multiclass_zero_shot` | 0,9872 | 16.416 | 213 |
| `qwen2_5_binary_zero_shot` | 0,9907 | 16.474 | 155 |

- **Pior caso:** Mistral multiclasse few-shot avalia só **71,5%** do teste.
- **Sub-achado:** `n_eval_samples` permanece 16.629 (atemptado) mesmo com `coverage<1`, em vez de refletir o nº parseado — diverge do contrato descrito na guarda ("`n_eval_samples` reflete apenas as amostras que passaram"). Conferir na conversão `predictions.csv`.
- **Decisão necessária:** ao comparar LLM × TF-IDF/BERT, **restringir aos mesmos índices** (já existe `scripts/coverage_aligned_metrics.py`) **ou** marcar explicitamente que o LLM foi avaliado em subconjunto. Comparação direta sem alinhar é o anti-padrão de coverage.

### W3 — AUC-ROC determinística de LLM (4 cards `llm-auc`)

Os 4 cards binários de LLM reportam `auc_roc`, mas o `y_score` do pipeline LLM é determinístico (`1.0`/`0.0`) ⇒ a curva ROC tem um único ponto operacional e não mede separabilidade.

- **Cards:** `{mistral_v0_3, qwen2_5}_binary_{zero_shot, few_shot}_test_set`.
- **Decisão necessária:** no artigo, marcar AUC de LLM como **`N/A — deterministic scores`** e nunca compará-la com AUC de TF-IDF/BERT.

### W4 — `bert_norberto_base_nostop` multiclasse sem `test_set` (1 `missing-regime` WARN)

- Tem `cv_5fold` + `fixed_split`, mas **não tem `test_set`** (pasta existe vazia, §4).
- **Consequência:** o NorBERTo multiclasse não tem número final de teste. Se for reportado nessa tarefa, rodar o `test_set`; senão, declarar fora de escopo.

---

## 6. Nota de transparência — correção no próprio auditor

Na primeira passada sobre os dados reais, o auditor emitiu **83 ERROS falsos** na trilha OOD: assumia que cards OOD teriam `regime = <nome do dataset>`, mas eles usam `regime="test_set"` com `hyperparameter_search=null` (são inferência). Isso disparava `missing-regime` (cv/fixed "ausentes") e `tfidf-missing-hp` indevidamente.

**Correção:** detector `is_ood_card()` baseado em `config.evaluation_type`; cards OOD ficam isentos das regras de cobertura de regimes e de busca de HP, mas mantêm `bad-scoring` (caso vaze payload com scoring errado) e checagem de métrica primária. Coberto por 4 testes novos. Após a correção, a trilha OOD ficou 100% limpa.

---

## 7. O que está correto (validações que passaram)

- **TF-IDF** (logreg/linearsvc/nb, original e `_nostop`, binário e multiclasse): 3 regimes completos, `scoring` correto (`f1` binário / `f1_macro` multiclasse), `hyperparameter_search` populado, `cv_5fold` com `*_std`.
- **BERT** (bertimbau/finbert_ptbr/deb3rta_base, original e `_nostop`, binário e multiclasse): 3 regimes completos, `hyperparameter_search=null` (exceção declarada da guarda), `cv_5fold` com `*_std`.
- **NorBERTo `_nostop` binário:** cobertura completa.
- **Ensembles `majority` e `weighted`:** 3 regimes (exceto `agreement`, §4, e `stacking`, E1).
- **OOD (117 cards):** todos válidos, com Brier/ECE, métrica primária presente, scoring coerente.

A matriz completa de cobertura está no cabeçalho da saída da CLI e no JSON.

---

## 8. Plano de remediação priorizado

| # | Item | Ação | Esforço | Bloqueia tabela final? |
|---|---|---|---|---|
| 1 | E2 + pasta legada | Remover `bert_NorBERTo_binary_*` (legado) | trivial | **Sim** (1 ERROR) |
| 2 | W1 calibração | *Backfill* Brier/ECE in-domain a partir dos `predictions.csv` | baixo | Recomendado |
| 3 | E1 stacking | Decidir: rodar CV aninhada **ou** declarar ponto único | médio / decisão | **Sim** (2 ERROR) |
| 4 | W2 coverage | Alinhar índices LLM×TF-IDF/BERT (`coverage_aligned_metrics.py`) | baixo | Sim (p/ comparação LLM) |
| 5 | W3 AUC LLM | Marcar AUC de LLM como `N/A` no artigo | trivial | Não (relato) |
| 6 | §4 agreement | Decidir se `ensemble_agreement` entra (e rodar) ou sai de escopo | médio / decisão | Não |
| 7 | W4 NorBERTo multi | Rodar `test_set` ou declarar fora de escopo | médio / decisão | Não |

---

## 9. Proveniência

- Cards in-domain gerados em 6 commits distintos: `61c71dc` (35), `d9c449b` (18), `c88a40b` (18), `8860642` (11), `9ca8c84` (10), `59cec77` (5). Heterogeneidade esperada (runs incrementais); relevante apenas se algum card pré-datar uma correção de guarda.
- Espelho local read-only dos cards: `artifacts/drive_audit/{runs,ood_runs}/<run>/result_card.json` (+ stub `predictions.csv` apenas para a checagem de existência).
- Relatório-máquina: `artifacts/analysis/audit_drive.json`.

> **Aviso:** este documento audita **apenas os `result_card.json`**. Verificações que dependem de artefatos separados (McNemar com Bonferroni, alinhamento real de coverage nas tabelas agregadas) exigem os CSVs de `reports/` e ficam para uma rodada subsequente.

---

## 10. Re-verificação direta no Drive — 2026-06-06

Varredura ao vivo do Drive (MCP) + merge dos cards alterados sobre o espelho + re-execução do auditor (`/tmp/audit_drive_20260606.json`). **Estado: 215 cards, ERROR=3, WARN=42, INFO=117.**

### W4 RESOLVIDO — NorBERTo re-run completo

O NorBERTo (`bert_norberto_base_nostop`) foi integralmente re-rodado em 05–06/06 (Colab `21_bert(norberto).ipynb`), incluindo o `multiclass_test_set` que faltava (card gerado 06/06 07:09). Cobertura agora completa: 6/6 regimes na config canônica (`max_length=500`, 3 epochs, `seed=2026`, commit `59cec77`, `hyperparameter_search=null` — exceção BERT correta). CV com `*_std` presente.

| Tarefa | fixed_split | cv_5fold | test_set |
|---|---|---|---|
| binário (F1) | 0.8973 | 0.8994 ± 0.0034 | **0.9063** (AUC 0.9904) |
| multiclasse (macro-F1) | 0.9265 | 0.9263 ± 0.0020 | **0.9325** (mercado 0.9152) |

A geração supersedida (03–04/06, **mesma config** — supersede temporal, não mudança de hiperparâmetros) foi arquivada corretamente em `runs_archive/` com sufixo `__<timestamp>Z`. Drift métrico entre gerações ≤ 0.005 em todas as métricas — reprodutibilidade do re-treino confirmada (ruído de GPU).

### ERRORs remanescentes (3) — inalterados

- **E1** `ensemble_stacking` binário + multiclasse sem `cv_5fold` (decisão pendente: CV aninhada ou declarar ponto único).
- **E2** legado `bert_NorBERTo_binary_fixed_split` (card 4000 tokens/5 epochs, 26/05) **ainda no Drive**, com `model/` + `checkpoints/` pesados; `bert_NorBERTo_binary_cv_5fold` contém só `fold_0/` abandonado. Remover ambas.

### Lacunas e staleness confirmados na varredura

- `ensemble_agreement`: 6 pastas, 0 cards (decisão de escopo pendente — item §8.6).
- **Geração não-`nostop` BERT inconsistente**: cards misturam (512 tokens, 5 ep, 25–26/05) e (128 tokens, 3 ep, 29–30/04) *dentro do mesmo modelo/tarefa* (ex.: `bertimbau_multiclass`: fixed em 512/5ep, cv+test em 128/3ep). Nenhum na config canônica 500/3ep — re-treino do NB 21 §8.5 pendente para bertimbau/finbert/deb3rta.
- **OOD: 117/117 cards a `max_length=128`** — nenhum re-rodado desde 04/06; todos obsoletos pela decisão de inferência a 500. Re-run pendente.
- **Ensembles (11 cards, 04/05)**: membros incluem `bert_bertimbau` de geração anterior aos re-treinos — ficam stale a cada re-treino de membro; re-rodar após estabilizar a geração BERT.
- Pasta órfã `runs_` no Drive (backup de abril, 58 cards antigos) — fora do escopo da auditoria, candidata a limpeza.
- 12 pastas `tfidf_*_search_*` sem card são **esperadas** (contêm `search_result.json` da busca de HP, não são runs de avaliação).
