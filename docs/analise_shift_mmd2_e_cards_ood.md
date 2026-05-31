# Análise — Shift MMD²/KTS e Tabela de Result Cards OOD

Análise consolidada de dois artefatos do notebook `45_ood_evaluation.ipynb` (produzidos por `scripts/coverage_aligned_metrics.py` e exportados para o Drive em `economy-classifier/reports/ood_evaluation/`):

- `shift_mmd2_kts_summary.csv` — agregação por OOD do teste de Kernel Two-Sample (MMD² + p-valor permutacional) entre Folha (in-domain) e cada base OOD.
- `ood_cards_table.csv` — tabela "longa" com métricas de cada `result_card.json` por (modelo, tarefa, domínio, nível de partição).

Fonte primária dos números: arquivos CSV citados acima (snapshot de 2026-05-22). Cruzamento com `ood_pivot_binary_f1.csv` e `shift_mmd2_kts_runs.csv` (mesma pasta).

---

## 1. Resumo do teste de shift (MMD² + KTS)

### 1.1 Tabela do `shift_mmd2_kts_summary.csv`

| OOD | MMD² (média) | MMD² (desvio) | p-valor (mediana) | n_reject (de 5) |
|---|---:|---:|---:|---:|
| `folha_self_control` | 0,0009 | 0,0001 | 0,248 | 1 |
| `recognasumm_propor2024` | 0,0201 | 0,0008 | 0,000 | 5 |
| `fake_br` | 0,0372 | 0,0013 | 0,000 | 5 |
| `portuguese_news_wikinotices` | 0,0881 | 0,0027 | 0,000 | 5 |
| `fake_recogna` | 0,1559 | 0,0031 | 0,000 | 5 |

Parâmetros do teste (extraídos de `shift_mmd2_kts_runs.csv`): 5 réplicas por OOD, `n_per_side = 1500`, kernel RBF com σ ∈ [3,10; 3,83] (heurística da mediana).

### 1.2 Leitura estatística

**Sanity check passou.** `folha_self_control` (duas subamostras independentes da própria Folha) dá MMD² ≈ 9 × 10⁻⁴ — três ordens de grandeza abaixo dos OODs reais — e rejeita só 1 de 5 réplicas (≈ 20%, dentro do esperado para H₀ verdadeira a α = 0,05). O teste KTS está calibrado: as rejeições nos demais domínios não são artefato do procedimento.

**Os 4 OODs reais são inequivocamente shifted.** `p_median = 0,000` e `n_reject = 5/5` em todos. Efeito massivo, sem fragilidade entre réplicas. CVs (`std/mean`) ficam entre 1% e 4% — estimativa pontual estável.

**Ranking de covariate shift (input X):**

```
recognasumm_propor2024  →  ~22× o baseline (folha-vs-folha)
fake_br                 →  ~41× o baseline
portuguese_news         →  ~98× o baseline
fake_recogna            → ~173× o baseline
```

### 1.3 Recomendações de redação

- Reportar **média ± desvio** das 5 réplicas (CVs ≤ 4% justificam essa apresentação).
- Incluir explicitamente a linha `folha_self_control` na tabela do artigo — é a evidência de calibração do KTS e fortalece todas as 4 rejeições.
- Justificar a escolha do σ do RBF (médias 3,1–3,8, compatíveis entre domínios — citar "median heuristic" se foi o critério).

---

## 2. Tabela de result cards OOD (`ood_cards_table.csv`)

48 linhas: 6 modelos × 4 domínios × até 3 níveis de partição (binary full, multiclass mapped, e níveis especiais: subgroup_fake/true para Fake.Br; uol_balanced_binary/multiclass para FakeRecogna).

### 2.1 Mapa do recorte

| Domínio (OOD) | Níveis disponíveis | n_eval | Prev. positiva |
|---|---|---:|---:|
| `fake_br_full_texts` | 1_binary_full, 3_subgroup_fake, 3_subgroup_true, 2_multiclass_mapped | 7.200 (3.600 subgrupos) | **0,0061** |
| `fake_recogna_economia_uol` | 1_binary_full, 3_uol_balanced_binary, 2_multiclass_mapped, 3_uol_balanced_multiclass | 11.872 (624 balanced) | 0,0263 / **0,5** |
| `portuguese_news_wikinotices` | 1_binary_full, 2_multiclass | 9.135 | 0,126 |
| `recognasumm_propor2024` | 1_binary_full, 2_multiclass | **135.272** | 0,0932 |

In-domain de referência (Folha): prevalência ≈ 0,125. Apenas `portuguese_news` (0,126) e `recognasumm` (0,0932) preservam prevalência próxima ao treino. Fake.Br e FakeRecogna (versão full) sofrem **label shift severo**.

### 2.2 F1 binário consolidado (level `1_binary_full_corpus`) versus MMD²

| OOD | MMD² | F1 médio (6 modelos) | Melhor modelo (F1) | AUC médio |
|---|---:|---:|---|---:|
| `recognasumm` | 0,020 | **0,592** | `bertimbau` (0,617) | 0,913 |
| `fake_br` | 0,037 | 0,118 | `tfidf_nb` (0,146) | 0,812 |
| `portuguese_news` | 0,088 | **0,570** | `tfidf_nb` (0,659) | 0,880 |
| `fake_recogna` | 0,156 | 0,284 | `finbert` (0,412) | 0,852 |

`recognasumm` (menor MMD²) e `portuguese_news` (MMD² ~4× maior) têm F1 quase idêntico. `fake_br` e `fake_recogna` desabam — mas o que difere entre as duas duplas é a **prevalência da classe positiva**, não o MMD². **AUC é muito mais estável que F1 OOD** (0,81–0,91 em todos os domínios full): ordenação se preserva, problema está no threshold.

### 2.3 Achados cross-domain

#### TF-IDF NB é surpreendentemente robusto OOD

| OOD / level | Líder | F1 / macro-F1 |
|---|---|---:|
| `fake_br` binary full | **tfidf_nb** | 0,146 |
| `fake_br` subgroup_fake | tfidf_linearsvc ≈ **tfidf_nb** | 0,28 |
| `fake_br` multiclass | **tfidf_nb** | macro_f1 = 0,205 |
| `portuguese_news` binary | **tfidf_nb** | 0,659 |
| `recognasumm` binary | bertimbau ≈ finbert ≈ **tfidf_nb** | 0,617 |
| `recognasumm` multiclass | **tfidf_nb** | — |
| `fake_recogna` multiclass full | **tfidf_nb** | 0,260 |

Hipótese: o viés indutivo simples (independência condicional) e a capacidade limitada de NB fazem o modelo **subajustar idiossincrasias da Folha**. BERTs e modelos discriminativos sobrecarregados de capacidade memorizam estilo da Folha melhor in-domain, e pagam caro OOD. Contradiz a intuição "modelo mais expressivo → melhor sempre".

#### DeB3RTa underperforma em quase todos os OODs

| OOD | DeB3RTa F1 | Posição entre BERTs | Posição geral |
|---|---:|---|---|
| `fake_br` binary | 0,111 | 1º | 4º |
| `fake_recogna` binary | **0,171** | **3º (último BERT)** | **5º** |
| `fake_recogna` balanced | **0,303** | **3º** | **5º** |
| `portuguese_news` | 0,557 | 3º | 5º |
| `recognasumm` | 0,542 | 3º | 6º |

DeB3RTa foi pré-treinado em domínio financeiro PT-BR — era a aposta natural para "mercado". Mas a especialização pré-treinamento **degrada generalização OOD**. BERTimbau (PT-BR geral) e FinBERT-PT-BR são mais robustos. Argumento citável contra a expectativa naive de que "modelo de domínio resolve sempre".

#### Subgrupo `fake` >> `true` em Fake.Br

| Métrica | subgroup_fake | subgroup_true | Razão |
|---|---:|---:|---:|
| F1 médio (6 modelos) | 0,240 | 0,074 | **3,2×** |
| AUC médio | 0,912 | 0,748 | 1,22× |

O classificador "mercado vs outros" identifica `mercado` **muito melhor em notícias fake do que em notícias jornalisticamente sérias** do Fake.Br. Hipótese: fake news tendem a vocabulário estereotipado por tópico; o jornalismo profissional (subgrupo `true`) tem registro próximo ao da Folha e confunde mais o classificador binário.

#### Calibração colapsa no balanced

| Run | Brier médio | ECE médio |
|---|---:|---:|
| `fake_recogna` binary full (prev 0,026) | 0,035 | 0,043 |
| `fake_recogna` **balanced** (prev 0,5) | **0,279** | **0,316** |

Uma ordem de grandeza a mais. Calibração in-domain (otimizada para prev ≈ 0,125) **não sobrevive a mudança de prevalência**. Evidência de que `CalibratedClassifierCV` e softmax-temperatura do BERT são frágeis a label shift.

#### Macro-F1 versus macro_f1_present_only no multiclasse

| Run | macro_f1 (8 classes) | present_only | Δ |
|---|---:|---:|---:|
| `fake_br` multiclass | 0,14–0,20 | 0,22–0,33 | ~1,6× |
| `fake_recogna` mc full | 0,18–0,26 | 0,24–0,35 | ~1,3× |
| `fake_recogna` mc balanced | 0,10–0,21 | 0,26–0,55 | **~2,6×** |

Em OOD, classes da Folha como `colunas` e `ilustrada` não aparecem ou aparecem em volume ínfimo. Reportar só `macro_f1` puxa a média para baixo de forma artificial (classes ausentes contribuem zero). **Sempre reportar `macro_f1_present_only` ao lado em runs OOD multiclasse**, explicitando o conjunto de classes efetivamente avaliado.

---

## 3. Síntese: MMD² × F1 com label shift incluído

Cruzamento dos dois artefatos com magnitude de label shift |p_OOD − 0,125|:

| OOD | MMD² | |Δprev| | F1 médio OOD | F1 in-domain (~0,85) | Drop |
|---|---:|---:|---:|---:|---:|
| `recognasumm` | 0,020 | 0,032 | 0,592 | 0,85 | −30% |
| `fake_br` | 0,037 | **0,119** | 0,118 | 0,85 | **−86%** |
| `portuguese_news` | 0,088 | 0,001 | 0,570 | 0,85 | −33% |
| `fake_recogna` | 0,156 | 0,099 | 0,284 | 0,85 | −67% |

**O preditor do drop não é MMD² isoladamente — é MMD² combinado com a magnitude do label shift.** Onde o label shift é mínimo (`portuguese_news`, `recognasumm`), o drop é controlado (~30%) mesmo com MMD² alto. Onde há label shift forte (`fake_br`, `fake_recogna`), o drop explode independentemente do MMD².

Para a dissertação: MMD² é um dos componentes do diagnóstico OOD, mas **insuficiente sozinho**. Propor reportar a tripla `(MMD², |Δprev|, AUC OOD)` como diagnóstico mais informativo que F1 OOD isolado.

---

## 4. Recomendações de redação para o paper

1. **Reorganizar a tabela OOD** por dimensão de shift: agrupar `recognasumm` + `portuguese_news` (covariate shift puro, label preservado) versus `fake_br` + `fake_recogna` (covariate + label shift). Hoje o pivot está em ordem alfabética.
2. **Reportar AUC junto de F1** em OOD — AUC é estável (0,78–0,94 em quase tudo), F1 varia até 6× entre runs. O contraste é informativo.
3. **Adicionar coluna "label shift" (Δprev) na tabela final** — explica a variância de F1 OOD que o MMD² não explica.
4. **Destacar os 3 negative results contra-intuitivos:**
   - TF-IDF NB ≥ BERTs em OOD;
   - DeB3RTa (especializado) < BERTimbau (geral) OOD;
   - F1 melhor em fake news do que em jornalismo real (subgrupos Fake.Br).
5. **Reportar calibração explicitamente em runs balanceadas** — a degradação Brier/ECE é parte do diagnóstico, não detalhe técnico.
6. **Sempre apresentar `macro_f1_present_only` ao lado de `macro_f1`** em multiclasse OOD, com nota explicando a diferença.
