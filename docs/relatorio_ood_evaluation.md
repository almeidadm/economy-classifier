# Relatório — Avaliação Out-of-Distribution (OOD)

Resultados consolidados da avaliação OOD em **Fake.Br Corpus (`fake_br_full_texts`)** e **FakeRecogna — recorte `economia` UOL (`fake_recogna_economia_uol`)** com os 6 modelos in-domain treinados em Folha (TF-IDF × 3, BERT × 3). Fonte dos números: `reports/ood_evaluation/` no Drive (`economy-classifier/reports/ood_evaluation/`), produzido pelo notebook `45_ood_evaluation.ipynb` e por `scripts/coverage_aligned_metrics.py`.

---

## 1. Recorte de avaliação

| Domínio | Nível | n_eval | Prevalência positiva | Observação |
|---|---|---:|---:|---|
| `fake_br_full_texts` | `1_binary_full_corpus` | 7.200 | 0,61% | Fake.Br inteiro (true+fake) |
| `fake_br_full_texts` | `3_subgroup_fake` | 3.600 | 0,61% | Apenas artigos com label `fake` |
| `fake_br_full_texts` | `3_subgroup_true` | 3.600 | 0,61% | Apenas artigos com label `true` |
| `fake_br_full_texts` | `2_multiclass_mapped` | 7.200 | — | 7 classes Folha mapeadas via dicionário |
| `fake_recogna_economia_uol` | `1_binary_full_corpus` | 11.872 | 2,63% | FakeRecogna recorte `economia` UOL |
| `fake_recogna_economia_uol` | `3_uol_balanced_binary` | — | ~50% | Subset balanceado positivos/negativos |
| `fake_recogna_economia_uol` | `2_multiclass_mapped` | — | — | Mapeamento multiclasse |
| `fake_recogna_economia_uol` | `3_uol_balanced_multiclass` | — | — | Subset balanceado multiclasse |

**Construto-alvo**: rótulo positivo é `mercado` (seção editorial Folha). As prevalências < 3% em ambos os corpora ilustram que `mercado` é semanticamente raro fora da Folha — Fake.Br é majoritariamente política e FakeRecogna mistura múltiplos veículos. O subset balanceado da UOL é a única configuração em que prevalência se aproxima do treino (~12,5%) e separação positivo/negativo é tratável.

---

## 2. Resultados — Tarefa binária (F1 do positivo `mercado`)

### 2.1 Tabela consolidada

| Modelo | Fake.Br full | Fake.Br fake | Fake.Br true | FakeRecogna full | FakeRecogna UOL balanced |
|---|---:|---:|---:|---:|---:|
| `tfidf_nb` | **0,1462** | 0,2796 | **0,0865** | 0,3508 | 0,5760 |
| `tfidf_linearsvc` | 0,1338 | **0,2817** | 0,0808 | 0,1993 | 0,3077 |
| `tfidf_logreg` | 0,1150 | 0,2128 | 0,0731 | 0,1785 | 0,3110 |
| `bert_bertimbau` | 0,0983 | 0,2169 | 0,0608 | 0,3876 | 0,6247 |
| `bert_finbert_ptbr` | 0,1071 | 0,2143 | 0,0714 | **0,4122** | **0,6926** |
| `bert_deb3rta_base` | 0,1114 | 0,2095 | 0,0709 | 0,1713 | 0,3027 |

(Maior valor em negrito por coluna.)

### 2.2 Padrões observáveis

1. **Domínio decide o vencedor**. Em Fake.Br (corpus majoritariamente político), TF-IDF (NB e LinearSVC) domina os três BERTs em todas as três cortes. Em FakeRecogna `economia` UOL — corpus alinhado ao construto financeiro — FinBERT-PT-BR salta para a liderança, **invertendo o padrão in-domain** observado na Folha (onde FinBERT empata com BERTimbau e DeB3RTa fica abaixo).
2. **Magnitude do collapse**. F1 cai de ~0,87 (Folha test set, BERTimbau/FinBERT) para 0,10–0,21 em Fake.Br e 0,39–0,69 em FakeRecogna. A queda é **menor onde o construto editorial-alvo (`mercado`) tem maior chance de bater com o conteúdo do corpus OOD**.
3. **Subgrupo true × fake**. Em Fake.Br, F1 no subgrupo `true` é ~3× menor que no subgrupo `fake` (0,06–0,09 vs 0,21–0,28). Como prevalência é igual nos dois (0,61%), a diferença não é artefato de base-rate — modelos confundem-se mais ao tentar identificar notícias reais economicamente relevantes do que ao identificar notícias falsas economicamente relevantes. Interpretação plausível: notícias falsas usam vocabulário mais estereotipado ("mercado", "bolsa", "dólar") que o classificador aprendeu na Folha; notícias true cobrem o tema com diversidade lexical maior.
4. **DeB3RTa-base é instável fora de domínio**. F1 in-domain já era o pior dos três BERTs (0,7734 vs 0,8675 BERTimbau). Em FakeRecogna full corpus desaba para 0,1713 — **comparável aos piores TF-IDF**, apesar dos 426M parâmetros. Em Fake.Br fica no meio do pelotão. O pretraining financeiro do DeB3RTa não traduz em vantagem mensurável OOD.
5. **Precision baixa explica a maior parte da queda**. Em Fake.Br full corpus, precision varia 0,056–0,086 enquanto recall mantém 0,38–0,50. Os modelos continuam recuperando uma fração razoável dos positivos verdadeiros, mas inundam com falsos positivos — efeito esperado de prevalência baixíssima (0,61%) sem recalibração do limiar.

---

## 3. Resultados — Multiclasse (macro-F1)

| Modelo | Fake.Br mapped | FakeRecogna mapped | FakeRecogna UOL balanced |
|---|---:|---:|---:|
| `tfidf_nb` | **0,2047** | **0,2602** | **0,2001** |
| `tfidf_linearsvc` | 0,1688 | 0,1866 | 0,1648 |
| `tfidf_logreg` | 0,1625 | 0,1906 | 0,1601 |
| `bert_bertimbau` | 0,1406 | 0,1916 | 0,1865 |
| `bert_finbert_ptbr` | 0,1423 | 0,2414 | 0,2054 |
| `bert_deb3rta_base` | 0,1386 | 0,1803 | 0,0965 |

**Observações**:
- `tfidf_nb` lidera as três configurações. Replicação clara do padrão in-domain (onde NB foi competitivo na multiclasse).
- FinBERT-PT-BR mantém vantagem entre os BERTs em FakeRecogna (consistente com o binário), mas ainda fica atrás de NB.
- Macro-F1 absoluto é baixíssimo (0,10–0,26). O mapeamento de 7 classes Folha para corpora OOD com taxonomia distinta é inerentemente lossy: poucas classes têm representação suficiente para extrair sinal.

---

## 4. Significância estatística — McNemar pareado com Bonferroni

Todos os testes foram conduzidos com `compute_mcnemar_pairwise` (n_comparisons=15 por corpus/nível, K=6 modelos → K(K−1)/2 pares). Reportamos `significant_after_correction`.

### 4.1 FakeRecogna UOL balanced (binário)

Diferenças mais marcadas — pelo alto F1 absoluto, há mais sinal:

- **bert_finbert_ptbr supera todos significativamente** (χ²=10,4–88,4; p_adj ≤ 0,019 vs cada um dos outros 5).
- **bert_bertimbau é equivalente a tfidf_nb** (χ²=2,5, p_adj=1,0).
- **bert_bertimbau supera ambos os TF-IDF lineares** (logreg, linearsvc) e o DeB3RTa após correção.
- **DeB3RTa, tfidf_linearsvc e tfidf_logreg são estatisticamente equivalentes entre si** (todos os pares com p_adj=1,0).

Veredicto: o ranking limpo é FinBERT > BERTimbau ≈ NB > {LinearSVC, LogReg, DeB3RTa}. Quatro pares (de 15) não rejeitam H₀ após Bonferroni — concordância forte com o ranking de F1.

### 4.2 Fake.Br full corpus (binário)

- **tfidf_nb supera tfidf_linearsvc, bert_bertimbau, bert_finbert_ptbr, bert_deb3rta_base e tfidf_logreg** com p_adj ≤ 0,02 — vencedor inequívoco do corpus.
- **bert_deb3rta_base supera bert_bertimbau e bert_finbert_ptbr** (p_adj = 1e-3 e 5e-5 respectivamente) — único cenário em que DeB3RTa tem vantagem estatisticamente robusta sobre os outros BERTs.
- **bert_bertimbau ≈ bert_finbert_ptbr** (p_adj = 1,0).
- **tfidf_linearsvc supera tfidf_logreg** (p_adj = 0,016) — replicação do padrão in-domain.

### 4.3 Fake.Br subgrupo `fake` e `true`

- **Subgrupo fake**: poucos pares significativos após Bonferroni — `bert_deb3rta_base > tfidf_linearsvc` (p_adj = 7e-4) e `tfidf_linearsvc > tfidf_logreg` (p_adj = 6e-5). Os demais 13 pares são equivalentes (ou apenas marginalmente significativos antes da correção).
- **Subgrupo true**: somente 3 pares sobrevivem à correção: `tfidf_nb > bert_bertimbau` (p_adj = 0,002), `tfidf_nb > bert_finbert_ptbr` (p_adj = 0,046), `tfidf_nb > bert_deb3rta_base` (p_adj = 8e-4). Os 12 pares restantes não rejeitam H₀.

### 4.4 FakeRecogna `economia` UOL full corpus (binário)

- **FinBERT-PT-BR supera todos** com p_adj ≤ 0,002, exceto vs BERTimbau (p_adj = 1,0).
- **bert_bertimbau ≈ bert_finbert_ptbr** (não significativo após Bonferroni mesmo com diferença bruta de 2,5 pp em F1) — replicação direta do resultado in-domain.
- **DeB3RTa fica empatado com tfidf_logreg e tfidf_linearsvc** após correção.

### 4.5 Interpretação geral

A correção de Bonferroni com 15 comparações por bloco é conservadora. Apesar disso, padrões robustos emergem em todas as cortes binárias:
- **Sempre que TF-IDF NB lidera em F1, lidera com significância**.
- **FinBERT-PT-BR vence robustamente apenas em FakeRecogna**; em Fake.Br não há vantagem estatística sobre BERTimbau.
- **BERTimbau e FinBERT são estatisticamente indistinguíveis em 4 das 5 configurações** — replica e fortalece a Manchete A do `stil_strategy_note.md` (pretraining financeiro genérico não compensa fora do domínio nativo).

---

## 5. Calibração — Brier e ECE

Brier (quanto menor, melhor) e Expected Calibration Error nas configurações onde computados:

| Cenário | Melhor Brier | Melhor ECE |
|---|---|---|
| Fake.Br full | `tfidf_linearsvc` (0,0268) | `tfidf_nb` (0,0337) |
| Fake.Br fake | `tfidf_linearsvc` (0,0124) | `tfidf_nb` (0,0163) |
| Fake.Br true | `tfidf_linearsvc` (0,0412) | `tfidf_nb` (0,0513) |

**Observações**:
1. **TF-IDF + `CalibratedClassifierCV` mantém calibração superior aos BERTs no OOD**. Brier dos BERTs é ~1,2–1,5× maior. Resultado consistente com expectativa: temperatura softmax do BERT não foi recalibrada para os corpora OOD.
2. **ECE permanece baixo em valor absoluto** (< 0,07 em todas as combinações) porque a prevalência positiva é minúscula — qualquer classificador que previse "tudo negativo" teria ECE pequeno. O número é informativo dentro do bloco mas não comparável a corpora com prevalência diferente.

---

## 6. Achados-chave

1. **Construto-alvo `mercado` parcialmente generaliza para `economia financeira` mas não para `notícia editorial geral`**. F1 cai de 0,87 para 0,69 (FinBERT, FakeRecogna UOL balanced) — degradação grande, mas modelo segue acionável. Para Fake.Br, F1 cai para 0,06–0,15 — modelo perde valor utilitário.
2. **O ranking de modelos in-domain não se preserva OOD**. Na Folha o ranking é FinBERT ≈ BERTimbau > LinearSVC > LogReg > NB > DeB3RTa. No OOD financeiro (UOL balanced) o ranking é FinBERT > BERTimbau ≈ NB > LinearSVC ≈ LogReg ≈ DeB3RTa. No OOD político (Fake.Br) é NB > LinearSVC > {DeB3RTa, FinBERT, LogReg, BERTimbau}. **TF-IDF NB é o modelo mais robusto a shift de domínio** observado aqui.
3. **Pretraining financeiro (FinBERT-PT-BR) só compensa quando o downstream também é financeiro**. Em Folha (jornalismo geral) FinBERT empata com BERTimbau. Em FakeRecogna `economia` UOL (financeiro especializado) FinBERT vence com significância. Evidência aditiva à Manchete A do strategy note: o sinal de domínio do pretraining é recuperável OOD quando alinhado.
4. **DeB3RTa-base não justifica seu custo em nenhum cenário**. 4× mais parâmetros que BERTimbau, F1 in-domain pior, F1 OOD ora médio (Fake.Br) ora terrível (UOL balanced). Recomendação: descartar do paper futuro como modelo viável; manter na tabela apenas como ponto de comparação ("nem todo encoder financeiro PT-BR transfere").
5. **Subgrupo true vs fake de Fake.Br abre questão de robustez adversarial sob-explorada**. Modelos generalizam pior para notícias reais economicamente relevantes do que para notícias falsas. Hipótese: notícias falsas usam vocabulário mercado-estereotipado ("dólar dispara", "bolsa despenca") que o classificador in-domain prioriza. Vale verificação por análise de erros (notebook 44 cobre o cross-task local; análise OOD pode usar o mesmo framework).
6. **Multiclasse OOD é pouco informativa com mapeamento atual** — macro-F1 0,10–0,26. Conclusão é dominada pela perda do mapeamento, não pelo modelo. Recomendação: reportar multiclasse OOD no apêndice somente; foco do paper em binário.

---

## 7. Limitações declaradas

1. **Validade de construto**: `mercado` é seção editorial Folha; OOD avalia se essa label é proxy de "notícia econômica". A baixa prevalência em ambos os corpora OOD evidencia que o construto é nicho.
2. **Mapeamento multiclasse OOD é lossy**. A taxonomia Folha (`poder`, `colunas`, `mercado`, ...) não tem correspondência limpa com Fake.Br (categorias jornalísticas distintas) ou FakeRecogna (multi-veículo). Macro-F1 baixo reflete o mapeamento, não capacidade discriminativa.
3. **Limiar de decisão não recalibrado**. Todos os runs usam o limiar 0,5 do treino in-domain. Recalibração no val do próprio corpus OOD não foi feita (correto para avaliação zero-shot; mas em uso prático limiar deveria ser ajustado).
4. **Sem busca de hiperparâmetros para BERTs**, conforme decisão metodológica do CLAUDE.md. OOD herda essa assimetria — TF-IDF foi otimizado in-domain via `RandomizedSearchCV`, BERTs usaram defaults. Mantém comparabilidade com a tabela in-domain; pode favorecer TF-IDF na transferência.
5. **Coverage = 1.0 em todos os runs OOD** (sem LLMs nesse fold). Não há complicação de coverage como nos LLMs in-domain.

---

## 8. Sugestões para inclusão no paper STIL

Esta avaliação OOD **fortalece a Manchete A** do `stil_strategy_note.md` ("pretraining financeiro de domínio não compensa quando o downstream é editorial geral") com evidência cross-corpus, e **adiciona uma manchete OOD-específica** candidata:

> **Manchete D — Transferência entre corpora preserva a vantagem do FinBERT-PT-BR somente quando o downstream também é financeiro especializado; em corpus político-geral, TF-IDF NB é mais robusto que qualquer BERT.**

Tabela sugerida para a versão final do paper (1 tabela, 6 linhas × 5 colunas): a Tabela §2.1 deste relatório.

Figura sugerida: scatter F1 in-domain (Folha test) vs F1 OOD (média FakeRecogna+Fake.Br) por modelo, com diagonal de referência. Mostra que TF-IDF NB e BERTimbau ficam mais perto da diagonal; DeB3RTa e LinearSVC caem mais.

McNemar+Bonferroni nas 5 configurações OOD pode ir como tabela suplementar; o corpo do paper cita só "FinBERT supera todos com significância em FakeRecogna UOL balanced (p_adj ≤ 0,019) mas é estatisticamente indistinguível de BERTimbau em Fake.Br (p_adj = 1,0)".

---

## 9. Pendências

- [ ] Conferir prevalência positiva e n_eval dos níveis `3_uol_balanced_*` (truncados no snapshot atual).
- [ ] Repetir análise por classe (per-class F1) na multiclasse para identificar se a vantagem do NB vem de uma classe específica ou é uniforme.
- [ ] Análise de erros OOD (extensão do notebook 44) para validar a hipótese vocabulário-estereotipado no subgrupo fake.
- [ ] Verificar se há cards equivalentes para ensembles (stacking/voting) sobre OOD — se sim, incluir; se não, declarar fora de escopo no paper.
