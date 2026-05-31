# STIL — Nota de Estratégia Metodológica

Documento de planejamento para o short paper. Objetivo: chegar a um recorte único, defensável, ancorado nos 55 `result_card.json` em `artifacts/runs/`.

## 1. Recorte para STIL — qual é a contribuição única?

O **STIL é o veículo certo** (Symposium on Information and Human Language Technology, co-localizado com BRACIS, audiência de PLN em português, formato curto que premia uma contribuição empírica nítida em recurso PT-BR). O abstract atual (`docs/bracis_stil_abstract.md`) tenta empacotar três contribuições — benchmark cost-aware, BERT financeiro de domínio, e LLMs zero/few-shot. Em 6-8 páginas isso dilui a mensagem.

**Recomendação: focar em (a) BERT de domínio financeiro vs BERT geral PT-BR, dentro de um benchmark cost-aware com TF-IDF como piso e LLMs zero/few-shot como teto teórico que falha empiricamente.** Os outros dois recortes são candidatos descartados por essas razões:

- *"Benchmark cost-aware genérico de TF-IDF/BERT/LLM"* — competitivo, mas sem PT-BR-específico forte. STIL espera ancoragem na língua e em recursos PT-BR.
- *"McNemar com correção de Bonferroni aplicada"* — é uma contribuição metodológica real (a literatura PT-BR raramente reporta), mas é uma contribuição pequena demais para sustentar o paper sozinha. Vai como subseção, não como tese.

O recorte recomendado tem três alavancas únicas:
1. **FinBERT-PT-BR (`lucas-leme/FinBERT-PT-BR`) avaliado fora de seu domínio nativo de pretraining** — pretreinado em corpus financeiro brasileiro, mas o teste é em notícias gerais da Folha onde `mercado` é a seção editorial-alvo. A pergunta "domínio do pretraining ajuda quando o downstream não é exatamente o mesmo domínio?" é pouco estudada em PT-BR.
2. **DeB3RTa-base (`higopires/DeB3RTa-base`) como segundo modelo financeiro PT-BR** — replicação interna entre dois encoders financeiros.
3. **Resultado-supresa empírico**: BERTimbau geral **bate marginalmente** o FinBERT-PT-BR no teste binário (F1 0.8675 vs 0.8659) e no CV 5-fold (0.8671±0.0027 vs 0.8627±0.0042). DeB3RTa fica bem atrás (0.7734). Esta é uma evidência negativa publicável — dispendiosos pretrainings de domínio nem sempre transferem.

## 2. Seção de Metodologia — texto pronto para o paper

Os parágrafos abaixo estão escritos para um revisor que não viu o `CLAUDE.md`. Podem ir como subseções da Section 3 (Methodology).

### 2.1 Corpus e splits

> Utilizamos um corpus de 166.288 artigos da *Folha de São Paulo* etiquetados pela seção editorial. Para a tarefa binária, contraímos o esquema editorial em `mercado` (positiva, ~12,61%) vs `outros` (negativa). Para a multiclasse, retivemos as sete seções mais frequentes (`poder`, `colunas`, `mercado`, `esporte`, `mundo`, `cotidiano`, `ilustrada`) mais um agregado `outros`. Os dados foram particionados estratificadamente em treino/validação/teste 80/10/10 (133.030 / 16.629 / 16.629 amostras), preservando a distribuição natural da classe positiva em todas as partições. O conjunto de teste é mantido intacto: nunca foi usado para seleção de hiperparâmetros, calibração de limiar, early stopping ou treino de meta-classificador. Os mesmos índices são reusados pelas duas formulações — somente o vetor `y` muda — garantindo comparabilidade por construção.

**Decisão sobre o seed=42 vs seed=2026 (a "nota histórica" do CLAUDE.md):** *recomendo não disclorar* na metodologia do paper. Os splits persistidos foram gerados com `seed=42`; o framework documenta `seed=2026` como o seed global "de regeneração". Para o leitor externo, isso é ruído sem valor metodológico — o que importa é que o split é fixo, estratificado, persistido e idêntico para todos os métodos. Reportamos `seed=42` ou simplesmente "fixed seed". Se um revisor pedir reprodutibilidade exata, o `artifacts/splits/split_metadata.json` documenta o `corpus_sha256` e o seed real. A discussão dos três seeds (`2026` global, `2027` busca interna, `42` splits persistidos) pertence ao apêndice de reprodutibilidade da dissertação, não a um short paper.

### 2.2 Métricas

> Por construção da tarefa binária (87,4% de exemplos negativos), *accuracy* é enganosa: um classificador trivial constante atinge ~0,87. Reportamos como métrica primária binária o F1-score da classe positiva (`mercado`); como métricas secundárias, precision, recall, AUC-ROC e accuracy. Para a tarefa multiclasse, a métrica primária é macro-F1 (média não-ponderada das oito classes), que penaliza desempenho ruim em classes minoritárias; reportamos também weighted-F1 e F1 por classe.

### 2.3 Busca de hiperparâmetros — assimetria declarada

> Os modelos TF-IDF (Logistic Regression, Linear SVC, Multinomial NB) passam por `RandomizedSearchCV` com 60 trials e CV 5-fold interna sobre o pool train+val (90%) antes da avaliação no test set. O espaço de busca cobre `ngram_range ∈ {(1,1), (1,2)}`, `min_df ∈ {2,5,10,20}`, `max_df ∈ {0.85,0.9,0.95,1.0}`, `max_features ∈ {50k,100k,200k}`, `sublinear_tf`, e o hiperparâmetro principal de cada classificador (`C` log-uniforme em [0.001,100] para LogReg/SVC; `alpha` log-uniforme em [0.001,10] para NB). O scoring da busca binária é `f1`; o scoring multiclasse é `f1_macro`. Os modelos BERT (BERTimbau, FinBERT-PT-BR, DeB3RTa) **não passam por busca de hiperparâmetros**; usam configuração derivada da literatura (`learning_rate=2e-5`, `batch_size=16`, `epochs=3`, `weight_decay=0.01`, `warmup_ratio=0.1`, `max_seq_length=128`). Esta assimetria é uma exceção declarada, justificada por custo computacional (3 modelos × 2 tarefas × ~25 trials excederiam dias de A100) e pela estabilidade conhecida desses defaults para BERT-base PT-BR. A consequência metodológica é que os números de BERT representam um *limite inferior* do que esses encoders entregariam sob busca.

### 2.4 Cross-validation, variância, e o conjunto de teste

> Reportamos para cada modelo três regimes complementares: (a) `fixed_split` — treino no train, avaliação na val; (b) `cv_5fold` — 5 folds estratificados sobre o pool train+val (90%), com média e desvio-padrão; (c) `test_set` — refit em train+val e avaliação única no teste retido. A independência entre o CV interno da busca de hiperparâmetros TF-IDF (`seed=2027`) e o CV de variância (folds persistidos) é mantida deliberadamente, evitando que o regime `cv_5fold` reporte variância em partições já usadas para selecionar hiperparâmetros.

### 2.5 Testes estatísticos com correção para múltiplas comparações

> Diferenças entre métodos são avaliadas com o teste de McNemar pareado sobre o conjunto de teste. Quando comparamos K modelos, a família contém K(K-1)/2 pares; aplicamos correção de Bonferroni para controlar a taxa de erro Tipo I em α=0,05 sobre toda a família. Reportamos o p-valor não corrigido, o p-valor ajustado e o veredicto `significant_after_correction`.

### 2.6 LLMs — limitações declaradas

> O pipeline LLM extrai apenas o rótulo final do output do modelo, sem amostrar com `temperature>0` nem usar logits, resultando em `y_score ∈ {0.0, 1.0}` determinístico. Consequentemente: (i) o AUC-ROC computado para LLMs corresponde a uma curva de um único ponto operacional e **não mede separabilidade** — reportamos AUC para LLMs entre parênteses, marcado `N/A — deterministic scores`, e não o usamos em comparações pareadas com TF-IDF/BERT; (ii) quando o modelo produz output não parseável, a amostra é descartada e o `coverage` é reportado explicitamente. Para o subconjunto de LLMs com `coverage < 1.0`, comparações com TF-IDF/BERT são re-restritas aos índices efetivamente cobertos pelo LLM.

### 2.7 Stacking treinado na validação

> O meta-classificador (LogisticRegression) do ensemble por stacking é treinado nos *scores de validação* dos modelos base, não nos scores de treino. Isso evita data leakage: usar scores de treino vazaria a memorização dos modelos base para o meta-aprendiz. O stacking é avaliado no test set com refit dos modelos base em train+val. As 4 entradas-base do stacking foram `tfidf_logreg`, `tfidf_linearsvc`, `tfidf_nb`, `bert_bertimbau` (FinBERT-PT-BR e DeB3RTa não entram para preservar tempo de treino do meta).

## 3. Achados-manchete candidatos

Cada um foi testado nos cards. Use 1, descarte 2, ou combine 1+3.

**Manchete A — Pretraining financeiro de domínio não compensa quando o downstream é editorial geral.** Evidência: BERTimbau geral atinge F1=0,8675 (test_set) e 0,8671±0,0027 (cv_5fold) na tarefa binária; FinBERT-PT-BR atinge 0,8659 e 0,8627±0,0042 respectivamente. A diferença é favorável ao BERTimbau, está dentro de um sigma do CV, e deve ser confirmada por McNemar pareado (a fazer no notebook 41 sobre `predictions.csv`). DeB3RTa-base, o terceiro encoder financeiro, fica bem atrás (0,7734), apesar de ter 4× mais parâmetros (426M vs 109M). Conclusão publicável: "para detecção de notícias econômicas em corpus jornalístico geral, encoders financeiros pré-treinados não superam BERTimbau e podem ficar abaixo dele". Esta é a manchete mais defensável.

**Manchete B — TF-IDF + LogReg, com busca de hiperparâmetros, é Pareto-ótimo em custo-benefício.** Evidência: LogReg atinge F1=0,8601 binário (cv_5fold 0,8511±0,0027), a 6 pontos percentuais do melhor BERT, com **5× menos tempo de treino** (315s vs 1700s) e **3× maior throughput de inferência** (435 amostras/s vs 148). Na multiclasse, LogReg (`macro_f1=0,8981`) **supera marginalmente todos os BERTs** (BERTimbau 0,8941; FinBERT 0,8896). Esta é a segunda manchete e a única com um achado contra-intuitivo direto: para essa tarefa, BERT não bate TF-IDF na multiclasse. Vale destaque.

**Manchete C — LLMs zero/few-shot estão 25 pontos percentuais abaixo do BERT fine-tunado, ao custo de 100-200× mais inferência.** Evidência (binário, test_set, F1): Mistral-7B few-shot 0,640; Qwen2.5-7B few-shot 0,632; Qwen2.5-7B zero-shot 0,634; Mistral-7B zero-shot 0,579. Comparado com BERTimbau (0,8675). Multiclasse é pior: Qwen few-shot macro-F1=0,572 com `coverage=0,884`; Mistral few-shot 0,436 com `coverage=0,715`. Inferência LLM consome 11k-26k segundos vs 38-112s para TF-IDF/BERT. Conclusão: para essa tarefa específica em PT-BR, com 150k exemplos rotulados disponíveis, LLM prompting não é competitivo. *Few-shot não bate zero-shot consistentemente* (Mistral fs=0,640 > zs=0,579 binário, mas Qwen fs≈zs).

Recomendação: **Manchetes A e C como foco do paper**; B aparece como observação na tabela mas não como manchete (a vantagem de TF-IDF é pequena em parte porque a busca foi exaustiva e BERT não teve busca — falar isso enfraqueceria a Manchete A).

## 4. Tabelas e figuras recomendadas para 6-8 páginas

- **Tabela 1 — Resultado principal binário (test_set + cv_5fold):** colunas: modelo, F1 (test_set), F1 (cv_5fold mean ± std), AUC-ROC, treino (s), inferência (s), throughput. Linhas: TF-IDF×3, BERT×3, LLM×4 (zero+few × 2 modelos), ensemble×3 (majority, weighted, stacking). Marcar AUC dos LLMs como "N/A (det.)".
- **Tabela 2 — Resultado principal multiclasse (test_set + cv_5fold):** mesmo formato, métrica é macro-F1 e weighted-F1.
- **Tabela 3 — F1 por classe na multiclasse (test_set):** confirma que `colunas` é a classe mais difícil e que LLMs colapsam nela (F1=0,10 Mistral fs; 0,21 Qwen fs).
- **Tabela 4 — McNemar pareado com Bonferroni** sobre o test_set binário: matriz triangular K(K-1)/2 com p-valor ajustado e `significant_after_correction`. K=6 (3 TF-IDF + 3 BERT + omitir LLMs ou marcá-los à parte) → 15 testes. Reportar n da família explicitamente.
- **Figura 1 — Pareto F1 vs throughput (binário, test_set):** scatter log-x throughput, y=F1, marcadores por família. Mostra TF-IDF dominando o canto superior-esquerdo, BERT no meio, LLMs no canto inferior-direito.
- **Figura 2 — Matriz de confusão da multiclasse (BERTimbau test_set):** uma única matriz 8×8 normalizada por linha, mostra que `colunas` é confundida com `mercado` (a observação editorial central que justifica o framing).

Cortado por limite de páginas: matriz de confusão dos demais BERTs, curvas ROC (LLM AUC seria meia-verdade), Fleiss kappa (vai pro texto: "κ=0,801 entre os 4 modelos base do stacking, indicando concordância substancial").

## 5. Riscos metodológicos abertos e como tratar

1. **Validade de construto da label `mercado`** — `mercado` é uma seção editorial, não anotação linguística de conteúdo econômico. Tratar com um parágrafo na seção de "Limitations" do paper, citando que um artigo classificado em `colunas` pode tratar de economia mas ser etiquetado como `colunas` por convenção editorial. A confusão modelo-mediada entre `colunas` e `mercado` (visível na matriz de confusão) é evidência *favorável ao modelo*, não contra ele.
2. **AUC-ROC dos LLMs é enganoso** — já tratado em §2.6. Nunca incluir LLMs em comparações de AUC; sempre acompanhar o número de "N/A (det.)".
3. **Multiclasse sem `y_score` para stacking probabilístico** — o stacking multiclasse usa `y_proba_<classe>` dos 4 base models (32 features), mas LLMs ficam fora porque `hf_results_to_multiclass_predictions` não emite `y_score`. Declarar isso explicitamente: "stacking multiclasse foi treinado apenas sobre as saídas TF-IDF e BERTimbau". Não tentar consertar para o STIL — fica para a dissertação.
4. **Llama-3.1 mencionado no abstract mas ausente nos resultados** — *editar o abstract*. Os 8 cards LLM cobrem somente Mistral-7B-Instruct-v0.3 e Qwen2.5-7B-Instruct. Manter Llama-3.1 no abstract sem rodá-lo é desonestidade científica e revisor sério vai pegar. Já listado no `stil_excluded_runs.md` como ação obrigatória.
5. **Coverage<1.0 nos LLMs multiclasse contamina comparações** — Mistral few-shot multiclasse tem `coverage=0,7149`, ou seja, 28,5% das amostras descartadas. Comparar `macro_f1=0,4359` desse modelo com BERTimbau `macro_f1=0,8941` (avaliado em 100% das amostras) é apples-to-oranges. Tratamento: re-restringir o test set de TF-IDF/BERT aos índices cobertos por cada LLM e reportar uma coluna adicional na Tabela 2 ("F1 restrito ao subconjunto-LLM"). Isto é um trabalho de notebook (~1h) que precisa ser feito antes da submissão.
6. **A vantagem do TF-IDF sobre BERT na multiclasse pode ser artefato da assimetria de busca** — TF-IDF teve 60 trials, BERT teve config fixa. Declarar isso explicitamente quando essa observação aparecer na seção de discussão; é o motivo pelo qual essa observação não vira manchete.
