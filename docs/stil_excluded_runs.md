# STIL — Runs e análises a excluir do paper

Lista opinativa do que **não** entra no short paper STIL, com motivo específico para cada exclusão e onde cada item permanece útil. Os runs continuam em `artifacts/runs/` e nada é removido fisicamente — esta lista é apenas para a redação.

## A. Mencionado no abstract mas ausente do disco — *editar o abstract*

- **Llama-3.1 (qualquer variante)** — citado em `docs/bracis_stil_abstract.md` (linha 9: "Mistral-7B, Qwen2.5, Llama-3.1") mas **não há nenhum `llm_llama*` em `artifacts/runs/`**. Único LLMs presentes: `llm_mistral_7b_instruct_v0_3_*` e `llm_qwen2_5_7b_instruct_*`. **Ação obrigatória antes da submissão**: remover "Llama-3.1" do abstract. Manter sem rodar é vulnerabilidade trivial em revisão. Onde permanece útil: como trabalho futuro explícito ("estender a comparação de LLMs para Llama-3.1-8B-Instruct e Sabiá").

## B. Regimes auxiliares quando o regime principal já cobre

- **Todos os runs `*_fixed_split`** (regimes intermediários: `tfidf_*_fixed_split`, `bert_*_fixed_split`, `ensemble_majority_*_fixed_split`, `ensemble_weighted_binary_fixed_split`, `ensemble_agreement_*_fixed_split`) — o regime `cv_5fold` já dá variância (média ± std) e o `test_set` já dá o número final que vai na tabela. O `fixed_split` foi parte do fluxo de iteração rápida (calibração, threshold tuning, treino do stacking-meta), mas o número dele não tem interpretação separada no paper. Onde permanece útil: dissertação (apêndice de "fluxo de desenvolvimento"); auditoria de que o stacking-meta foi treinado em val do `fixed_split`, não no test.
- **`ensemble_agreement_*_cv_5fold` e `ensemble_agreement_*_fixed_split`** — Fleiss kappa só tem interpretação útil no test set (única partição que define a comparação final). As versões CV/fixed são redundantes. Reportar somente o número do test (binário κ=0,801, multiclasse a confirmar).

## C. Combinações de ensemble que distraem do recorte

- **`ensemble_weighted_binary_*` (3 cards)** — pesos derivados das próprias F1 dos modelos base. Conceitualmente é uma calibração-por-F1 sem aprendizado. F1=0,8828 no test_set, **abaixo do stacking** (0,8882) e marginalmente **acima do BERTimbau** (0,8675). Não acrescenta evidência distinta de stacking. Onde permanece útil: dissertação como ponto de comparação "ensemble sem meta-aprendiz vs com meta-aprendiz".
- **`ensemble_majority_*` (6 cards)** — voto majoritário entre 4 base models. F1 binário=0,8644 (test_set) está abaixo do BERTimbau sozinho (0,8675); macro-F1 multiclasse=0,9025 fica entre o melhor BERT (0,8941) e o stacking (0,9227). Mensagem confusa: "ensemble simples às vezes piora". Cortar do paper. Onde permanece útil: dissertação ("voto majoritário não é robusto quando os classificadores base têm qualidade muito diferente").
- **Manter no paper apenas `ensemble_stacking_binary_test_set` e `ensemble_stacking_multiclass_test_set`** (2 cards) como única evidência de ensemble. Eles são o melhor resultado em ambas as tarefas e justificam um parágrafo curto ("um meta-aprendiz treinado na validação consegue extrair ~2 pp de F1 acima do melhor modelo individual; o ganho é estatisticamente significativo após Bonferroni").

## D. Análises de coverage<1.0 sem re-restrição do test set

- **Comparação direta `llm_mistral_7b_instruct_v0_3_multiclass_few_shot_test_set` (`coverage=0,7149`) vs BERT/TF-IDF multiclasse** — descartar essa célula da tabela principal. 28,5% das amostras foram descartadas pelo LLM por falha de parsing; comparar 71,5% de cobertura com 100% de cobertura é injusto.
- **Comparação `llm_mistral_7b_instruct_v0_3_multiclass_zero_shot_test_set` (`coverage=0,9518`)**, **`..._binary_few_shot_test_set` (`coverage=0,9518`)**, **`..._binary_zero_shot_test_set` (`coverage=0,9835`)** — coverage menos catastrófico mas ainda <1.0. *Manter no paper somente se* a Tabela 1/2 incluir a coluna "F1 restrito ao subconjunto-LLM" (ver risco 5 do `stil_strategy_note.md`). Sem essa coluna, descartar Mistral inteiro do paper e usar apenas Qwen como representante de LLM (Qwen tem coverage 0,9764-0,9907 no binário, 0,8835-0,9872 na multiclasse, mais defensável).
- **Recomendação curta**: **manter Qwen2.5-7B-Instruct como único LLM no paper**, em zero-shot e few-shot, binário e multiclasse (4 cards). Mover Mistral-7B-Instruct-v0.3 para o apêndice de robustez ("um segundo LLM open-weight foi avaliado e atinge desempenho consistentemente inferior ao Qwen2.5; ver Tabela suplementar S1").

## E. Diretórios de busca de hiperparâmetros

- **`tfidf_logreg_search_binary/`, `tfidf_linearsvc_search_binary/`, `tfidf_nb_search_binary/`** e seus equivalentes multiclasse (6 dirs) — não vão no corpo do paper. O `result_card.json` dos runs `tfidf_*_test_set` já carrega o payload compacto `hyperparameter_search` (best_params, best_score, n_trials, search_seconds, search_space). O `search_result.json` interno (60 trials × hiperparâmetros) é evidência suplementar de rigor metodológico, não conteúdo de paper. Onde permanece útil: material suplementar online (anexo CSV); dissertação (capítulo de metodologia).

## F. Versões de modelo já dominadas

- **DeB3RTa-base na multiclasse** (`bert_deb3rta_base_multiclass_test_set` e seus regimes companheiros) — macro_F1=0,7951 fica 11 pontos abaixo do BERTimbau (0,8941). **Manter** o número binário (F1=0,7734) na Tabela 1 porque a Manchete A precisa mostrar o panorama "encoders financeiros vs encoder geral". Manter o número multiclasse na Tabela 2 pelo mesmo motivo. Não cortar — apenas não fazer dele um parágrafo de discussão. Análise per-class (`colunas` F1=0,6496 — pior do conjunto) fica no apêndice da dissertação.

## G. Métricas que o framework calcula mas o paper não usa

- **AUC-ROC dos LLMs** — calculado em todos os 4 cards LLM binários, mas **não comparável**: `y_score ∈ {0.0, 1.0}` é determinístico, ROC é um único ponto. Reportar na tabela com marca explícita "N/A (det.)" e nunca comparar com AUC de TF-IDF/BERT.
- **Brier score e ECE** — definidos no `evaluation.py`, mas não computados nos cards atuais (não estão na chave `metrics`). Não tentar adicioná-los para o STIL — adicionaria complexidade metodológica sem evidência empírica preparada. Vão para a dissertação.
- **`accuracy` como manchete** — calculado em todos os cards, mas a baseline trivial (predizer sempre a classe majoritária) atinge 0,8739 no binário. Reportar na tabela como informação contextual, **nunca como métrica principal**, e adicionar uma sentença na introdução: "Reportamos accuracy por convenção mas a métrica primária é F1 — accuracy de uma baseline trivial é 0,87".

## Resumo numérico do que entra vs o que sai

- **Cards usados no corpo do paper** (Tabela 1 + Tabela 2 + Figura 1 + Figura 2): 3 TF-IDF binário test_set + 3 TF-IDF multiclasse test_set + 3 BERT binário test_set + 3 BERT multiclasse test_set + 2 Qwen binário test_set + 2 Qwen multiclasse test_set + 1 stacking binário test_set + 1 stacking multiclasse test_set = **18 cards**, mais os 12 cards `cv_5fold` correspondentes para reportar variância = **~30 cards no corpo**.
- **Cards excluídos do paper, mantidos para dissertação/apêndice**: ~25 cards (todos `fixed_split`, todos `agreement`, todos `majority` e `weighted`, Mistral, search dirs).
