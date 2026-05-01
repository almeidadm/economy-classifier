# Guia: Treino BERT no Google Colab

Este guia descreve como executar o treino dos 3 modelos BERT (M4a, M4b, M4c) no Google Colab e integrar os resultados de volta ao repositorio local.

---

## Por que usar o Colab?

O fluxo agora inclui **busca de hiperparametros** (RandomizedSearchCV) antes dos 3 regimes de avaliacao por modelo. Cada modelo BERT roda 25 trials de busca + 5 folds de CV + 2 single shots por tarefa (binario + multiclasse). Sao **~50+ fits por modelo por tarefa**, totalizando ~300+ fits para os 3 modelos.

A GPU T4 (16 GB) e insuficiente para o orcamento completo no tempo disponivel. **A100** (40 GB) e a recomendacao:

| Parametro | T4 (16 GB) | A100 (40 GB) |
|-----------|-----------|--------------|
| `per_device_train_batch_size` | 8 | 32 |
| Tempo medio por trial | ~30-60 min | ~10-15 min |
| Tempo total (3 modelos x 2 tarefas x ~50 fits) | ~5-10 dias | ~1-3 dias |

Se voce so tem T4 disponivel, ajuste no notebook 21:
- Reduza `N_ITER_BERT` de 25 para 5-10
- Comente modelos em `MODELS = [...]` para rodar um por vez
- Considere rodar so as tarefas binarias (comente as celulas multiclasse)

---

## Visao geral do fluxo

```
LOCAL                           COLAB                           LOCAL
─────                           ─────                           ─────

1. Notebook 01                  3. Abrir notebook               6. Baixar resultados
   (gerar splits)                  21_bert.ipynb                   do Drive
        │                              │
        ▼                              ▼
2. colab_pack.py                4. Treinar M4a, M4b, M4c       7. colab_unpack_streaming
   (gerar zip)                     (GPU T4)                        (integrar artefatos)
        │                              │
        ▼                              ▼                              │
   Upload zip ──────────────►   Resultados + backup              ◄────┘
   para Drive                   salvos no Drive                     │
                                                                    ▼
                                                            8. Notebook 07
                                                               (avaliacao comparativa)
```

---

## Passo a passo

### Passo 1 — Gerar os splits (local)

Execute o notebook `01_preparacao_dados.ipynb` para gerar os splits 80/10/10 e os 5 folds da CV. Isso cria os arquivos em `artifacts/splits/`.

```bash
# No terminal, verificar que os splits existem:
ls artifacts/splits/
# Esperado: train.parquet  val.parquet  test.parquet  cv_folds.json  split_metadata.json  ...
```

### Passo 2 — Empacotar splits para upload (local)

```bash
uv run python scripts/colab_pack.py
```

Saida esperada:

```
  + balanced_train.parquet (X.X MB)
  + val.parquet (X.X MB)
  + test.parquet (X.X MB)
  + split_metadata.json (0.0 MB)
  + train_indices.csv (X.X MB)
  + val_indices.csv (X.X MB)
  + test_indices.csv (X.X MB)

Arquivo gerado: /path/to/economy-classifier/colab_splits.zip
Tamanho total: X.X MB
```

### Passo 3 — Upload para o Google Drive

1. Abra [Google Drive](https://drive.google.com)
2. Crie uma pasta chamada `economy-classifier` na raiz do Drive
3. Faca upload do arquivo `colab_splits.zip` para esta pasta

Estrutura esperada no Drive:

```
My Drive/
  economy-classifier/
    colab_splits.zip
```

### Passo 4 — Abrir o notebook no Colab

**Opcao A — Via GitHub (recomendado se o repo esta no GitHub):**

1. Acesse [colab.research.google.com](https://colab.research.google.com)
2. Arquivo > Abrir notebook > GitHub
3. Cole a URL do repositorio
4. Selecione `notebooks/21_bert.ipynb`

**Opcao B — Upload direto:**

1. Acesse [colab.research.google.com](https://colab.research.google.com)
2. Arquivo > Fazer upload de notebook
3. Selecione o arquivo `notebooks/21_bert.ipynb` do repositorio local

### Passo 5 — Configurar runtime com GPU

1. No Colab: **Runtime > Change runtime type**
2. Selecione **A100 GPU** (recomendado) ou **T4** (com `N_ITER_BERT` reduzido)
3. Clique em **Save**

A primeira celula do notebook (`## 0. Verificacao de GPU`) emite aviso se a GPU detectada nao for A100.

### Passo 6 — Ajustar configuracao (se necessario)

Na celula de configuracao do notebook, verifique:

```python
REPO_URL = "https://github.com/almeidadm/economy-classifier.git"
REPO_BRANCH = "main"
DRIVE_FOLDER = "economy-classifier"
```

- `REPO_URL`: URL do seu repositorio no GitHub
- `REPO_BRANCH`: branch com o codigo atualizado
- `DRIVE_FOLDER`: pasta no Google Drive onde esta o `colab_splits.zip`

### Passo 7 — Executar o notebook

Execute as celulas na ordem. O notebook:

1. Verifica a GPU disponivel (A100 recomendado)
2. Monta o Google Drive, clona o repositorio, instala o pacote, extrai os splits
3. Define o protocolo `run_full_protocol(model_key, model_name)` que para cada modelo:
   - Roda `random_search_bert` para a tarefa **binaria** (25 trials em train+val) → emite `bert_{model}_search_binary/search_result.json`
   - Roda `random_search_bert` para a tarefa **multiclasse** (25 trials) → emite `bert_{model}_search_multiclass/search_result.json`
   - Treina e avalia nos 6 regimes (binario/multi x fixed_split/cv_5fold/test_set) com `best_params`, emitindo 6 `result_card.json`
4. Loop sobre `MODELS = [bertimbau, finbert_ptbr, deb3rta_base]`
5. Sumario final com 18 result cards (3 modelos x 6 regimes)

Saida total: 18 result_cards + 6 search logs por execucao completa. Cada `result_card` carrega o payload `hyperparameter_search` (best_params, best_score, search_seconds, etc).

**Dica para sessoes longas:** Cada modelo e independente. Se a sessao expirar, edite `MODELS = [...]` para rodar so o que faltou. Os resultados ja salvos no Drive (`runs/bert_*`) nao serao retreinados.

### Passo 8 — Baixar resultados

Os resultados sao salvos automaticamente no Google Drive:

```
My Drive/
  economy-classifier/
    colab_splits.zip              (input)
    colab_bert_results.zip        (output — tudo junto)
    results/                      (output — backup por modelo)
      {timestamp}-bert-bertimbau/
        predictions_val.csv
        predictions_test.csv
        metrics.json
        run_metadata.json
      {timestamp}-bert-finbert-ptbr/
        ...
      {timestamp}-bert-deb3rta-base/
        ...
```

Baixe os zips `runs-*-XXX.zip` do Google Drive para `~/Downloads/`.

### Passo 9 — Integrar resultados ao repositorio (local)

```bash
# Inspecao previa (nao escreve em disco)
uv run python scripts/colab_unpack_streaming.py --dry-run

# Extracao seletiva (filtra checkpoints/, model/*.safetensors, tokenizer)
uv run python scripts/colab_unpack_streaming.py --delete-after
```

Saida esperada (resumo final):

```
Zips processados com sucesso: N
Total extraido: M arquivos, X.X MiB
Total result_card.json: K
Total predictions.csv:  K
```

O script auto-descobre zips em `~/Downloads/` (ajustavel via `--zips-dir`) e escreve em `artifacts/runs/`. Veja a docstring do script para flags adicionais (`--pattern`, `--keep-tfidf-joblib`, `--continue-on-error`).

Os artefatos ficam em `artifacts/runs/`, no formato padrao do projeto. Os notebooks 41 (EDA), 42 (tabela final, reservado) e 43 (ensemble) podem carrega-los para a avaliacao comparativa final.

### Passo 10 — Continuar com a avaliacao (local)

Com os artefatos BERT integrados, prossiga com os notebooks restantes:

```
41_eda_resultados.ipynb       → EDA dos result_cards
42_comparacao.ipynb           → Tabela final do artigo (reservado)
43_ensemble.ipynb             → Voting + stacking sobre os 4 modelos base
```

---

## Estrutura dos artefatos gerados

Cada modelo BERT gera um diretorio de run com a seguinte estrutura:

```
artifacts/runs/{timestamp}-bert-training-{variant}/
    model/                          ← Modelo treinado (tokenizer + weights)
        config.json
        model.safetensors
        tokenizer.json
        tokenizer_config.json
        vocab.txt
        special_tokens_map.json
    predictions_val.csv             ← Predicoes no split de validacao
    predictions_test.csv            ← Predicoes no split de teste
    metrics.json                    ← Metricas de avaliacao
    run_metadata.json               ← Metadados completos do run
```

### Formato das predicoes (CSV)

```csv
index,y_true,y_pred,y_score,method
0,1,1,0.923,bertimbau
1,0,0,0.087,bertimbau
```

### Formato das metricas (JSON)

```json
{
    "precision": 0.9245,
    "recall": 0.8901,
    "f1": 0.9070,
    "accuracy": 0.9734,
    "auc_roc": 0.9812
}
```

---

## Configuracoes de treino no Colab

A maioria dos hiperparametros vem da **busca aleatoria** (`random_search_bert`). Apenas os defaults nao-otimizaveis ficam em `BASE_OVERRIDES`:

| Parametro | Valor (BASE_OVERRIDES) | Justificativa |
|-----------|------------------------|---------------|
| `max_length` | 256 | Padrao do projeto |
| `per_device_eval_batch_size` | 64 | A100 suporta confortavelmente |
| `early_stopping_patience` | 1 | Para se epoca N+1 nao melhora |
| `save_total_limit` | 1 | Minimiza disco no Colab |
| `gradient_checkpointing` | False | A100 com 40 GB nao precisa |

**Espaco de busca** (`build_bert_search_space`):

| Parametro | Distribuicao | Range |
|-----------|--------------|-------|
| `learning_rate` | loguniform | 1e-5 — 5e-5 |
| `per_device_train_batch_size` | choice | {8, 16, 32} |
| `num_train_epochs` | int | 2 — 5 |
| `weight_decay` | loguniform | 1e-3 — 1e-1 |
| `warmup_ratio` | uniform | 0.0 — 0.2 |
| `gradient_accumulation_steps` | choice | {1, 2, 4} |

Para ajustar, modifique `BASE_OVERRIDES`, `SEARCH_SPACE` ou `N_ITER_BERT` na celula de configuracao.

---

## Solucao de problemas

### Sessao expirou durante treino

Os resultados de modelos ja concluidos estao salvos no Drive (pasta `results/`). Re-execute o notebook a partir do modelo que faltou — os anteriores nao precisam ser retreinados.

### OutOfMemoryError (OOM) na GPU

Improvavel com a T4, mas se ocorrer:

1. Reduza `COLAB_TRAIN_BATCH` para 4
2. Aumente `COLAB_GRAD_ACCUM` para 4 (mantendo batch efetivo = 16)
3. Reinicie o runtime: Runtime > Restart runtime

### Repositorio nao encontrado ao clonar

Verifique se:
- O repositorio esta no GitHub e e publico (ou configure autenticacao)
- A variavel `REPO_URL` esta correta
- A branch `REPO_BRANCH` existe

Alternativa: faca upload manual do repositorio como zip e descompacte no Colab.

### colab_splits.zip nao encontrado

Verifique se:
- O arquivo esta em `Google Drive > economy-classifier/colab_splits.zip`
- A variavel `DRIVE_FOLDER` corresponde ao nome exato da pasta no Drive
- O Google Drive foi montado com sucesso

### DeB3RTa requer SentencePiece

DeB3RTa-base usa tokenizer SentencePiece (DeBERTa-v2). No Colab, instale com `pip install sentencepiece` se a celula de bootstrap nao instalar automaticamente. Localmente, certifique-se que `sentencepiece` esta listado no `pyproject.toml`.

---

## Arquivos relacionados

| Arquivo | Descricao |
|---------|-----------|
| `scripts/colab_pack.py` | Empacota splits para upload ao Colab |
| `scripts/colab_unpack_streaming.py` | Integra resultados do Colab ao repositorio local (multi-zip, streaming, seletivo) |
| `notebooks/21_bert.ipynb` | Notebook de treino para Google Colab |
| `src/economy_classifier/bert.py` | Modulo de treino e inferencia BERT |
| `docs/estimativa_recursos_computacionais.md` | Estimativas detalhadas de tempo e recursos |
