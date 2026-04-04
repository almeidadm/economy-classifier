# Guia: Treino BERT no Google Colab

Este guia descreve como executar o treino dos 3 modelos BERT (M4a, M4b, M4c) no Google Colab e integrar os resultados de volta ao repositorio local.

---

## Por que usar o Colab?

O treino de modelos BERT exige GPU com VRAM suficiente. A GTX 1650 local (4 GB VRAM) consegue treinar com `batch_size=1`, mas o tempo total estimado e de ~10-14 horas. O Google Colab oferece uma GPU T4 (16 GB VRAM) gratuitamente, permitindo:

| Parametro | Local (GTX 1650) | Colab (T4) |
|-----------|-----------------|------------|
| VRAM | 4 GB | 16 GB |
| Batch size treino | 1 | 8 |
| Batch efetivo | 8 | 16 |
| Batch eval | 1-8 | 32 |
| Tempo estimado (3 modelos) | ~10-14 h | ~2-4 h |

---

## Visao geral do fluxo

```
LOCAL                           COLAB                           LOCAL
─────                           ─────                           ─────

1. Notebook 01                  3. Abrir notebook               6. Baixar resultados
   (gerar splits)                  05_bert_colab.ipynb              do Drive
        │                              │
        ▼                              ▼
2. colab_pack.py                4. Treinar M4a, M4b, M4c       7. colab_unpack.py
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

Execute o notebook `01_preparacao_dados.ipynb` para gerar os splits estratificados. Isso cria os arquivos em `artifacts/splits/`.

```bash
# No terminal, verificar que os splits existem:
ls artifacts/splits/
# Esperado: balanced_train.parquet  val.parquet  test.parquet  split_metadata.json  ...
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
4. Selecione `notebooks/05_bert_colab.ipynb`

**Opcao B — Upload direto:**

1. Acesse [colab.research.google.com](https://colab.research.google.com)
2. Arquivo > Fazer upload de notebook
3. Selecione o arquivo `notebooks/05_bert_colab.ipynb` do repositorio local

### Passo 5 — Configurar runtime com GPU

1. No Colab: **Runtime > Change runtime type**
2. Selecione **T4 GPU**
3. Clique em **Save**

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

1. Verifica a GPU disponivel
2. Monta o Google Drive
3. Clona o repositorio e instala o pacote
4. Extrai os splits do zip
5. Treina M4a (BERTimbau) — ~40-80 min
6. Treina M4b (FinBERT) — ~40-80 min
7. Treina M4c (FinBERT-PT-BR) — ~40-80 min
8. Exibe tabela comparativa de metricas no split de validacao
9. Empacota resultados em `colab_bert_results.zip`

**Dica:** Cada modelo e treinado de forma independente. Se a sessao do Colab expirar, execute novamente a partir do modelo que faltou. Os resultados de modelos ja treinados estarao salvos no Drive (`economy-classifier/results/`).

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
      {timestamp}-bert-finbert/
        ...
      {timestamp}-bert-finbert-ptbr/
        ...
```

Baixe o `colab_bert_results.zip` do Google Drive para a maquina local.

### Passo 9 — Integrar resultados ao repositorio (local)

```bash
uv run python scripts/colab_unpack.py ~/Downloads/colab_bert_results.zip
```

Saida esperada:

```
Runs encontrados no zip: 3
  - {timestamp}-bert-training-bertimbau
  - {timestamp}-bert-training-finbert
  - {timestamp}-bert-training-finbert-ptbr

Artefatos extraidos em: /path/to/economy-classifier/artifacts/runs/
```

Os artefatos ficam em `artifacts/runs/`, no formato padrao do projeto. O notebook 07 pode carrega-los para a avaliacao comparativa final.

### Passo 10 — Continuar com a avaliacao (local)

Com os artefatos BERT integrados, prossiga com os notebooks restantes:

```
06_heuristica.ipynb          → Avaliar M5 no val
07_avaliacao_comparativa.ipynb → Comparacao final no teste
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

O notebook usa configuracoes otimizadas para a GPU T4:

| Parametro | Valor | Justificativa |
|-----------|-------|---------------|
| `per_device_train_batch_size` | 8 | T4 suporta confortavelmente |
| `gradient_accumulation_steps` | 2 | Batch efetivo = 16 |
| `per_device_eval_batch_size` | 32 | Inferencia usa menos VRAM |
| `num_train_epochs` | 3 | Maximo; early stopping pode parar antes |
| `early_stopping_patience` | 1 | Para se epoca N+1 nao melhora F1 |
| `save_total_limit` | 2 | Limita checkpoints em disco |
| `fp16` | True (auto) | Ativado automaticamente com CUDA |
| `max_length` | 256 | Padrao do projeto |

Para ajustar, modifique as constantes `COLAB_*` na celula de funcoes auxiliares.

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

### Modelo FinBERT (M4b) com desempenho baixo

Esperado: o FinBERT foi pre-treinado em ingles, e o fine-tuning em portugues pode nao transferir bem. Compare com as outras variantes antes de concluir.

---

## Arquivos relacionados

| Arquivo | Descricao |
|---------|-----------|
| `scripts/colab_pack.py` | Empacota splits para upload ao Colab |
| `scripts/colab_unpack.py` | Integra resultados do Colab ao repositorio local |
| `notebooks/05_bert_colab.ipynb` | Notebook de treino para Google Colab |
| `src/economy_classifier/bert.py` | Modulo de treino e inferencia BERT |
| `docs/estimativa_recursos_computacionais.md` | Estimativas detalhadas de tempo e recursos |
