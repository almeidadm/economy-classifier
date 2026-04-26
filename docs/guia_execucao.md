# Guia de Execucao do Pipeline

Documento pratico para conduzir uma rodada completa do pipeline (search + 3 regimes x 2 tarefas, para todos os modelos). Pensado como **checklist** — siga na ordem, marque o que ja foi feito.

---

## 1. Visao geral em 1 pagina

```
┌──────────────────────────────────────────────────────────────────────┐
│                       ORDEM DE EXECUCAO                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ETAPA 1 (LOCAL)                                                    │
│   01_preparacao_dados   →  artifacts/splits/{train,val,test}.parquet │
│                            artifacts/splits/cv_folds.json            │
│                                                                      │
│   ETAPA 2 (LOCAL OU COLAB CPU)                                       │
│   02_tfidf_logreg       →  6 result_cards + 2 search logs (M1)       │
│   03_tfidf_linearsvc    →  6 result_cards + 2 search logs (M2)       │
│   04_tfidf_multinomialnb→  6 result_cards + 2 search logs (M3)       │
│                                                                      │
│   ETAPA 3 (COLAB A100)                                               │
│   colab_pack.py         →  colab_splits.zip  (upload pro Drive)      │
│   05_bert_colab         →  18 result_cards + 6 search logs (M4a/b/c) │
│   colab_unpack.py       →  integra resultados localmente             │
│                                                                      │
│   ETAPA 4 (LOCAL)                                                    │
│   07_ensemble           →  4 ensembles (E1-E4) sobre o test_set      │
│   12_comparacao         →  tabela final (todos x todos os regimes)   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

Saida total esperada:
  - 36 result_cards (6 modelos x 6 regimes)
  -  8 search logs (TF-IDF: 6 = 3 modelos x 2 tarefas; BERT: 6 = 3 x 2)
  -  4 ensembles
  -  1 tabela comparativa
```

---

## 2. Pre-requisitos

| Requisito | Como verificar |
|-----------|----------------|
| Python >= 3.12 | `python --version` |
| `uv` instalado | `uv --version` |
| Repositorio clonado e dependencias instaladas | `uv sync --all-extras` |
| Corpus em `data/news-of-the-site-folhauol/articles.csv` | `ls data/news-of-the-site-folhauol/` |
| Conta Google (para Drive + Colab) | Login em [colab.research.google.com](https://colab.research.google.com) |
| **Plano Colab Pro** ou Pro+ (para A100) | Check em [colab.research.google.com/signup](https://colab.research.google.com/signup) |
| `git` configurado e repositorio acessivel pelo Colab | Push do branch que voce vai usar |

**Decisao importante antes de comecar**: vai rodar BERT em A100 ou T4?
- **A100** (recomendado): tempo total ~1-3 dias; custo ~5-15 unidades de computacao Colab/hora
- **T4** (orcamento reduzido): edite `N_ITER_BERT=10` e/ou `MODELS=[("bertimbau", ...)]` no notebook 05; tempo ~1 dia por modelo

---

## 3. Etapa 1 — Preparacao de dados (LOCAL, ~5 min)

**Objetivo**: gerar splits 80/10/10 + 5 folds da CV, persistidos em `artifacts/splits/`.

### Como executar

```bash
uv run jupyter notebook notebooks/01_preparacao_dados.ipynb
# Run all cells
```

### Saida esperada

```
artifacts/splits/
    train.parquet              (~133.030 linhas, ~12.5% mercado)
    val.parquet                (~16.629 linhas)
    test.parquet               (~16.629 linhas)
    cv_folds.json              (5 folds estratificados sobre train+val)
    split_metadata.json        (seed, hash do corpus, contagens)
```

### Como saber se deu certo

```bash
ls artifacts/splits/
# Deve listar os 5 arquivos acima

uv run python -c "
import json, pandas as pd
print('split_metadata:', json.load(open('artifacts/splits/split_metadata.json')))
print('cv_folds count:', len(json.load(open('artifacts/splits/cv_folds.json'))))
for s in ['train', 'val', 'test']:
    df = pd.read_parquet(f'artifacts/splits/{s}.parquet')
    print(f'{s}: {len(df):,} linhas, mercado={df[\"label\"].mean()*100:.2f}%')
"
```

Voce deve ver `mercado` ~12.5% em cada split, e `cv_folds count: 5`.

### Antes de prosseguir

Esta etapa so precisa rodar **uma vez por seed** (default 42). Se voce ja tem `artifacts/splits/`, pule para a etapa 2.

---

## 4. Etapa 2 — TF-IDF (LOCAL, ~3-7h por notebook)

**Objetivo**: para cada classificador linear (logreg/linearsvc/nb), executar `RandomizedSearchCV` (60 trials x 5 inner folds) e depois os 6 regimes (binario/multi x fixed/cv/test) com os melhores hiperparametros.

### Estrategia recomendada

Os 3 notebooks sao **independentes** e podem rodar em paralelo (em terminais diferentes) se voce tiver RAM/CPU suficiente. Cada um leva ~3-7h em CPU local.

### Como executar (sequencial, terminal a terminal)

```bash
# Notebook 02 — Logistic Regression
uv run jupyter nbconvert --to notebook --execute notebooks/02_tfidf_logreg.ipynb \
    --output 02_tfidf_logreg.executed.ipynb

# Notebook 03 — LinearSVC
uv run jupyter nbconvert --to notebook --execute notebooks/03_tfidf_linearsvc.ipynb \
    --output 03_tfidf_linearsvc.executed.ipynb

# Notebook 04 — Multinomial NB
uv run jupyter nbconvert --to notebook --execute notebooks/04_tfidf_multinomialnb.ipynb \
    --output 04_tfidf_multinomialnb.executed.ipynb
```

Alternativa interativa: abra cada notebook no Jupyter e rode celula a celula. Util para debugar a primeira vez.

### Saida esperada por notebook

Para `02_tfidf_logreg`:

```
artifacts/runs/
    tfidf_logreg_search_binary/
        search_result.json         (60 trials, best_params, scores)
    tfidf_logreg_search_multiclass/
        search_result.json
    tfidf_logreg_binary_fixed_split/
        result_card.json
        predictions.csv
        model/tfidf_pipeline.joblib
    tfidf_logreg_binary_cv_5fold/
        result_card.json           (metricas medias dos 5 folds)
        predictions.csv            (concatenacao das 5 folds)
        fold_0/, fold_1/, ..., fold_4/
    tfidf_logreg_binary_test_set/
        result_card.json
        predictions.csv
    tfidf_logreg_multiclass_fixed_split/
        result_card.json
        confusion_matrix.csv
    tfidf_logreg_multiclass_cv_5fold/
    tfidf_logreg_multiclass_test_set/
```

8 diretorios por notebook; 24 no total apos 02+03+04.

### Como saber se deu certo

A ultima celula de cada notebook imprime um DataFrame com 6 linhas (uma por regime). Verifique:
- `primary` (F1 ou macro_F1) > 0.5 em todos
- `train_s`, `inf_s`, `size_mb` preenchidos
- `n_trials` = 60, `search_s` > 0

```bash
# Verificacao rapida via shell
uv run python -c "
import json
from pathlib import Path
for d in sorted(Path('artifacts/runs').glob('tfidf_*_*')):
    card = d / 'result_card.json'
    if card.exists():
        c = json.loads(card.read_text())
        print(f'{d.name:55s}  task={c[\"task\"]:10s}  regime={c[\"regime\"]:12s}  '
              f'primary={(c[\"metrics\"].get(\"f1\") or c[\"metrics\"].get(\"f1_mean\") or c[\"metrics\"].get(\"macro_f1\") or c[\"metrics\"].get(\"macro_f1_mean\") or 0):.4f}')
"
```

### Recuperacao de falhas

| Sintoma | Causa | Como resolver |
|---------|-------|---------------|
| `MemoryError` durante search | RAM insuficiente para `n_jobs=-1` | Edite a celula 1 do notebook: `N_JOBS = 4` (ou menor) |
| `After pruning, no terms remain` | Combinacao `min_df` alta em dataset pequeno | Esperado em smoke tests; ignorar — sklearn so descarta o trial |
| Notebook trava em `RandomizedSearchCV` | Demora normal — uma busca completa leva ~30-60 min | Espere ou reduza `N_ITER_BINARY`/`N_ITER_MULTI` na celula 1 |

---

## 5. Etapa 3 — BERT no Colab A100 (~1-3 dias)

**Objetivo**: para cada modelo BERT (bertimbau, finbert_ptbr, deb3rta_base), executar busca de hiperparametros (25 trials cada x 2 tarefas) e os 6 regimes com `best_params`.

### 5.1 Empacotar splits (LOCAL, ~30s)

```bash
uv run python scripts/colab_pack.py
```

Saida: `colab_splits.zip` no diretorio raiz do projeto. Contem train/val/test parquet + `cv_folds.json` + metadados. ~50-100 MB.

### 5.2 Upload para o Google Drive

1. Abra [Google Drive](https://drive.google.com)
2. Crie pasta `economy-classifier` na raiz (se ainda nao existe)
3. Faca upload do `colab_splits.zip` para esta pasta

Estrutura esperada:
```
My Drive/
  economy-classifier/
    colab_splits.zip
```

### 5.3 Push do branch para o GitHub

O notebook 05 clona o repositorio durante o bootstrap. Garanta que sua branch esta no GitHub:

```bash
git push origin main
```

Ou ajuste `REPO_BRANCH` na celula de bootstrap do notebook 05 se voce esta numa branch alternativa.

### 5.4 Abrir o notebook no Colab

1. [colab.research.google.com](https://colab.research.google.com) > Arquivo > Abrir notebook > GitHub
2. Cole `https://github.com/almeidadm/economy-classifier`
3. Selecione `notebooks/05_bert_colab.ipynb`

### 5.5 Selecionar GPU A100

1. Runtime > Change runtime type
2. Hardware accelerator: **A100 GPU**
3. Save

A primeira celula (`## 0. Verificacao de GPU`) emite **AVISO** se a GPU detectada nao for A100. Em T4, prossiga somente apos reduzir `N_ITER_BERT` ou comentar modelos em `MODELS`.

### 5.6 Executar todas as celulas

Runtime > Run all. O notebook:

1. Verifica GPU
2. Monta o Drive, clona o repo, instala dependencias
3. Extrai splits
4. Define `run_full_protocol(model_key, model_name)`
5. Loop sobre `MODELS` — para cada modelo:
   - Busca binaria (25 trials x ~10-15 min em A100 = ~4-6h)
   - Busca multiclasse (~4-6h)
   - 6 regimes com best_params (~6-10h)
6. Imprime sumario com 18 result cards

**Tempo total estimado**: ~30-60h em A100. **Conecte ao Colab Pro+ para sessoes mais longas**.

### 5.7 Backup automatico no Drive

Cada `result_card.json` e `search_result.json` e gravado em `My Drive/economy-classifier/runs/` em tempo real, conforme cada regime conclui. Se a sessao expirar:

1. Veja em `runs/bert_*` quais modelos/regimes ja completaram
2. Edite `MODELS = [("modelo_que_faltou", "...")]` na celula 5 do notebook
3. Re-execute a partir da celula 12 (`run_full_protocol`)

Os resultados ja salvos nao serao retreinados (cada `run_full_protocol` cria diretorios novos com timestamp implicito).

### 5.8 Baixar resultados (apos conclusao)

No Drive, navegue ate `My Drive/economy-classifier/runs/`. Selecione todos os diretorios `bert_*` e baixe como zip (Drive faz isso automaticamente). Salve como `colab_bert_results.zip` na sua maquina.

### 5.9 Integrar localmente (LOCAL, ~30s)

```bash
uv run python scripts/colab_unpack.py ~/Downloads/colab_bert_results.zip
```

Saida esperada:

```
Runs encontrados no zip: 24  (18 result cards + 6 search logs)
  - bert_bertimbau_binary_fixed_split
  - bert_bertimbau_binary_cv_5fold
  - bert_bertimbau_binary_test_set
  - bert_bertimbau_multiclass_fixed_split
  - bert_bertimbau_multiclass_cv_5fold
  - bert_bertimbau_multiclass_test_set
  - bert_bertimbau_search_binary
  - bert_bertimbau_search_multiclass
  - bert_finbert_ptbr_*
  - bert_deb3rta_base_*

Artefatos extraidos em: artifacts/runs/
```

---

## 6. Etapa 4 — Ensembles e comparacao (LOCAL, ~30 min)

**Objetivo**: combinar predicoes dos 6 modelos com 4 estrategias de ensemble (E1-E4) e produzir a tabela final.

### 6.1 Ensembles (notebook 07)

```bash
uv run jupyter notebook notebooks/07_ensemble.ipynb
# Run all cells
```

O notebook 07 le os `result_card.json` da pasta `artifacts/runs/` (todos os 6 modelos x regime `test_set`), aplica:

- **E1**: Votacao majoritaria (limiares >=4/6, >=5/6, >=6/6)
- **E2**: Votacao ponderada por F1 da validacao
- **E3**: Stacking (meta-LogReg treinado nas predicoes do val)
- **E4**: Concordancia como confianca

Saida em `artifacts/runs/ensemble_*/`.

### 6.2 Tabela comparativa final (notebook 12)

```bash
uv run jupyter notebook notebooks/12_comparacao.ipynb
# Run all cells
```

Agrega todos os `result_card.json` em uma tabela unica com colunas: `model_id, task, regime, primary, train_s, inf_s, size_mb, n_parameters, search_s, n_trials`. Exporta CSV + figuras (barplot comparativo, heatmap de concordancia, frente de Pareto custo-beneficio).

> Se o notebook 12 ainda nao existe, ele esta na lista de pendencias. Voce pode produzir uma tabela ad-hoc com:
> ```bash
> uv run python -c "
> import json
> from pathlib import Path
> import pandas as pd
> rows = []
> for card in Path('artifacts/runs').glob('*/result_card.json'):
>     c = json.loads(card.read_text())
>     m = c['metrics']
>     primary = m.get('f1') or m.get('f1_mean') or m.get('macro_f1') or m.get('macro_f1_mean')
>     rows.append({
>         'model_id': c['model_id'], 'task': c['task'], 'regime': c['regime'],
>         'primary': primary,
>         'train_s': c['cost'].get('train_seconds_mean'),
>         'size_mb': c['cost'].get('model_size_mb'),
>     })
> pd.DataFrame(rows).sort_values(['task','regime','primary'], ascending=[True,True,False]).to_csv('tabela_final.csv', index=False)
> print('Salvo em tabela_final.csv')
> "
> ```

---

## 7. Como interpretar os artefatos

### 7.1 result_card.json

```json
{
    "model_id": "tfidf_logreg",
    "task": "binary",
    "regime": "test_set",
    "metrics": {
        "f1": 0.8234,
        "precision": 0.8512,
        "recall": 0.7976,
        "accuracy": 0.9543,
        "auc_roc": 0.9712
    },
    "cost": {
        "train_seconds_mean": 23.4,
        "inference_seconds_mean": 1.2,
        "throughput_per_second": 13_858.3,
        "model_size_mb": 12.5,
        "n_parameters": 100_023,
        "hardware": "Local-CPU"
    },
    "config": { ... },
    "hyperparameter_search": {
        "best_params": {"C": 0.31, "ngram_range": [1,2], "min_df": 2, ...},
        "best_score": 0.8156,
        "n_trials": 60,
        "scoring": "f1",
        "search_seconds": 1850.3
    },
    "n_train_samples": 149_659,
    "n_eval_samples": 16_629,
    "predictions_path": "...",
    "git_commit": "a1b2c3d",
    "generated_at": "2026-04-26T..."
}
```

**Como ler**:
- `regime: test_set` → este e o numero que vai pra dissertacao
- `regime: cv_5fold` → reporte como `f1_mean ± f1_std`, e a evidencia de variancia
- `regime: fixed_split` → diagnostico apenas, nao e a metrica final
- `hyperparameter_search.best_score` → F1 da busca interna (sempre maior que `cv_5fold` por causa da otimizacao)

### 7.2 search_result.json

Versao expandida do `hyperparameter_search` do `result_card`, com **todos** os trials:

```json
{
    "best_params": {...},
    "best_score": 0.8156,
    "n_trials": 60,
    "scoring": "f1",
    "search_seconds": 1850.3,
    "search_space": {...},
    "trials": [
        {"trial": 0, "params": {...}, "mean_test_score": 0.7891, "std_test_score": 0.0123, ...},
        {"trial": 1, ...},
        ...
    ]
}
```

Util para auditoria (mostrar no artigo que a busca foi exaustiva) e para identificar trials com erro (campo `error`).

### 7.3 predictions.csv

Formato padrao em todos os modelos/regimes:

```csv
index,y_true,y_pred,y_score,method
0,1,1,0.923,logreg
1,0,0,0.087,logreg
```

- `index` aponta pra linha original do `articles.csv`
- `y_score` em `[0,1]` (probabilidade ou score normalizado)
- `method` permite concatenar predicoes de varios modelos sem perder a origem

---

## 8. FAQ

### Quanto tempo o pipeline inteiro leva?

| Etapa | Tempo | Hardware |
|-------|-------|----------|
| 1. Splits | ~5 min | Local |
| 2. TF-IDF (3 notebooks sequencial) | ~9-21h | Local CPU |
| 2. TF-IDF (paralelo, 3 terminais) | ~3-7h | Local CPU |
| 3. BERT (3 modelos x 2 tarefas) | 30-60h | Colab A100 |
| 4. Ensembles + comparacao | ~30 min | Local |
| **Total (sequencial)** | **~3-5 dias** | A100 |

### Posso pular a busca de hiperparametros?

**Nao**. O protocolo definido em `CLAUDE.md` exige `RandomizedSearchCV` antes dos 3 regimes para todos os modelos. Cards sem `hyperparameter_search` nao devem entrar na tabela final.

Se voce quer **debugar** sem rodar a busca: edite `N_ITER_BINARY=2`, `N_ITER_MULTI=2`, `CV_INNER_SPLITS=2` na celula 1 do notebook 02 (e similar para 03/04/05). Vai rodar em segundos. **Nao publique resultados desse modo**.

### Posso rodar BERT em T4 ao inves de A100?

Sim, mas:
- Reduza `N_ITER_BERT` para 5-10 (default 25)
- Ou comente modelos em `MODELS = [("bertimbau", ...)]`
- Ou rode so `binary` (comente as celulas multiclasse no notebook 05)

Cada uma dessas mitigacoes deve ser **declarada** na sessao de discussao do artigo como limitacao do orcamento computacional.

### O Colab desconectou, perdi tudo?

Nao. Cada `result_card.json` e gravado em tempo real em `My Drive/economy-classifier/runs/`. Reabra o notebook, edite `MODELS` para incluir so o que falta, e re-execute. Os modelos ja completos nao serao retreinados.

**Truque para evitar desconexao**: deixe uma aba do navegador aberta em primeiro plano com o Colab. O Colab Pro+ ainda assim pode desconectar apos ~24h.

### O `cv_5fold` e a `best_score` da busca sao a mesma coisa?

**Nao**. A `best_score` e a media de F1 nos 5 folds **internos** da busca (`cv_seed=43`). O regime `cv_5fold` reportado posteriormente usa os folds **externos** persistidos em `artifacts/splits/cv_folds.json` (`seed=42`). Sao particoes **diferentes** do mesmo pool train+val. Por construcao, `best_score` tende a ser ligeiramente maior que `cv_5fold.f1_mean` (otimizacao em folds proprios).

### Por que TF-IDF usa inner-CV e BERT usa val unico?

Custo computacional. TF-IDF: cada trial leva segundos, 5-fold inner = 5x; toleravel. BERT: cada trial leva 10-30 min em A100; 5-fold inner = 5x = ~1-2.5h por trial; 25 trials x 5 folds x 6 modelos x 2 tarefas = >1500 fits = inviavel mesmo em A100.

A assimetria fica registrada no `result_card.hyperparameter_search.scoring`. Na discussao do artigo, declare que a busca BERT subestima ligeiramente a generalizacao por causa do val unico.

---

## 9. Checklist final (antes de submeter o artigo)

- [ ] `artifacts/splits/` existe com 5 arquivos (incluindo `cv_folds.json`)
- [ ] 36 `result_card.json` em `artifacts/runs/` (6 modelos x 6 regimes)
- [ ] 12 `search_result.json` (6 modelos x 2 tarefas)
- [ ] Todos os cards tem `hyperparameter_search` preenchido (nao `null`)
- [ ] Tabela comparativa final gerada (CSV + figuras)
- [ ] McNemar entre os 2 melhores modelos no `test_set` reportado
- [ ] Todos os 200+ testes passam: `uv run pytest`
- [ ] Nenhum modelo viu o test_set durante busca ou treino dos demais regimes
- [ ] `git status` limpo (todos os notebooks executados commitados)

---

## 10. Onde pedir ajuda

| Problema | Onde |
|----------|------|
| Erro num notebook | Veja a saida da celula que falhou; cheque o traceback |
| Erro nos modulos `src/` | Rode `uv run pytest -v` para ver qual contrato quebrou |
| Duvida metodologica | `CLAUDE.md` (guardia metodologica) e `docs/arquitetura.md` |
| Estimativa de tempo | `docs/estimativa_recursos_computacionais.md` (adendo no final) |
| Setup do Colab | `docs/guia_colab.md` (passo a passo detalhado) |
