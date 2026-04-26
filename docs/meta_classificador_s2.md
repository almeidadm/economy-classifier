# Meta-Classificador por Stacking: Fundamentos e o Modelo S2

## Sumario

1. [Introducao](#1-introducao)
2. [Classificadores-base](#2-classificadores-base)
   - 2.1 [TF-IDF + LinearSVC](#21-tf-idf--linearsvc)
   - 2.2 [BERTimbau](#22-bertimbau)
3. [Stacking: a ideia de um meta-classificador](#3-stacking-a-ideia-de-um-meta-classificador)
4. [Regressao Logistica como meta-classificador](#4-regressao-logistica-como-meta-classificador)
5. [O modelo S2 deste projeto](#5-o-modelo-s2-deste-projeto)
   - 5.1 [Arquitetura](#51-arquitetura)
   - 5.2 [Parametros aprendidos](#52-parametros-aprendidos)
   - 5.3 [Exemplo numerico](#53-exemplo-numerico)
   - 5.4 [Resultados no conjunto de teste](#54-resultados-no-conjunto-de-teste)
6. [Por que o stacking funciona?](#6-por-que-o-stacking-funciona)
7. [Cuidados metodologicos](#7-cuidados-metodologicos)
8. [Links e recursos online](#8-links-e-recursos-online)
9. [Referencias bibliograficas](#9-referencias-bibliograficas)

---

## 1. Introducao

Na classificacao de textos, diferentes modelos capturam diferentes aspectos
da linguagem. Um modelo baseado em frequencia de palavras (TF-IDF) e bom em
detectar vocabulario discriminativo, enquanto um modelo de linguagem neural
(BERT) compreende relacoes contextuais e semanticas. A pergunta natural e:
**podemos combinar as predicoes de ambos para obter um resultado melhor?**

A tecnica de **stacking** (empilhamento) faz exatamente isso. Ela treina um
segundo modelo — o **meta-classificador** — que aprende a ponderar as saidas
dos classificadores-base. Em vez de usar regras fixas de votacao, o
meta-classificador *aprende com dados* qual peso dar a cada modelo.

Este documento explica os fundamentos matematicos da tecnica e como ela e
aplicada neste projeto com o modelo **S2**, que combina **BERTimbau** e
**LinearSVC** por meio de uma **Regressao Logistica**.

---

## 2. Classificadores-base

### 2.1 TF-IDF + LinearSVC

#### TF-IDF (Term Frequency — Inverse Document Frequency)

O TF-IDF transforma textos em vetores numericos. Cada dimensao do vetor
corresponde a um termo (palavra ou bigrama), e o valor reflete a importancia
do termo para aquele documento em relacao ao corpus inteiro.

**Term Frequency** com escala sublinear (usada neste projeto):

$$
\text{tf}(t, d) = 1 + \log(\text{contagem}(t, d))
$$

A escala sublinear (`sublinear_tf=True`) evita que um termo que aparece 100
vezes tenha 100x mais peso do que um que aparece 1 vez. Com o logaritmo, a
relacao passa a ser 1 vs. ~5.6.

**Inverse Document Frequency** com suavizacao:

$$
\text{idf}(t) = \log\!\left(\frac{1 + n}{1 + \text{df}(t)}\right) + 1
$$

onde $n$ e o total de documentos e $\text{df}(t)$ o numero de documentos que
contem o termo $t$. Termos que aparecem em todos os documentos recebem IDF
proximo de 1; termos raros recebem IDF alto.

**Peso final:**

$$
\text{tfidf}(t, d) = \text{tf}(t, d) \times \text{idf}(t)
$$

O vetor resultante e normalizado para norma L2 unitaria.

**Parametros do vetorizador neste projeto:**

| Parametro | Valor | Significado |
|-----------|-------|-------------|
| `max_features` | 50.000 | Maximo de termos no vocabulario |
| `ngram_range` | (1, 2) | Unigramas e bigramas |
| `sublinear_tf` | True | Escala logaritmica da frequencia |
| `min_df` | 2 | Ignora termos em menos de 2 docs |
| `max_df` | 0.95 | Ignora termos em mais de 95% dos docs |
| `strip_accents` | unicode | Remove acentos |
| `lowercase` | True | Converte para minusculas |

#### LinearSVC (Support Vector Classifier linear)

O LinearSVC encontra o hiperplano que separa as duas classes com a maior
margem possivel. Para um vetor de features $\mathbf{x}$, a funcao de decisao
e:

$$
f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b
$$

onde $\mathbf{w}$ e o vetor de pesos e $b$ o bias. A classe predita e:

$$
\hat{y} = \begin{cases} 1 & \text{se } f(\mathbf{x}) \geq 0 \\ 0 & \text{caso contrario} \end{cases}
$$

O treinamento minimiza a funcao de perda hinge com regularizacao L2:

$$
\min_{\mathbf{w}, b} \; \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \max\!\big(0, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)\big)
$$

O parametro $C = 1.0$ controla o trade-off entre margem e erros de treino.

#### Calibracao de probabilidades

O LinearSVC nao produz probabilidades nativamente — apenas o valor bruto
$f(\mathbf{x})$. Para obter probabilidades, usamos o
**CalibratedClassifierCV** com validacao cruzada de 3 folds. Ele ajusta uma
funcao sigmoide (metodo de Platt) ou uma regressao isotonica sobre as saidas
do SVC, mapeando $f(\mathbf{x})$ para uma probabilidade $p \in [0, 1]$.

O score final desse classificador e:

$$
s_{\text{svc}}(\mathbf{x}) = P(y = 1 \mid \mathbf{x})_{\text{calibrado}}
$$

#### Pipeline completo

```
Texto → Tokenizacao → TF-IDF (50K features) → LinearSVC → Calibracao → score ∈ [0, 1]
```

**Referencia tecnica:** Cortes & Vapnik (1995); Platt (1999) para calibracao.

---

### 2.2 BERTimbau

#### O que e BERT?

BERT (Bidirectional Encoder Representations from Transformers) e um modelo
de linguagem pre-treinado que aprende representacoes contextuais de palavras.
Diferente do TF-IDF, onde cada palavra tem um vetor fixo independente do
contexto, no BERT a representacao de "banco" muda conforme a frase seja
"banco de dados" ou "banco central".

O modelo e baseado na arquitetura **Transformer** (Vaswani et al., 2017),
usando apenas o encoder. O mecanismo central e a **self-attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

onde $Q$ (queries), $K$ (keys) e $V$ (values) sao projecoes lineares dos
embeddings de entrada, e $d_k$ e a dimensao das keys. Cada token "atende" a
todos os outros, capturando dependencias bidirecionais.

#### BERTimbau

**BERTimbau** (`neuralmind/bert-base-portuguese-cased`) e uma versao do BERT
pre-treinada especificamente em textos em portugues brasileiro, desenvolvida
por Souza, Nogueira e Lotufo (2020).

**Arquitetura:**

| Parametro | Valor |
|-----------|-------|
| Camadas (layers) | 12 |
| Dimensao oculta | 768 |
| Cabecas de atencao | 12 |
| Vocabulario | 29.794 tokens |
| Posicoes maximas | 512 tokens |

#### Fine-tuning para classificacao

No fine-tuning, uma camada linear de classificacao e adicionada sobre a
representacao do token `[CLS]`:

$$
\mathbf{z} = W_{\text{cls}} \cdot \mathbf{h}_{\text{[CLS]}} + \mathbf{b}_{\text{cls}}
$$

onde $\mathbf{h}_{\text{[CLS]}} \in \mathbb{R}^{768}$ e a representacao do
token `[CLS]` na ultima camada, $W_{\text{cls}} \in \mathbb{R}^{2 \times 768}$
sao pesos aprendidos, e $\mathbf{z} \in \mathbb{R}^2$ sao os logits para
as duas classes.

A probabilidade da classe positiva e obtida por **softmax**:

$$
s_{\text{bert}}(\mathbf{x}) = P(y = 1 \mid \mathbf{x}) = \frac{e^{z_1}}{e^{z_0} + e^{z_1}}
$$

**Parametros de fine-tuning neste projeto:**

| Parametro | Valor |
|-----------|-------|
| Learning rate | 2e-5 |
| Batch size | 8 (x2 acumulacao) |
| Epocas | 3 |
| Max length | 256 tokens |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Early stopping patience | 1 |
| Seed | 42 |

#### Pipeline completo

```
Texto → Tokenizacao WordPiece (max 256) → BERT Forward → Logits → Softmax → score ∈ [0, 1]
```

**Referencia tecnica:** Devlin et al. (2019); Souza, Nogueira e Lotufo (2020).

---

## 3. Stacking: a ideia de um meta-classificador

### O conceito

Stacking (Wolpert, 1992) e uma tecnica de ensemble em dois niveis:

- **Nivel 0 — Classificadores-base:** Cada modelo $M_k$ recebe o texto $x_i$
  e produz um score $s_k(x_i) \in [0, 1]$, representando a probabilidade
  estimada de $x_i$ pertencer a classe positiva.

- **Nivel 1 — Meta-classificador:** Um novo modelo recebe como entrada os
  scores dos classificadores-base e aprende a producao a predicao final.

Formalmente, dado um texto $x_i$ e $K$ classificadores-base:

$$
\mathbf{s}_i = \big[s_1(x_i), \; s_2(x_i), \; \dots, \; s_K(x_i)\big] \in \mathbb{R}^K
$$

O meta-classificador e uma funcao $g: \mathbb{R}^K \to \{0, 1\}$ que recebe
$\mathbf{s}_i$ e prediz a classe final:

$$
\hat{y}_i = g(\mathbf{s}_i)
$$

### Por que nao simplesmente votar?

A votacao simples (maioria) trata todos os modelos como iguais. Mas na
pratica os modelos tem acuracias diferentes e cometem erros em exemplos
diferentes. O meta-classificador:

1. **Aprende pesos diferentes** para cada modelo (o que a votacao ponderada
   tambem faz, mas com pesos fixos).
2. **Aprende interacoes** — por exemplo, pode aprender que quando o BERT esta
   confiante (score alto) mas o SVC nao, a resposta do BERT e mais confiavel.
3. **Calibra o limiar de decisao** automaticamente via o intercepto aprendido.

### Diagrama do fluxo

```
                     ┌─────────────────────────┐
                     │     Texto de entrada     │
                     └────────┬────────────────-┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
        ┌────────────────┐       ┌────────────────┐
        │   BERTimbau    │       │  TF-IDF + SVC  │
        │  (Nivel 0)     │       │   (Nivel 0)    │
        └───────┬────────┘       └───────┬────────┘
                │                        │
         s_bert �� [0,1]           s_svc ∈ [0,1]
                │                        │
                └───────────┬────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Regressao Logistica  │
                │     (Nivel 1)        │
                └───────────┬───────────┘
                            │
                    ┌───────┴───────┐
                    │               │
                    ▼               ▼
              y_score ∈ [0,1]   y_pred ∈ {0,1}
              (probabilidade)   (classe final)
```

---

## 4. Regressao Logistica como meta-classificador

### A funcao logistica (sigmoide)

A Regressao Logistica modela a probabilidade da classe positiva como:

$$
P(y = 1 \mid \mathbf{s}) = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

onde $z$ e uma combinacao linear dos scores de entrada:

$$
z = \beta_0 + \beta_1 s_1 + \beta_2 s_2 + \dots + \beta_K s_K
$$

- $\beta_0$ e o **intercepto** (bias)
- $\beta_k$ sao os **coeficientes** (pesos) para cada classificador-base
- $\sigma(\cdot)$ e a funcao **sigmoide**

### Interpretacao dos coeficientes

Os coeficientes tem interpretacao direta em termos de **log-odds**
(logaritmo da razao de chances):

$$
\log\!\left(\frac{P(y=1)}{P(y=0)}\right) = \beta_0 + \beta_1 s_1 + \beta_2 s_2
$$

Um coeficiente $\beta_k$ positivo significa que, a medida que o score do
modelo $k$ aumenta, a chance da classe positiva cresce. **Quanto maior o
coeficiente, mais o meta-classificador confia naquele modelo.**

### Treinamento

O treinamento minimiza a **entropia cruzada binaria** (binary cross-entropy)
com regularizacao L2:

$$
\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i \log \sigma(z_i) + (1 - y_i)\log\big(1 - \sigma(z_i)\big)\Big] + \frac{1}{2C}\|\boldsymbol{\beta}\|^2
$$

onde $C$ (default 1.0 no sklearn) controla a forca da regularizacao. O
otimizador utilizado e o **L-BFGS** (Limited-memory
Broyden-Fletcher-Goldfarb-Shanno), um metodo quasi-Newton eficiente para
problemas de media escala.

### Decisao final

$$
\hat{y} = \begin{cases} 1 \; (\text{mercado}) & \text{se } \sigma(z) \geq 0.5 \\ 0 \; (\text{outros}) & \text{se } \sigma(z) < 0.5 \end{cases}
$$

O limiar 0.5 corresponde a $z = 0$, ou seja, o ponto onde as log-odds sao
neutras.

---

## 5. O modelo S2 deste projeto

### 5.1 Arquitetura

O S2 e o ensemble **top-2** por stacking: combina os dois melhores
classificadores individuais por F1 no conjunto de validacao.

| Componente | Modelo | Papel |
|-----------|--------|-------|
| Base 1 | BERTimbau (`neuralmind/bert-base-portuguese-cased`) | Classificador neural contextual |
| Base 2 | LinearSVC + TF-IDF (calibrado) | Classificador linear sobre frequencia de termos |
| Meta | LogisticRegression (`solver=lbfgs`, `C=1.0`, `seed=42`) | Combinacao aprendida dos scores |

**Dados de treino do meta-classificador:** Conjunto de **validacao** (16% do
corpus, 26.606 textos), nunca usado para treinar os classificadores-base.
Isso evita **data leakage** — se treinassemos o meta-classificador nos mesmos
dados usados para treinar os modelos-base, os scores seriam otimistamente
enviesados.

### 5.2 Parametros aprendidos

A equacao do meta-classificador S2 com os coeficientes exatos:

$$
z = -6{,}6599 + 3{,}6745 \cdot s_{\text{bert}} + 4{,}9776 \cdot s_{\text{svc}}
$$

$$
P(\text{mercado} \mid \text{texto}) = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

| Parametro | Valor | Interpretacao |
|-----------|-------|---------------|
| $\beta_0$ (intercepto) | $-6{,}6599$ | Bias fortemente negativo — sem evidencia dos modelos, a predicao padrao e "outros" |
| $\beta_{\text{bert}}$ | $+3{,}6745$ | Peso do BERTimbau |
| $\beta_{\text{svc}}$ | $+4{,}9776$ | Peso do LinearSVC — **o meta-classificador confia mais no SVC** |

O coeficiente do LinearSVC e ~35% maior que o do BERTimbau. Isso indica que,
para este corpus especifico, o vocabulario discriminativo capturado pelo
TF-IDF (termos como "acao", "bolsa", "dolar") e um sinal mais forte do que
a compreensao contextual do BERT. Isso faz sentido para textos jornalisticos
com vocabulario economico bem definido.

### 5.3 Exemplo numerico

Considere o texto: *"Bolsa sobe 2% com expectativa de corte na Selic"*

**Passo 1 — Scores dos classificadores-base:**

| Modelo | Score | Interpretacao |
|--------|-------|---------------|
| BERTimbau | 0.92 | 92% de confianca em "mercado" |
| LinearSVC | 0.88 | 88% de confianca em "mercado" |

**Passo 2 — Combinacao linear:**

$$
z = -6{,}6599 + 3{,}6745 \times 0{,}92 + 4{,}9776 \times 0{,}88
$$

$$
z = -6{,}6599 + 3{,}3805 + 4{,}3803 = 1{,}1009
$$

**Passo 3 — Sigmoide:**

$$
P(\text{mercado}) = \frac{1}{1 + e^{-1{,}1009}} = \frac{1}{1 + 0{,}3327} = 0{,}7504
$$

**Passo 4 — Decisao:**

$$
0{,}7504 \geq 0{,}5 \implies \hat{y} = 1 \; (\text{mercado})
$$

Agora considere: *"Governo anuncia reforma ministerial"*

| Modelo | Score |
|--------|-------|
| BERTimbau | 0.15 |
| LinearSVC | 0.10 |

$$
z = -6{,}6599 + 3{,}6745 \times 0{,}15 + 4{,}9776 \times 0{,}10 = -6{,}6599 + 0{,}5512 + 0{,}4978 = -5{,}6109
$$

$$
P(\text{mercado}) = \frac{1}{1 + e^{5{,}6109}} = \frac{1}{1 + 273{,}1} = 0{,}0036
$$

$$
0{,}0036 < 0{,}5 \implies \hat{y} = 0 \; (\text{outros})
$$

#### Caso interessante: desacordo entre modelos

Quando os modelos discordam, o meta-classificador mostra seu valor. Considere
um texto ambiguo onde BERTimbau prediz 0.80 mas LinearSVC prediz 0.30:

$$
z = -6{,}6599 + 3{,}6745 \times 0{,}80 + 4{,}9776 \times 0{,}30 = -6{,}6599 + 2{,}9396 + 1{,}4933 = -2{,}2270
$$

$$
P(\text{mercado}) = \sigma(-2{,}2270) = 0{,}0975
$$

Mesmo com o BERT confiante em "mercado", o peso maior do SVC e o intercepto
negativo fazem a decisao pender para "outros". O meta-classificador aprendeu
que, neste corpus, a ausencia de vocabulario discriminativo (SVC baixo) e um
sinal forte.

### 5.4 Resultados no conjunto de teste

Metricas do S2 no conjunto de teste (que nao participou de nenhum treino):

| Metrica | Valor |
|---------|-------|
| **F1-score** | 0.8613 |
| Precision | 0.8184 |
| Recall | 0.9089 |
| Accuracy | 0.9631 |
| AUC-ROC | 0.9885 |

**Comparacao com os individuais:**

| Metodo | F1 (teste) | Diferenca para S2 |
|--------|-----------|-------------------|
| BERTimbau individual | ~0.80 | S2 superior |
| LinearSVC individual | ~0.85 | S2 superior |
| **S2 (stacking)** | **0.8613** | — |

O ensemble supera ambos os modelos individuais, demonstrando que a combinacao
aprendida extrai valor complementar.

---

## 6. Por que o stacking funciona?

### Diversidade de erros

A chave do stacking e a **diversidade** dos classificadores-base. Se dois
modelos cometem exatamente os mesmos erros, combina-los nao traz ganho.
BERTimbau e LinearSVC sao diversas por construcao:

| Aspecto | BERTimbau | LinearSVC + TF-IDF |
|---------|-----------|-------------------|
| Representacao | Contextual (atencao) | Bag-of-words (frequencia) |
| Parametros | ~110M | ~50K features |
| Captura | Semantica, relacoes distantes | Vocabulario discriminativo |
| Fraqueza tipica | Textos curtos sem contexto | Polissemia, contexto |

Quando o BERT erra por falta de vocabulario especifico, o TF-IDF acerta.
Quando o TF-IDF erra por ambiguidade de termos, o BERT acerta pelo contexto.

### Fundamentacao teorica

Dietterich (2000) identifica tres razoes pelas quais ensembles funcionam:

1. **Razao estatistica:** Com dados limitados, varios modelos podem ter
   desempenho similar no treino mas diferir na generalizacao. O ensemble
   reduz o risco de escolher um unico modelo ruim.
2. **Razao computacional:** Algoritmos de otimizacao podem ficar presos em
   minimos locais. Combinar multiplos modelos reduz essa dependencia.
3. **Razao representacional:** O espaco de hipoteses verdadeiro pode nao ser
   representavel por nenhum modelo individual, mas uma combinacao pode
   aproxima-lo melhor.

---

## 7. Cuidados metodologicos

### Data leakage

O meta-classificador e treinado no conjunto de **validacao**, nao no treino.
Se usassemos o treino, os scores dos modelos-base seriam inflados (ja que
eles viram esses dados), e o meta-classificador aprenderia a explorar essa
inflacao em vez de aprender combinacoes genuinas.

### Avaliacao no teste

Todas as metricas reportadas sao do conjunto de **teste** (20%), que nao
participou do treino dos modelos-base nem do meta-classificador. O split de
teste foi criado uma unica vez com `seed=42` e nunca e reutilizado.

### Balanceamento

O conjunto de validacao e teste preservam a distribuicao natural (~12.5%
positivos). Apenas o treino dos modelos-base usa downsample para
balanceamento.

---

## 8. Links e recursos online

### Stacking e ensembles

- [Stacked Generalization — Wolpert (1992), PDF original](https://doi.org/10.1016/S0893-6080(05)80023-1)
- [Scikit-learn: StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
- [Ensemble Methods — Scikit-learn User Guide](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization)
- [Machine Learning Mastery: Stacking Ensemble](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)

### Regressao Logistica

- [Scikit-learn: LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Stanford CS229: Logistic Regression (Andrew Ng)](https://cs229.stanford.edu/notes2022fall/main_notes.pdf)
- [StatQuest: Logistic Regression (video)](https://www.youtube.com/watch?v=yIYKR4sgzI8)

### TF-IDF e LinearSVC

- [Scikit-learn: TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Scikit-learn: LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- [Scikit-learn: CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
- [Wikipedia: tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

### BERT e BERTimbau

- [BERT paper: Devlin et al. (2019)](https://arxiv.org/abs/1810.04805)
- [BERTimbau paper: Souza, Nogueira e Lotufo (2020)](https://arxiv.org/abs/2003.09364)
- [Hugging Face: BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- [The Illustrated BERT (Jay Alammar)](https://jalammar.github.io/illustrated-bert/)

---

## 9. Referencias bibliograficas

- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297. https://doi.org/10.1007/BF00994018

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT 2019*, 4171-4186. https://arxiv.org/abs/1810.04805

- Dietterich, T. G. (2000). Ensemble Methods in Machine Learning. *Multiple Classifier Systems (MCS 2000)*, Lecture Notes in Computer Science, vol 1857. https://doi.org/10.1007/3-540-45014-9_1

- Platt, J. (1999). Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. *Advances in Large Margin Classifiers*, 61-74.

- Souza, F., Nogueira, R., & Lotufo, R. (2020). BERTimbau: Pretrained BERT Models for Brazilian Portuguese. *Proceedings of BRACIS 2020*, 403-417. https://arxiv.org/abs/2003.09364

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*, 5998-6008. https://arxiv.org/abs/1706.03762

- Wolpert, D. H. (1992). Stacked Generalization. *Neural Networks*, 5(2), 241-259. https://doi.org/10.1016/S0893-6080(05)80023-1
