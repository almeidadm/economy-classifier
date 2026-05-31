# REFERENCES.md — Referencias bibliograficas do projeto

> Levantamento das referencias canonicas que embasam as decisoes metodologicas do projeto (framework comparativo de classificadores de texto da Folha de Sao Paulo, formulacoes binaria e multiclasse 7+other). Organizado por topico, com formula bibliografica completa e justificativa do uso. Itens marcados como "model card / technical report" nao tem publicacao revisada por pares — citamos o repositorio oficial.

---

## 1. Splits estratificados e validacao cruzada k-fold estratificada

- **Kohavi, R. (1995).** A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection. In *Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI)*, vol. 2, pp. 1137–1143. https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf
  Estabelece empiricamente que k-fold estratificado (k=10, e por extensao k=5) tem melhor compromisso bias-variance que hold-out simples ou leave-one-out — embasa a escolha de `StratifiedKFold(5)` sobre o pool train+val.

- **Forman, G., & Scholz, M. (2010).** Apples-to-Apples in Cross-Validation Studies: Pitfalls in Classifier Performance Measurement. *ACM SIGKDD Explorations Newsletter*, 12(1), 49–57. https://doi.org/10.1145/1882471.1882479
  Mostra que a forma de agregar F1 entre folds (media de F1 por fold vs. F1 sobre predicoes concatenadas) e fonte de vies sistematico sob desbalanceamento — justifica reportar F1 medio +/- desvio entre os 5 folds, nao um unico F1 agregado.

- **Geisser, S. (1975).** The Predictive Sample Reuse Method with Applications. *Journal of the American Statistical Association*, 70(350), 320–328. https://doi.org/10.2307/2285815
  Formalizacao original do principio de cross-validation como estimador de erro de generalizacao — embasa o uso de CV como ground truth de robustez do modelo.

- **Stone, M. (1974).** Cross-Validatory Choice and Assessment of Statistical Predictions. *Journal of the Royal Statistical Society: Series B*, 36(2), 111–147. https://doi.org/10.1111/j.2517-6161.1974.tb00994.x
  Trabalho seminal sobre validacao cruzada — citacao historica para o principio metodologico.

---

## 2. Hold-out test set intocado

- **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7
  Capitulo 7 (Model Assessment and Selection) define o protocolo train/validation/test e proibe explicitamente o uso do test para selecionar modelo ou hiperparametros — embasa a regra "test fixo nunca usado para tuning, threshold ou early stopping".

- **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer. ISBN: 978-0-387-31073-2.
  Capitulo 1.3 estabelece a separacao treino/validacao/teste como condicao para estimativa nao-enviesada do erro de generalizacao — citacao canonica de livro-texto para o principio.

- **Russell, S. J., & Norvig, P. (2020).** *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. ISBN: 978-0134610993.
  Capitulo 19.4 reforca o protocolo de tres particoes e o anti-padrao "peeking at test set" — referencia didatica complementar.

---

## 3. F1-score como metrica primaria em dados desbalanceados

- **van Rijsbergen, C. J. (1979).** *Information Retrieval* (2nd ed.). Butterworth-Heinemann. http://www.dcs.gla.ac.uk/Keith/Preface.html
  Define F-measure como media harmonica de precisao e recall — origem historica da metrica F1 e justificativa de seu uso em recuperacao de informacao (cenario com classe positiva minoritaria, como `mercado`).

- **Powers, D. M. W. (2011).** Evaluation: From Precision, Recall and F-measure to ROC, Informedness, Markedness and Correlation. *Journal of Machine Learning Technologies*, 2(1), 37–63. https://arxiv.org/abs/2010.16061
  Analise critica de accuracy e por que F1, em conjunto com outras metricas, fornece avaliacao mais informativa em problemas de classificacao reais — embasa o anti-padrao "reportar accuracy como metrica principal" do CLAUDE.md.

- **Saito, T., & Rehmsmeier, M. (2015).** The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. *PLOS ONE*, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
  Demonstra empiricamente que ROC pode mascarar ma performance em datasets com forte desbalanceamento (ex: 87.5% de negativos) — embasa o uso de F1 e curvas PR como complemento ao AUC-ROC.

- **He, H., & Garcia, E. A. (2009).** Learning from Imbalanced Data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239
  Survey canonico sobre classificacao desbalanceada — embasa F1 e AUC sobre accuracy quando a distribuicao natural e mantida (val/test ~12.5% mercado).

---

## 4. Macro-F1 vs. Weighted-F1 vs. Micro-F1 para multiclasse

- **Sokolova, M., & Lapalme, G. (2009).** A Systematic Analysis of Performance Measures for Classification Tasks. *Information Processing & Management*, 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002
  Define formalmente macro/micro/weighted averaging e suas invariancias sob mudancas de distribuicao de classes — embasa o uso de macro-F1 como metrica primaria multiclasse e weighted-F1 como complemento.

- **Opitz, J., & Burst, S. (2019).** Macro F1 and Macro F1. arXiv:1911.03347. https://arxiv.org/abs/1911.03347
  Mostra que existem duas formulas distintas para "macro F1" (media de F1s vs. F1 da media de P/R) que podem divergir e produzir rankings opostos — justifica reportar explicitamente a definicao usada (sklearn `average='macro'`).

---

## 5. AUC-ROC e curvas PR para classificacao binaria desbalanceada

- **Fawcett, T. (2006).** An Introduction to ROC Analysis. *Pattern Recognition Letters*, 27(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010
  Tutorial canonico sobre ROC e AUC — embasa o uso de AUC como metrica complementar a F1 no regime binario.

- **Davis, J., & Goadrich, M. (2006).** The Relationship Between Precision-Recall and ROC Curves. In *Proceedings of the 23rd International Conference on Machine Learning (ICML)*, pp. 233–240. https://doi.org/10.1145/1143844.1143874
  Estabelece a correspondencia formal entre os dois espacos e mostra que dominancia em ROC implica dominancia em PR (mas nao vice-versa) — embasa o uso de PR curves alem de ROC sob desbalanceamento.

- **Saito, T., & Rehmsmeier, M. (2015).** The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. *PLOS ONE*, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
  (Citado tambem na secao 3.) Argumenta empiricamente que PR e mais sensivel em cenarios como o do projeto.

---

## 6. Teste de McNemar para comparacao pareada de classificadores

- **McNemar, Q. (1947).** Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages. *Psychometrika*, 12(2), 153–157. https://doi.org/10.1007/BF02295996
  Definicao original do teste — base estatistica para comparacao pareada de proporcoes.

- **Dietterich, T. G. (1998).** Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation*, 10(7), 1895–1923. https://doi.org/10.1162/089976698300017197
  Compara 5 testes para comparacao de classificadores e identifica McNemar como tendo baixa probabilidade de erro tipo I quando os dois modelos sao avaliados no mesmo conjunto de teste — embasa a escolha do McNemar em `evaluation.compute_mcnemar_test`.

- **Salzberg, S. L. (1997).** On Comparing Classifiers: Pitfalls to Avoid and a Recommended Approach. *Data Mining and Knowledge Discovery*, 1(3), 317–328. https://doi.org/10.1023/A:1009752403260
  Alerta contra comparar F1 sem teste estatistico — embasa o anti-padrao "comparar metodos sem McNemar" do CLAUDE.md.

- **Demsar, J. (2006).** Statistical Comparisons of Classifiers over Multiple Data Sets. *Journal of Machine Learning Research*, 7, 1–30. https://www.jmlr.org/papers/v7/demsar06a.html
  Complementar — embasa o paradigma de testes pareados nao-parametricos quando ha multiplos modelos sendo comparados.

---

## 7. Matriz de confusao normalizada

- **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer.
  Secao 1.5 introduz matriz de confusao como instrumento canonico de diagnostico de erros por classe — embasa a opcao `normalize="true"` em `evaluation.compute_confusion_matrix` (recall por classe na diagonal).

- **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning* (2nd ed.). Springer.
  Discussao complementar sobre matrizes de confusao e a importancia da normalizacao para datasets desbalanceados.

- **Pedregosa, F., et al. (2011).** Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. https://www.jmlr.org/papers/v12/pedregosa11a.html
  Documentacao da implementacao `sklearn.metrics.confusion_matrix` usada no projeto.

---

## 8. TF-IDF e bag-of-words para classificacao de texto

- **Salton, G., & Buckley, C. (1988).** Term-Weighting Approaches in Automatic Text Retrieval. *Information Processing & Management*, 24(5), 513–523. https://doi.org/10.1016/0306-4573(88)90021-0
  Trabalho canonico sobre esquemas de ponderacao tf-idf — embasa toda a familia de pipelines TF-IDF do projeto.

- **Sparck Jones, K. (1972).** A Statistical Interpretation of Term Specificity and Its Application in Retrieval. *Journal of Documentation*, 28(1), 11–21. https://doi.org/10.1108/eb026526
  Origem do conceito de IDF — fundamento teorico para o componente IDF do TF-IDF.

- **Manning, C. D., Raghavan, P., & Schutze, H. (2008).** *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/
  Capitulos 6 e 13 fornecem formulacao moderna de TF-IDF e classificacao de texto — referencia padrao de livro-texto. Justifica tambem `sublinear_tf` (logaritmizacao do tf, equivalente ao "log frequency weighting" do livro).

- **Joachims, T. (1998).** Text Categorization with Support Vector Machines: Learning with Many Relevant Features. In *Proceedings of the European Conference on Machine Learning (ECML)*, pp. 137–142. https://doi.org/10.1007/BFb0026683
  Estabelece TF-IDF + SVM linear como baseline forte para classificacao de texto — embasa o pipeline `tfidf + LinearSVC`.

---

## 9. Logistic Regression, Linear SVM e Multinomial Naive Bayes para classificacao de texto

- **Joachims, T. (1998).** *(citado acima)* — embasa Linear SVM como classificador de texto.

- **McCallum, A., & Nigam, K. (1998).** A Comparison of Event Models for Naive Bayes Text Classification. In *AAAI-98 Workshop on Learning for Text Categorization*, pp. 41–48. https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf
  Compara Bernoulli e Multinomial NB e estabelece o Multinomial NB como variante padrao para textos longos — embasa `MultinomialNB` no projeto.

- **Wang, S., & Manning, C. D. (2012).** Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. In *Proceedings of the 50th Annual Meeting of the ACL*, vol. 2, pp. 90–94. https://aclanthology.org/P12-2018/
  Mostra que NB e SVM com bigrams + log-count ratios sao baselines muito dificeis de bater — justifica `ngram_range=(1,2)` e a presenca simultanea de NB, LogReg e SVM como baselines fortes.

- **Pedregosa, F., et al. (2011).** Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. https://www.jmlr.org/papers/v12/pedregosa11a.html
  Implementacao dos tres classificadores (`LogisticRegression`, `LinearSVC`, `MultinomialNB`), do `Pipeline`, do `RandomizedSearchCV` e do `train_test_split` estratificado — embasa toda a stack TF-IDF.

- **Platt, J. C. (2000).** Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. In *Advances in Large Margin Classifiers*, MIT Press. https://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf
  Calibracao sigmoidal de scores de SVM — embasa o uso de `CalibratedClassifierCV` em torno de `LinearSVC` para expor `predict_proba` (necessario para AUC-ROC e curvas PR).

---

## 10. One-vs-Rest (OvR) para multiclasse

- **Rifkin, R., & Klautau, A. (2004).** In Defense of One-Vs-All Classification. *Journal of Machine Learning Research*, 5, 101–141. https://www.jmlr.org/papers/v5/rifkin04a.html
  Argumenta empiricamente que OvR (binario com N classificadores) e competitivo com solucoes nativas multiclasse quando os classificadores base sao bem regularizados — embasa a estrategia `strategy="ovr"` em `tfidf.TfidfMulticlassConfig`.

- **Allwein, E. L., Schapire, R. E., & Singer, Y. (2001).** Reducing Multiclass to Binary: A Unifying Approach for Margin Classifiers. *Journal of Machine Learning Research*, 1, 113–141. https://www.jmlr.org/papers/v1/allwein00a.html
  Framework teorico para reducoes multiclasse-para-binario, do qual OvR e caso particular.

---

## 11. BERT e transfer learning para classificacao de texto

- **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017).** Attention Is All You Need. In *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30. https://arxiv.org/abs/1706.03762
  Arquitetura Transformer base — embasa toda a familia de modelos pre-treinados usados (BERTimbau, FinBERT-PT-BR, DeB3RTa).

- **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In *Proceedings of NAACL-HLT 2019*, pp. 4171–4186. https://aclanthology.org/N19-1423/
  Modelo BERT original e protocolo de fine-tuning para classificacao — embasa a arquitetura `AutoModelForSequenceClassification` usada em `bert.train_bert_classifier`.

- **Howard, J., & Ruder, S. (2018).** Universal Language Model Fine-tuning for Text Classification. In *Proceedings of the 56th Annual Meeting of the ACL*, pp. 328–339. https://aclanthology.org/P18-1031/
  ULMFiT — primeiro a estabelecer empiricamente o paradigma "pretrain LM + fine-tune para classificacao" como superior ao treino do zero com poucos exemplos rotulados; justifica a escolha de fine-tuning (vs. classificadores from scratch) para o regime do projeto.

---

## 12. BERTimbau (BERT em PT-BR)

- **Souza, F., Nogueira, R., & Lotufo, R. (2020).** BERTimbau: Pretrained BERT Models for Brazilian Portuguese. In *Brazilian Conference on Intelligent Systems (BRACIS 2020)*, *Lecture Notes in Computer Science*, vol. 12319, pp. 403–417. Springer. https://doi.org/10.1007/978-3-030-61377-8_28
  Modelo `neuralmind/bert-base-portuguese-cased` usado no projeto (`MODEL_REGISTRY["bertimbau"]`); justifica a escolha de um BERT pre-treinado em corpus brasileiro vs. mBERT.

- **Souza, F., Nogueira, R., & Lotufo, R. (2023).** BERT models for Brazilian Portuguese: Pretraining, evaluation and tokenization analysis. *Applied Soft Computing*, 149(A), 110901. https://doi.org/10.1016/j.asoc.2023.110901
  Versao estendida do paper BERTimbau com analise de tokenizacao — referencia adicional para discussao metodologica.

---

## 13. FinBERT-PT-BR e BERT financeiro

- **Araci, D. (2019).** FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063. https://arxiv.org/abs/1908.10063
  FinBERT original (ingles) — paradigma de domain adaptation para textos financeiros via further-pretraining; embasa a nocao de dominio especializado para textos economicos.

- **Santos, L. L., Bianchi, R. A. C., & Costa, A. H. R. (2023).** FinBERT-PT-BR: Analise de Sentimentos de Textos em Portugues do Mercado Financeiro. In *Anais do II Brazilian Workshop on Artificial Intelligence in Finance (BWAIF)*, pp. 144–155. SBC. https://doi.org/10.5753/bwaif.2023.230669
  Paper original do `lucas-leme/FinBERT-PT-BR` usado em `MODEL_REGISTRY["finbert_ptbr"]` — modelo BERT pre-treinado em ~1.4M textos financeiros em PT-BR e fine-tuned para sentiment.

- **Yang, Y., Uy, M. C. S., & Huang, A. (2020).** FinBERT: A Pretrained Language Model for Financial Communications. arXiv:2006.08097. https://arxiv.org/abs/2006.08097
  Variante FinBERT (FinBERT-tone) — referencia adicional sobre domain-adaptation financeira.

---

## 14. DeBERTa e DeB3RTa (DeBERTa em PT-BR)

- **He, P., Liu, X., Gao, J., & Chen, W. (2021).** DeBERTa: Decoding-Enhanced BERT with Disentangled Attention. In *International Conference on Learning Representations (ICLR 2021)*. arXiv:2006.03654. https://arxiv.org/abs/2006.03654
  Arquitetura DeBERTa (disentangled attention + enhanced mask decoder) — base do `higopires/DeB3RTa-base` usado no projeto.

- **He, P., Gao, J., & Chen, W. (2023).** DeBERTaV3: Improving DeBERTa Using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. In *International Conference on Learning Representations (ICLR 2023)*. arXiv:2111.09543. https://arxiv.org/abs/2111.09543
  Versao V3 — base direta do DeB3RTa (DeBERTaV3 em PT-BR financeiro).

- **Pires, H., Paucar, A., & Carvalho, J. P. (2025).** DeB3RTa: A Transformer-Based Model for the Portuguese Financial Domain. *Big Data and Cognitive Computing*, 9(3), 51. MDPI. https://doi.org/10.3390/bdcc9030051
  Paper original do `higopires/DeB3RTa-base` usado em `MODEL_REGISTRY["deb3rta_base"]` — DeBERTaV3 com mixed-domain pretraining (financas, politica, gestao, contabilidade) em PT-BR.

---

## 15. Random search para hyperparameter optimization

- **Bergstra, J., & Bengio, Y. (2012).** Random Search for Hyper-Parameter Optimization. *Journal of Machine Learning Research*, 13, 281–305. https://www.jmlr.org/papers/v13/bergstra12a.html
  Demonstra empirica e teoricamente que random search domina grid search com mesmo orcamento computacional, especialmente quando poucos hiperparametros realmente importam — embasa a escolha de `RandomizedSearchCV` (TF-IDF) e do loop random custom (BERT) em `hyperparameter_search.py`.

---

## 16. HuggingFace Transformers / Trainer

- **Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q., & Rush, A. M. (2020).** Transformers: State-of-the-Art Natural Language Processing. In *Proceedings of EMNLP 2020: System Demonstrations*, pp. 38–45. https://aclanthology.org/2020.emnlp-demos.6/
  Biblioteca `transformers` usada para `AutoTokenizer`, `AutoModelForSequenceClassification`, `Trainer`, `TrainingArguments`, `EarlyStoppingCallback` — base de toda a stack BERT do projeto.

- **Lhoest, Q., et al. (2021).** Datasets: A Community Library for Natural Language Processing. In *Proceedings of EMNLP 2021: System Demonstrations*, pp. 175–184. https://aclanthology.org/2021.emnlp-demos.21/
  Biblioteca `datasets` (HF Dataset usado em `_tokenize_dataframe`).

---

## 17. Ensembles: majority voting, stacking, bagging

- **Wolpert, D. H. (1992).** Stacked Generalization. *Neural Networks*, 5(2), 241–259. https://doi.org/10.1016/S0893-6080(05)80023-1
  Definicao original de stacking (meta-classificador treinado em saidas de classificadores base) — embasa `ensemble.train_stacking_classifier`.

- **Breiman, L. (1996).** Bagging Predictors. *Machine Learning*, 24(2), 123–140. https://doi.org/10.1007/BF00058655
  Bagging e o argumento de variancia — embasa o ganho esperado de combinar classificadores diversos.

- **Dietterich, T. G. (2000).** Ensemble Methods in Machine Learning. In *Multiple Classifier Systems (MCS 2000)*, *LNCS* 1857, pp. 1–15. Springer. https://doi.org/10.1007/3-540-45014-9_1
  Survey canonico sobre metodos de ensemble (voting, bagging, boosting, stacking) e os tres motivos para combinar (estatistico, computacional, representacional) — referencia didatica para a secao de ensembles.

- **Kuncheva, L. I. (2014).** *Combining Pattern Classifiers: Methods and Algorithms* (2nd ed.). Wiley. ISBN: 978-1-118-31523-1.
  Livro-texto canonico sobre ensembles — embasa majority voting, weighted voting e a relacao entre acordo entre classificadores e ganho do ensemble.

- **Lam, L., & Suen, C. Y. (1997).** Application of Majority Voting to Pattern Recognition: An Analysis of Its Behavior and Performance. *IEEE Transactions on Systems, Man, and Cybernetics — Part A: Systems and Humans*, 27(5), 553–568. https://doi.org/10.1109/3468.618255
  Analise teorica do voto majoritario — embasa `ensemble.majority_vote`.

---

## 18. Stacking treinado em hold-out (validacao) para evitar leakage

- **Wolpert, D. H. (1992).** *(citado na secao 17)* — define o protocolo: meta-classificador deve ser treinado em predicoes out-of-fold ou em hold-out, nunca nas predicoes do conjunto onde os classificadores base foram treinados.

- **Ting, K. M., & Witten, I. H. (1999).** Issues in Stacked Generalization. *Journal of Artificial Intelligence Research*, 10, 271–289. https://doi.org/10.1613/jair.594
  Discute em detalhe a obrigatoriedade do hold-out (ou CV out-of-fold) para o meta-learner e como o leakage corrompe o stacking — embasa a regra do CLAUDE.md "stacking treinado na validacao, nao no treino".

---

## 19. Acordo entre classificadores: Fleiss' Kappa, Cohen's Kappa

- **Cohen, J. (1960).** A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1), 37–46. https://doi.org/10.1177/001316446002000104
  Definicao original do kappa de Cohen — embasa `ensemble.compute_agreement_matrix` (Cohen's Kappa pareado entre classificadores) e `llm_review.compute_review_concordance` (kappa entre dois rotuladores).

- **Fleiss, J. L. (1971).** Measuring Nominal Scale Agreement Among Many Raters. *Psychological Bulletin*, 76(5), 378–382. https://doi.org/10.1037/h0031619
  Generalizacao para N raters — embasa `ensemble.compute_fleiss_kappa`.

- **Landis, J. R., & Koch, G. G. (1977).** The Measurement of Observer Agreement for Categorical Data. *Biometrics*, 33(1), 159–174. https://doi.org/10.2307/2529310
  Tabela de interpretacao de kappa (slight/fair/moderate/substantial/almost perfect) usada para reportar acordo nos resultados.

---

## 20. Class imbalance: balanceamento so no treino, nao em val/test

- **He, H., & Garcia, E. A. (2009).** Learning from Imbalanced Data. *IEEE TKDE*, 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239
  Survey definidor — embasa o principio "val e teste preservam distribuicao natural; balanceamento apenas no treino".

- **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).** SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357. https://doi.org/10.1613/jair.953
  SMOTE — referencia canonica para oversampling. Embora o pipeline atual nao use SMOTE (o legado 64/16/20 usava downsampling deterministico via `build_balanced_training_frame`), e a referencia obrigatoria ao discutir alternativas.

- **Japkowicz, N., & Stephen, S. (2002).** The Class Imbalance Problem: A Systematic Study. *Intelligent Data Analysis*, 6(5), 429–449. https://doi.org/10.3233/IDA-2002-6504
  Estudo sistematico mostrando que avaliar em conjunto balanceado infla metricas — embasa o anti-padrao "balancear val ou teste" do CLAUDE.md.

- **King, G., & Zeng, L. (2001).** Logistic Regression in Rare Events Data. *Political Analysis*, 9(2), 137–163. https://doi.org/10.1093/oxfordjournals.pan.a004868
  Justificativa estatistica para `class_weight='balanced'` em LogReg/SVM como alternativa a oversampling — embasa o uso de `class_weight` no espaco de busca de hiperparametros.

---

## 21. Reproducibilidade em ML (seeds, artefatos, lockfiles)

- **Pineau, J., Vincent-Lamarre, P., Sinha, K., Lariviere, V., Beygelzimer, A., d'Alche-Buc, F., Fox, E., & Larochelle, H. (2021).** Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program). *Journal of Machine Learning Research*, 22(164), 1–20. https://www.jmlr.org/papers/v22/20-303.html
  Define o checklist de reprodutibilidade do NeurIPS — embasa o conjunto de praticas do projeto: seeds fixas, artefatos versionados, `uv.lock`, `result_card.json`, `git_commit` em metadata.

- **Gundersen, O. E., & Kjensmo, S. (2018).** State of the Art: Reproducibility in Artificial Intelligence. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1), 1644–1651. https://doi.org/10.1609/aaai.v32i1.11503
  Quantifica o problema (apenas 20–30% das variaveis necessarias sao documentadas em papers de IA) — embasa o `result_card.json` schema com metadados explicitos.

- **Sculley, D., Snoek, J., Wiltschko, A., & Rahimi, A. (2018).** Winner's Curse? On Pace, Progress, and Empirical Rigor. In *ICLR 2018 Workshop Track*. https://openreview.net/forum?id=rJWF0Fywf
  Critica influente sobre rigor empirico em ML — reforca a obrigatoriedade de variancia (CV) e teste estatistico (McNemar) sobre comparacoes pontuais.

---

## 22. Classificacao de texto em portugues brasileiro

- **Hartmann, N., Fonseca, E., Shulby, C., Treviso, M., Silva, J., & Aluisio, S. (2017).** Portuguese Word Embeddings: Evaluating on Word Analogies and Natural Language Tasks. In *Proceedings of the 11th Brazilian Symposium in Information and Human Language Technology (STIL)*, pp. 122–131. https://aclanthology.org/W17-6615/ (arXiv:1708.06025)
  Trabalho canonico sobre embeddings em PT-BR (FastText, GloVe, Wang2Vec, Word2Vec) — referencia historica de baseline pre-BERT para NLP em PT-BR.

- **Pires, T., Schlinger, E., & Garrette, D. (2019).** How Multilingual is Multilingual BERT? In *Proceedings of ACL 2019*, pp. 4996–5001. https://aclanthology.org/P19-1493/
  Avaliacao do mBERT em transferencia cross-lingual — justifica preferir BERTimbau (monolingue PT) sobre mBERT.

---

## 23. LLMs zero-shot / few-shot para classificacao

- **Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. (2020).** Language Models are Few-Shot Learners. In *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 1877–1901. https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html (arXiv:2005.14165)
  GPT-3 e estabelecimento do paradigma in-context learning (zero-shot e few-shot via prompt) — embasa o protocolo de classificacao por LLM em `llm_review.py` (funcao `build_review_prompt` zero-shot e `build_review_prompt_few_shot`).

- **Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., & Le, Q. V. (2022).** Finetuned Language Models Are Zero-Shot Learners. In *International Conference on Learning Representations (ICLR 2022)*. arXiv:2109.01652. https://arxiv.org/abs/2109.01652
  FLAN — instruction tuning como mecanismo que torna LLMs eficazes em zero-shot; justifica usar checkpoints `*-Instruct` (Qwen2.5-Instruct, Mistral-Instruct) em vez dos modelos base.

- **Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., & Iwasawa, Y. (2022).** Large Language Models are Zero-Shot Reasoners. In *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35. arXiv:2205.11916. https://arxiv.org/abs/2205.11916
  Zero-Shot CoT — base teorica para prompts que podem incluir raciocinio explicito; situa o estado da arte mesmo que o projeto use prompts diretos sem CoT.

- **Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H., & Neubig, G. (2023).** Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing. *ACM Computing Surveys*, 55(9), Article 195. https://doi.org/10.1145/3560815
  Survey canonico de prompting — embasa o desenho do `SYSTEM_PROMPT` e `SYSTEM_PROMPT_MULTICLASS` com restricao de saida (uma palavra) e a estrutura few-shot interleaved.

---

## 24. Modelos LLM utilizados (Mistral, Qwen, Llama, Sabia)

- **Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. (2023).** Mistral 7B. arXiv:2310.06825. https://arxiv.org/abs/2310.06825
  Modelo `mistralai/Mistral-7B-Instruct-v0.3` usado em `llm_review.LLM_REGISTRY`.

- **Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., et al. (Qwen Team) (2024).** Qwen2.5 Technical Report. arXiv:2412.15115. https://arxiv.org/abs/2412.15115
  Modelo `Qwen/Qwen2.5-7B-Instruct` usado em `llm_review.LLM_REGISTRY`.

- **Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al. (2023).** Qwen Technical Report. arXiv:2309.16609. https://arxiv.org/abs/2309.16609
  Versao 1.x do Qwen — referencia historica para a linha do modelo.

- **Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E., Azhar, F., et al. (2023).** LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971. https://arxiv.org/abs/2302.13971
  Paper original do LLaMA — fundamento da familia.

- **Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. (2023).** Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288. https://arxiv.org/abs/2307.09288
  Llama 2 — referencia para o lineage de Llama-3.1-8B-Instruct (presente comentado em `LLM_REGISTRY` aguardando aprovacao de gating).

- **Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., et al. (Meta Llama Team) (2024).** The Llama 3 Herd of Models. arXiv:2407.21783. https://arxiv.org/abs/2407.21783
  Llama 3 / 3.1 — referencia para o `meta-llama/Llama-3.1-8B-Instruct` (gated, presente comentado em `LLM_REGISTRY`).

- **Pires, R., Abonizio, H., Almeida, T. S., & Nogueira, R. (2023).** Sabia: Portuguese Large Language Models. In *Brazilian Conference on Intelligent Systems (BRACIS 2023)*, *LNCS* 14197. arXiv:2304.07880. https://arxiv.org/abs/2304.07880
  Familia Sabia original (continued pretraining de GPT-J/LLaMA em PT) — base do `sabia-7b`.

- **Almeida, T. S., Abonizio, H., Nogueira, R., & Pires, R. (2024).** Sabia-2: A New Generation of Portuguese Large Language Models. arXiv:2403.09887. https://arxiv.org/abs/2403.09887
  Sabia-2 — geracao intermediaria da familia, base do `sabia-3` e `sabia-4` via APIs Maritaca.

- **Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D. de L., Hendricks, L. A., Welbl, J., Clark, A., et al. (2022).** Training Compute-Optimal Large Language Models (Chinchilla). arXiv:2203.15556. https://arxiv.org/abs/2203.15556
  Lei de escala compute-optimal — referencia adicional ao discutir o porque da escolha de modelos 7B (sweet spot custo/qualidade) sobre 70B+ no projeto.

---

## 25. Referencias adicionais identificadas pela inspecao do codigo

> Topicos presentes no codigo que merecem citacao explicita.

### 25.1. Sublinear TF (logaritmizacao do term frequency)

- **Manning, C. D., Raghavan, P., & Schutze, H. (2008).** *Introduction to Information Retrieval*, capitulo 6.4.2 ("Sublinear tf scaling"). Cambridge University Press. https://nlp.stanford.edu/IR-book/html/htmledition/sublinear-tf-scaling-1.html
  Justifica `sublinear_tf=True` em `TfidfVectorizer`: tf logaritmizado (1 + log(tf)) reduz o peso de termos com altissima frequencia intra-documento, importante para textos longos como artigos da Folha.

### 25.2. Calibracao de scores de SVM (`CalibratedClassifierCV`)

- **Platt, J. C. (2000).** Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. In *Advances in Large Margin Classifiers*, MIT Press. https://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf
  Calibracao sigmoidal — embasa `CalibratedClassifierCV(LinearSVC, cv=3)` em `tfidf._build_pipeline` para expor `predict_proba` necessario ao calculo de AUC-ROC e curvas PR.

- **Niculescu-Mizil, A., & Caruana, R. (2005).** Predicting Good Probabilities with Supervised Learning. In *Proceedings of ICML 2005*, pp. 625–632. https://doi.org/10.1145/1102351.1102430
  Analise empirica das tecnicas de calibracao — referencia complementar.

### 25.3. Early stopping em fine-tuning de redes neurais

- **Prechelt, L. (1998).** Early Stopping — But When? In *Neural Networks: Tricks of the Trade*, *LNCS* 1524, pp. 55–69. Springer. https://doi.org/10.1007/3-540-49430-8_3
  Embasamento teorico para o uso de `EarlyStoppingCallback` em `bert.train_bert_classifier` (monitorando F1 na validacao, paciencia configuravel).

### 25.4. Mixed precision (FP16) no treino

- **Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., & Wu, H. (2018).** Mixed Precision Training. In *International Conference on Learning Representations (ICLR 2018)*. arXiv:1710.03740. https://arxiv.org/abs/1710.03740
  Embasa `fp16=torch.cuda.is_available()` em `TrainingArguments` — necessario para fit dos BERTs (especialmente DeB3RTa) em GPUs Colab L4/A100.

### 25.5. AdamW (otimizador padrao do HF Trainer)

- **Loshchilov, I., & Hutter, F. (2019).** Decoupled Weight Decay Regularization. In *International Conference on Learning Representations (ICLR 2019)*. arXiv:1711.05101. https://arxiv.org/abs/1711.05101
  Otimizador AdamW — usado por padrao pelo HF Trainer em `train_bert_classifier`; justifica os hiperparametros `learning_rate` e `weight_decay` no espaco de busca BERT.

### 25.6. Linear warmup learning rate schedule

- **Goyal, P., Dollar, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., & He, K. (2017).** Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. arXiv:1706.02677. https://arxiv.org/abs/1706.02677
  Origem do warmup de learning rate adotado por padrao em fine-tuning de BERT — embasa o hiperparametro `warmup_ratio` no espaco de busca BERT.

### 25.7. Gradient accumulation

- **Ott, M., Edunov, S., Grangier, D., & Auli, M. (2018).** Scaling Neural Machine Translation. In *Proceedings of WMT 2018*, pp. 1–9. https://aclanthology.org/W18-6301/
  Tecnica de gradient accumulation para simular batch sizes maiores que cabem em memoria — embasa `gradient_accumulation_steps` no espaco de busca BERT (necessario em Colab L4/T4).

### 25.8. Tokenizacao WordPiece / BPE

- **Sennrich, R., Haddow, B., & Birch, A. (2016).** Neural Machine Translation of Rare Words with Subword Units (BPE). In *Proceedings of ACL 2016*, pp. 1715–1725. https://aclanthology.org/P16-1162/
  Subword tokenization — referencia canonica que embasa os tokenizers usados em todos os BERTs do projeto.

- **Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M., Macherey, W., Krikun, M., Cao, Y., Gao, Q., Macherey, K., et al. (2016).** Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation (WordPiece). arXiv:1609.08144. https://arxiv.org/abs/1609.08144
  WordPiece — variante usada pelo BERTimbau.

### 25.9. PyTorch (framework)

- **Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al. (2019).** PyTorch: An Imperative Style, High-Performance Deep Learning Library. In *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, pp. 8024–8035. https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library
  Framework usado pelos modelos BERT/LLM — citacao obrigatoria para reprodutibilidade.

### 25.10. NumPy / pandas / SciPy (stack cientifica)

- **Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., et al. (2020).** Array Programming with NumPy. *Nature*, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2

- **McKinney, W. (2010).** Data Structures for Statistical Computing in Python. In *Proceedings of the 9th Python in Science Conference (SciPy)*, pp. 56–61. https://doi.org/10.25080/Majora-92bf1922-00a

- **Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., et al. (SciPy 1.0 Contributors) (2020).** SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2
  Stack numerica usada em todo o projeto (incluindo `scipy.stats.chi2` no teste de McNemar e `scipy.stats.loguniform` no random search).

### 25.11. Matplotlib / Seaborn (visualizacao)

- **Hunter, J. D. (2007).** Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

- **Waskom, M. L. (2021).** Seaborn: Statistical Data Visualization. *Journal of Open Source Software*, 6(60), 3021. https://doi.org/10.21105/joss.03021
  Bibliotecas usadas em `visualization.py` para confusion matrix, ROC/PR curves e heatmaps de Kappa.

### 25.12. Gerenciador de dependencias

- **Astral.** *uv: An extremely fast Python package and project manager.* https://github.com/astral-sh/uv
  Sem publicacao revisada — citar como ferramenta. Embasa `uv.lock` para dependencias deterministas (item de reprodutibilidade).

---

## 26. Dataset FolhaUOL e trabalhos relacionados que o utilizam

> O corpus do projeto e derivado do dataset publico "News of the Brazilian Newspaper" (FolhaUOL). Esta secao reune a citacao de origem do dataset e os trabalhos previos que o utilizaram, fornecendo contexto comparativo direto para a dissertacao.

### 26.1. Citacao do dataset

- **Santana, M. R. O. (2019).** *News of the Brazilian Newspaper* [Data set]. Kaggle. https://www.kaggle.com/datasets/marlesson/news-of-the-site-folhauol
  Dataset original com 167.053 noticias da Folha de Sao Paulo (jan/2015 a set/2017), distribuidas de forma desbalanceada em 48 secoes editoriais. E a fonte primaria do corpus do projeto, do qual sao derivados (i) o esquema binario `mercado` vs `outros` e (ii) o esquema multiclasse 7+other (poder, colunas, mercado, esporte, mundo, cotidiano, ilustrada, outros).

### 26.2. Garcia, Shiguihara & Berton (2024) — analise comparativa multi-metodo no FolhaUOL

- **Garcia, K., Shiguihara, P., & Berton, L. (2024).** Breaking news: Unveiling a new dataset for Portuguese news classification and comparative analysis of approaches. *PLOS ONE*, 19(1), e0296929. https://doi.org/10.1371/journal.pone.0296929
  Trabalho diretamente comparavel: introduz um novo corpus em PT-BR (WikiNews) e, em paralelo, executa analise comparativa no FolhaUOL apos preprocessamento que descarta categorias com poucas entradas e secoes "nao-noticia" (analoga a heuristica do projeto que separa `colunas` no esquema multiclasse). O subset resultante tem 96.819 documentos em 5 categorias: poder (22.022), mercado (20.970), esporte (19.730), mundo (17.130), cotidiano (16.967) — alinhamento direto com 5 das 7 classes do esquema multiclasse do projeto. Compara SVM (BoW, TF-IDF), CNN, DJINN e BERT (com embeddings fastText), reportando BERT como melhor acuracia e SVM+TF-IDF como melhor compromisso acuracia/tempo. Embasa diretamente: (a) a escolha das 7 classes-alvo do esquema multiclasse, (b) o paralelismo TF-IDF + BERT no projeto, (c) a discussao de custo-beneficio (`result_card.json`), e (d) a justificativa metodologica para retirar `colunas` ou tratar sua heterogeneidade como limitacao.

### 26.3. Alcoforado et al. (2022) — ZeroBERTo, zero-shot no FolhaUOL

- **Alcoforado, A., Ferraz, T. P., Gerber, R., Bustos, E., Oliveira, A. S., Veloso, B. M., Siqueira, F. L., & Reali Costa, A. H. (2022).** ZeroBERTo: Leveraging Zero-Shot Text Classification by Topic Modeling. In *Computational Processing of the Portuguese Language (PROPOR 2022)*, *Lecture Notes in Computer Science*, vol. 13208, pp. 125–136. Springer. https://doi.org/10.1007/978-3-030-98305-5_12 (arXiv:2201.01337)
  Propoe um pipeline zero-shot que combina topic modeling nao-supervisionado com classificacao por similaridade semantica sobre rotulos verbalizados, evitando o custo do XLM-R em textos longos. Avalia exatamente no FolhaUOL e supera XLM-R em ~12 pontos de F1. Embasa diretamente: (a) o protocolo zero-shot do `llm_review.py` (rotulos verbalizados em PT-BR, restricao da saida a uma palavra), (b) a comparacao "encoder fine-tuned (BERT/BERTimbau/DeB3RTa) vs LLM zero-shot" como eixo central da dissertacao, e (c) a justificativa de avaliar tambem em few-shot dado que zero-shot puro pode subestimar a capacidade dos LLMs modernos.

### 26.4. Posicionamento do projeto vs. trabalhos previos no FolhaUOL

| Trabalho | Esquema | N classes | Metodos | Lacuna que o projeto endereca |
|----------|---------|-----------|---------|-------------------------------|
| Santana (2019) | dataset cru | 48 secoes | — | fornece o corpus, sem analise |
| Alcoforado et al. (2022) | multiclasse zero-shot | (subset) | XLM-R, ZeroBERTo | nao compara com fine-tuning supervisionado nem usa modelos PT-BR especializados (BERTimbau, FinBERT-PT-BR, DeB3RTa) |
| Garcia, Shiguihara & Berton (2024) | multiclasse | 5 (poder, mercado, esporte, mundo, cotidiano) | SVM, CNN, DJINN, BERT (multilingue + fastText) | nao usa BERTs PT-BR especializados; nao reporta variancia de CV; nao testa McNemar; nao avalia LLMs zero/few-shot; nao trata o caso binario `mercado` vs resto |
| **Este projeto** | **binario + multiclasse 7+other** | **2 / 8** | **TF-IDF (LogReg/SVM/NB) + BERTimbau/FinBERT-PT-BR/DeB3RTa + LLM zero/few-shot + ensembles** | unifica os tres paradigmas no mesmo split, com `RandomizedSearchCV`, CV 5-fold, McNemar, kappa entre classificadores e `result_card.json` padronizado |

---

## 27. Calibracao de probabilidades (Brier score, ECE)

> O projeto reporta Brier score e Expected Calibration Error (ECE) como metricas secundarias sempre que `y_score` for uma probabilidade calibrada (TF-IDF + `CalibratedClassifierCV`, BERT softmax). As referencias abaixo embasam a interpretacao e a obrigatoriedade de declarar `N/A` para LLM com `y_score` deterministico.

- **Brier, G. W. (1950).** Verification of Forecasts Expressed in Terms of Probability. *Monthly Weather Review*, 78(1), 1–3. https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2
  Definicao original do Brier score como erro quadratico medio entre probabilidades preditas e desfechos binarios — embasa `compute_brier_score` em `evaluation.py`.

- **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).** On Calibration of Modern Neural Networks. In *Proceedings of the 34th International Conference on Machine Learning (ICML 2017)*, vol. 70, pp. 1321–1330. arXiv:1706.04599. https://arxiv.org/abs/1706.04599
  Define formalmente Expected Calibration Error (ECE) via binning e demonstra empiricamente que redes neurais profundas modernas (incluindo BERT-base) sao sistematicamente miscalibradas mesmo quando atingem alta acuracia — embasa `compute_ece` em `evaluation.py` e justifica reportar ECE alem de Brier para diagnosticar overconfidence dos BERTs.

- **Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015).** Obtaining Well Calibrated Probabilities Using Bayesian Binning. In *Proceedings of the 29th AAAI Conference on Artificial Intelligence*, pp. 2901–2907. https://ojs.aaai.org/index.php/AAAI/article/view/9602
  Origem do conceito de binning equal-width que Guo et al. (2017) adotam para o ECE — referencia tecnica para a escolha de `n_bins` em `compute_ece`.

- **DeGroot, M. H., & Fienberg, S. E. (1983).** The Comparison and Evaluation of Forecasters. *The Statistician*, 32(1/2), 12–22. https://doi.org/10.2307/2987588
  Fundamento estatistico classico de reliability e refinement (decomposicao do Brier score) — referencia historica para o paradigma de calibracao.

---

## 28. Datasets out-of-distribution (OOD) para avaliacao fora de dominio

> O notebook 45 (`45_ood_evaluation.ipynb`) avalia os 6 modelos in-domain (TF-IDF x 3 + BERT x 3) em 4 corpora externos: `Fake.Br`, `FakeRecogna`, `PortugueseNewsDataset` e `RecognaSumm`. Estes dois primeiros tem paper canonico (refs abaixo); os outros dois sao recursos publicos sem paper academico autonomo e devem ser citados via URL/Kaggle.

- **Monteiro, R. A., Santos, R. L. S., Pardo, T. A. S., de Almeida, T. A., Ruiz, E. E. S., & Vale, O. A. (2018).** Contributions to the Study of Fake News in Portuguese: New Corpus and Automatic Detection Results. In *Computational Processing of the Portuguese Language (PROPOR 2018)*, *Lecture Notes in Computer Science*, vol. 11122, pp. 324–334. Springer. https://doi.org/10.1007/978-3-319-99722-3_33
  Apresenta o corpus **Fake.Br** com 7.200 pares (true/fake) majoritariamente politicos em PT-BR — usado no notebook 45 como corpus OOD politico-geral. Embasa a Manchete D do `stil_strategy_note.md` (TF-IDF NB e o mais robusto fora de dominio politico).

- **Garcia, G. L., Afonso, L. C. S., & Papa, J. P. (2022).** FakeRecogna: A New Brazilian Corpus for Fake News Detection. In *Computational Processing of the Portuguese Language (PROPOR 2022)*, *Lecture Notes in Computer Science*, vol. 13208, pp. 57–67. Springer. https://doi.org/10.1007/978-3-030-98305-5_6
  Apresenta o corpus **FakeRecogna** multi-veiculo com 11.872 noticias rotuladas — usado no notebook 45, especialmente o recorte `economia UOL` (n=11.872) onde FinBERT-PT-BR recupera vantagem (F1 0,69). Evidencia aditiva da Manchete A: pretraining financeiro compensa apenas quando o downstream tambem e financeiro especializado.

---

## 29. Domain-specific BERT (paisagem internacional)

> A secao 13 ja cobre o lineage FinBERT (Araci 2019, Yang 2020) e o eixo PT-BR (Santos 2023 FinBERT-PT-BR, Pires 2025 DeB3RTa). As referencias abaixo completam a paisagem internacional de domain-adaptation via further-pretraining em BERT — necessarias para enquadrar a Manchete A no estado da arte mais amplo: "quando pretraining de dominio compensa?".

- **Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2020).** BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*, 36(4), 1234–1240. https://doi.org/10.1093/bioinformatics/btz682
  Exemplar mais citado de domain-specific BERT (biomedico, PubMed + PMC). Estabelece o protocolo de "continued pretraining" sobre BERT-base que e a base metodologica do FinBERT (Araci 2019, Yang 2020) e do FinBERT-PT-BR (Santos 2023). Junto com SciBERT, ancora a discussao "quando pretraining de dominio compensa?".

- **Beltagy, I., Lo, K., & Cohan, A. (2019).** SciBERT: A Pretrained Language Model for Scientific Text. In *Proceedings of EMNLP-IJCNLP 2019*, pp. 3615–3620. https://doi.org/10.18653/v1/D19-1371
  SciBERT (Semantic Scholar corpus, biomedico + ciencia da computacao) — referencia complementar a BioBERT para o paradigma "from-scratch pretraining em dominio" (vs. continued pretraining). Util para a discussao da Manchete A: o DeB3RTa usa mixed-domain from-scratch, similar a SciBERT, mas com resultado pior em-dominio do que o BERTimbau geral (continued pretraining nao aconteceu).

---

## 30. Paisagem de LLMs PT-BR (complemento a secao 24)

> A secao 24 cobre os LLMs efetivamente avaliados no projeto (Mistral, Qwen, Sabia) e o lineage Llama. As referencias abaixo completam a paisagem PT-BR/PT de modelos generativos abertos publicados em 2023–2025 — necessarias para enquadrar a Manchete C ("LLMs zero/few-shot perdem para BERT fine-tunado") em relacao ao estado da arte regional, e para responder a pergunta "por que nao avaliaram Sabia-2/Bode/Tucano?".

- **Larcher, C. H. N., Piau, M., Finardi, P., Gengo, P., Esposito, P., & Caridade, V. (2023).** Cabrita: closing the gap for foreign languages. arXiv:2308.11878. https://arxiv.org/abs/2308.11878
  Cabrita: continued pretraining de LLaMA-1 em ~3B tokens PT-BR. Marco historico — uma das primeiras adaptacoes LLaMA → PT publicada, contemporanea de Sabia (Pires 2023). Util para citar quando se discute a cronologia de LLMs PT-BR.

- **Garcia, G. L., Paiola, P. H., Morelli, L. H., Candido, G., Candido Jr., A., Jodas, D. S., Afonso, L. C. S., Guilherme, I. R., Penteado, B. E., & Papa, J. P. (2024).** Introducing Bode: A Fine-Tuned Large Language Model for Portuguese Prompt-Based Task. arXiv:2401.02909. https://arxiv.org/abs/2401.02909
  Bode: LLaMA-2 fine-tunado em Alpaca-PT, totalmente aberto. Comparavel direto a Sabia (proprietary) e Tucano (from-scratch). Notar que os mesmos autores publicaram FakeRecogna (secao 28) — autoria sobreposta indica grupo Unesp/Bauru consolidado em recursos PT-BR.

- **Rodrigues, J., Gomes, L., Silva, J., Branco, A., Santos, R., Cardoso, H. L., & Osorio, T. (2023).** Advancing Neural Encoding of Portuguese with Transformer Albertina PT-*. In *Brazilian Conference on Intelligent Systems (BRACIS 2023)*, *Lecture Notes in Computer Science*, vol. 14195. Springer. https://doi.org/10.1007/978-3-031-49008-8_35 (arXiv:2305.06721)
  Albertina PT-* (PT-PT e PT-BR): DeBERTa-base continued pretraining sobre brWaC. Encoder, nao decoder — concorrente direto de BERTimbau e referencia obrigatoria se a dissertacao discutir alternativas encoder PT-BR alem de BERTimbau.

- **Santos, R., Silva, J., Gomes, L., Rodrigues, J., & Branco, A. (2024).** Advancing Generative AI for Portuguese with Open Decoder Gervasio PT*. arXiv:2402.18766. https://arxiv.org/abs/2402.18766
  Gervasio PT-*: decoder aberto PT (PORTULAN/NOVA Lisboa). Complementa Albertina (encoder) com decoder; util para cobertura PT-PT no panorama luso-brasileiro alem de Sabia/Bode.

- **Lopes, R., Magalhaes, J., & Semedo, D. (2024).** GlorIA — A Generative and Open Large Language Model for Portuguese. arXiv:2402.12969. https://arxiv.org/abs/2402.12969
  GlorIA: LLM PT (NOVA School Lisboa), 1.3B parametros aberto. Foco em PT-PT mas relevante para o panorama luso-brasileiro.

- **Correa, N. K., Sen, A., Falk, S., & Fatimah, S. (2025).** Tucano: Advancing Neural Text Generation for Portuguese. *Patterns* (Cell Press), 6(6), 101325. https://doi.org/10.1016/j.patter.2025.101325 (arXiv:2411.07854)
  Tucano: LLM PT-BR pretreinado **from-scratch** (nao continued pretraining), publicado em venue indexado da Cell Press. Diferenciador metodologico relevante — Sabia/Bode/Cabrita usam continued pretraining, Tucano nao. Citar como evidencia recente de que o paradigma from-scratch tambem e viavel em PT-BR.

**Justificativa metodologica para o paper STIL nao usar LLM PT-BR especifico:** o paper avalia Mistral-7B-Instruct-v0.3 e Qwen2.5-7B-Instruct, NAO Sabia/Bode/Tucano. Recomenda-se adicionar uma frase honesta na secao 4 do paper explicando: "Restringimos a avaliacao LLM aos modelos abertos multilinguais mais usados na pratica (Mistral-7B, Qwen2.5-7B); a comparacao com LLMs PT-BR especializados (Sabia-2, Bode, Tucano) fica como trabalho futuro pois exigiria infraestrutura de inferencia separada para modelos proprietarios (Sabia-2) e tratamento dedicado a adapter/LoRA-only models." Sem isso, o reviewer perguntara "por que nao Sabia?".

---

## 31. Analise qualitativa e quantitativa de erros (FN/FP) em classificacao

> O notebook `44_error_analysis_cross_task.ipynb` realiza analise de erros com foco em `mercado` (falsos positivos e falsos negativos), inspecionando casos individuais e identificando padroes sistematicos (ex.: confusao `colunas` vs `mercado` na multiclasse). As referencias abaixo embasam o protocolo de error analysis adotado, organizadas em cinco subtopicos.

### 31.1. Survey e fundamentos

- **Belinkov, Y., & Glass, J. (2019).** Analysis Methods in Neural Language Processing: A Survey. *Transactions of the Association for Computational Linguistics*, 7, 49–72. https://doi.org/10.1162/tacl_a_00254 (arXiv:1812.08951)
  Survey canonico que organiza os metodos de analise de modelos neurais em PLN em quatro eixos (analise por sondagem, visualizacao, exemplos adversariais, desafio de generalizacao) — referencia introdutoria obrigatoria para qualquer secao de "Error Analysis".

### 31.2. Frameworks de testes comportamentais e taxonomia de erros

- **Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020).** Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. In *Proceedings of the 58th Annual Meeting of the ACL*, pp. 4902–4912. https://doi.org/10.18653/v1/2020.acl-main.442 (Best Paper Award ACL 2020)
  CheckList: taxonomia de tres tipos de teste (MFT — Minimum Functionality, INV — Invariance, DIR — Directional) e capability matrix (vocabulario, negacao, NER, robustez a typos, etc.). Embasa diretamente o protocolo de organizar FN/FP por categoria linguistica (ex.: "FPs em artigos com vocabulario financeiro mas sem conteudo de mercado") em vez de listar erros sem estrutura.

- **Wu, T., Ribeiro, M. T., Heer, J., & Weld, D. S. (2019).** Errudite: Scalable, Reproducible, and Testable Error Analysis. In *Proceedings of the 57th Annual Meeting of the ACL*, pp. 747–763. https://doi.org/10.18653/v1/P19-1073
  Errudite: DSL e ferramenta para definir grupos de instancias (`group`), atribuir causas (`attribute`) e propor counterfactual rewrites — operacionaliza a transicao de "olhar exemplos individuais" para "validar hipoteses sobre subgrupos de erro". Diretamente aplicavel ao notebook 44 (FNs `mercado` por subdominio: macroeconomia, varejo, mercado financeiro, etc.).

- **Goel, K., Rajani, N. F., Vig, J., Tan, S., Wu, J., Zheng, S., Xiong, C., Bansal, M., & Re, C. (2021).** Robustness Gym: Unifying the NLP Evaluation Landscape. In *Proceedings of NAACL-HLT 2021: System Demonstrations*, pp. 42–55. https://doi.org/10.18653/v1/2021.naacl-demos.6
  Framework que unifica CheckList + slicing + adversariais + transformacoes em um pipeline unico de avaliacao com cards de robustez. Util como meta-referencia quando a secao de error analysis combina varias estrategias.

### 31.3. Slicing automatico — descobrir subgrupos com erro alto

- **Chung, Y., Kraska, T., Polyzotis, N., Tae, K. H., & Whang, S. E. (2019).** Slice Finder: Automated Data Slicing for Model Validation. In *Proceedings of the 35th IEEE International Conference on Data Engineering (ICDE 2019)*, pp. 33–44. https://doi.org/10.1109/ICDE.2019.00139
  Slice Finder: algoritmo (lattice search + false discovery control) para encontrar automaticamente subgrupos com desempenho significativamente pior que o overall — substitui inspecao manual por busca sistematica. Aplicavel ao FolhaUOL via slices por ano, secao editorial vizinha, comprimento de artigo.

- **Eyuboglu, S., Varma, M., Saab, K., Delbrouck, J.-B., Lee-Messer, C., Dunnmon, J., Zou, J., & Re, C. (2022).** Domino: Discovering Systematic Errors with Cross-Modal Embeddings. In *International Conference on Learning Representations (ICLR 2022)*. arXiv:2203.14960. https://arxiv.org/abs/2203.14960
  Domino: encontra slices coerentes via mixture models sobre embeddings (sem precisar de metadata pre-existente) e os descreve em linguagem natural via CLIP. Util quando os FNs/FPs nao caem em categorias obvias (secao editorial, autor) e o pesquisador precisa descobrir os agrupamentos.

### 31.4. Contrastes e categorizacao linguistica de erros

- **Gardner, M., Artzi, Y., Basmov, V., Berant, J., Bogin, B., Chen, S., Dasigi, P., Dua, D., Elazar, Y., Gottumukkala, A., et al. (2020).** Evaluating Models' Local Decision Boundaries via Contrast Sets. In *Findings of EMNLP 2020*, pp. 1307–1323. https://doi.org/10.18653/v1/2020.findings-emnlp.117
  Contrast Sets: protocolo de edicao minima manual de exemplos para flipar o gold label, isolando capacidades especificas. Para o projeto, util como inspiracao metodologica — em vez de avaliar so na distribuicao natural, construir um conjunto pequeno de contrastes mercado/colunas e mercado/outros para diagnostico fino.

- **Naik, A., Ravichander, A., Sadeh, N., Rose, C., & Neubig, G. (2018).** Stress Test Evaluation for Natural Language Inference. In *Proceedings of the 27th International Conference on Computational Linguistics (COLING 2018)*, pp. 2340–2353. arXiv:1806.00692. https://arxiv.org/abs/1806.00692
  Categoriza erros de NLI em seis classes (antonimia, negacao, sobreposicao lexical, falacia gramatical, comprimento, ruido) e mostra que cada uma expoe vulnerabilidades distintas. Modelo conceitual para categorizar FN/FP do `mercado` por mecanismo causal (vocabulario superficial, contexto editorial, comprimento do artigo, etc.).

### 31.5. Vieses de anotacao como fonte sistematica de FN/FP

- **Gururangan, S., Swayamdipta, S., Levy, O., Schwartz, R., Bowman, S., & Smith, N. A. (2018).** Annotation Artifacts in Natural Language Inference Data. In *Proceedings of NAACL-HLT 2018*, pp. 107–112. https://doi.org/10.18653/v1/n18-2017
  Demonstra que artefatos de anotacao (pistas superficiais correlacionadas com o label) permitem que modelos atinjam alta acuracia sem entender a tarefa — e que esses mesmos artefatos explicam grande parte dos FN/FP residuais. Embasa a discussao no projeto sobre validade de construto da label `mercado` (secao editorial vs conteudo): a confusao mediada por modelo entre `colunas` e `mercado` pode ser interpretada como artefato da rotulagem editorial, nao deficiencia do modelo.

### 31.6. Explicacao instance-level (inspecao qualitativa de FN/FP individuais)

> Quando o notebook 44 mostra um caso especifico de FN/FP (`predictions.csv` filtrado), as ferramentas abaixo permitem responder "quais tokens/features mais contribuiram para essa predicao?" — convertendo intuicao em evidencia atribuivel.

- **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).** "Why Should I Trust You?": Explaining the Predictions of Any Classifier. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 1135–1144. https://doi.org/10.1145/2939672.2939778 (versao demo NAACL: https://doi.org/10.18653/v1/n16-3020)
  LIME: explicacao local por perturbacao linear em torno de uma instancia. Aplicavel diretamente aos modelos TF-IDF do projeto (cada token vira uma feature interpretavel) para entender por que um artigo especifico foi FN ou FP.

- **Ribeiro, M. T., Singh, S., & Guestrin, C. (2018).** Anchors: High-Precision Model-Agnostic Explanations. In *Proceedings of the 32nd AAAI Conference on Artificial Intelligence*, pp. 1527–1535. https://doi.org/10.1609/aaai.v32i1.11491
  Anchors: extrai regras de decisao locais com precision garantida ("se contem 'bolsa' e nao contem 'opiniao' entao previsto `mercado` com 95% precision no entorno"). Complementa LIME ao oferecer explicacoes mais interpretaveis para texto.

- **Lundberg, S. M., & Lee, S.-I. (2017).** A Unified Approach to Interpreting Model Predictions. In *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 4765–4774. arXiv:1705.07874. https://arxiv.org/abs/1705.07874
  SHAP: framework unificado de feature attribution baseado em valores de Shapley, com garantias teoricas de consistencia. Aplicavel a TF-IDF (KernelSHAP) e a BERT (DeepSHAP via captum/integrated gradients).

- **Wallace, E., Tuyls, J., Wang, J., Subramanian, S., Gardner, M., & Singh, S. (2019).** AllenNLP Interpret: A Framework for Explaining Predictions of NLP Models. In *Proceedings of EMNLP-IJCNLP 2019: System Demonstrations*, pp. 7–12. https://doi.org/10.18653/v1/D19-3002
  Framework que reune saliency maps (integrated gradients, smoothgrad), input reduction e adversarial attacks sob uma interface unica — referencia tecnica quando se reporta saliencia de tokens para BERTimbau/FinBERT-PT-BR em FN/FP especificos.

### 31.7. Contrafactuais e triggers adversariais para diagnostico

- **Wu, T., Ribeiro, M. T., Heer, J., & Weld, D. S. (2021).** Polyjuice: Generating Counterfactuals for Explaining, Evaluating, and Improving Models. In *Proceedings of the 59th Annual Meeting of the ACL*, pp. 6707–6723. https://doi.org/10.18653/v1/2021.acl-long.523
  Polyjuice: gerador de contrafactuais controlados (negacao, troca de quantificador, troca de entidade) para identificar quais perturbacoes flipam predicoes. Aplicavel a `mercado` para verificar robustez: trocar "alta do dolar" por "queda do dolar" deveria preservar a label, trocar "Bolsa de Valores" por "Bolsa de Lisboa" deveria preservar tambem.

- **Wallace, E., Feng, S., Kandpal, N., Gardner, M., & Singh, S. (2019).** Universal Adversarial Triggers for Attacking and Analyzing NLP. In *Proceedings of EMNLP-IJCNLP 2019*, pp. 2153–2162. https://doi.org/10.18653/v1/d19-1221
  Demonstra triggers universais (sequencias curtas de tokens que, anexadas a qualquer entrada, forcam uma predicao alvo) como mecanismo de stress-test. Util como referencia para a discussao "quao fragil e o classificador a entradas adversariais?".

---

## Notas finais

1. **Citacoes a conferir antes da submissao da dissertacao:**
   - O DOI exato do paper FinBERT-PT-BR pode variar entre versoes SBC; usar o ID em https://sol.sbc.org.br/index.php/bwaif para a versao final.
   - O paper DeB3RTa (Pires et al. 2025) deve ser conferido na MDPI em https://www.mdpi.com/2504-2289/9/3/51 para autores e ano definitivos.
   - Caso o `sabia-4` venha a ser usado via API e nao haja technical report formal, citar como "Maritaca AI. Sabia-4 [API]. https://www.maritaca.ai".
   - Tucano (Correa et al. 2025, Patterns): a versao arXiv e 2411.07854; a versao em Patterns/Cell tem DOI 10.1016/j.patter.2025.101325 — usar a versao Patterns como canonica por estar em venue indexado e peer-reviewed.

2. **Referencias gerais recomendadas para a fundamentacao teorica:** *The Hundred-Page Machine Learning Book* (Burkov 2019) e *Deep Learning* (Goodfellow, Bengio & Courville 2016) caso a dissertacao tenha uma secao introdutoria de ML.
