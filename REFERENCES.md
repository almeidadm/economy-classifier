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

## Notas finais

1. **Citacoes a conferir antes da submissao da dissertacao:**
   - O DOI exato do paper FinBERT-PT-BR pode variar entre versoes SBC; usar o ID em https://sol.sbc.org.br/index.php/bwaif para a versao final.
   - O paper DeB3RTa (Pires et al. 2025) deve ser conferido na MDPI em https://www.mdpi.com/2504-2289/9/3/51 para autores e ano definitivos.
   - Caso o `sabia-4` venha a ser usado via API e nao haja technical report formal, citar como "Maritaca AI. Sabia-4 [API]. https://www.maritaca.ai".

2. **Referencias gerais recomendadas para a fundamentacao teorica:** *The Hundred-Page Machine Learning Book* (Burkov 2019) e *Deep Learning* (Goodfellow, Bengio & Courville 2016) caso a dissertacao tenha uma secao introdutoria de ML.
