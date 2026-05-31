# Bibliografia consolidada - STIL paper economy-classifier

_Gerado em 2026-05-21. Fontes: OpenAlex API + Hugging Face Hub (model cards)._

Total: **28 referencias canonicas** confirmadas, cobrindo as 3 manchetes (A, C, D), todos os pilares metodologicos (McNemar+Bonferroni, calibracao, stacking, busca de HP, OOD) e a paisagem de LLMs PT-BR (Sabia 1/2, Bode, Tucano, Albertina, Gervasio, Cabrita, Gloria).

---

## 1. Tabela mestre (ref -> secao -> claim)

| # | bibkey | Onde no paper | Por que cita |
|---|---|---|---|
| 1 | `souza2020bertimbau` | Modelos BERT (Sec. 3.3) | Encoder PT-BR geral; baseline e melhor modelo binario em-dominio (F1 0,8675). |
| 2 | `santos2023finbert` | Modelos BERT (Sec. 3.3) | FinBERT-PT-BR: encoder financeiro PT-BR. Eixo central da Manchete A (pretraining financeiro nao compensa em downstream e |
| 3 | `pires2025deb3rta` | Modelos BERT (Sec. 3.3) | DeB3RTa: 2o encoder financeiro PT-BR (DeBERTa-v2, mixed-domain pretraining). Replicacao da Manchete A; pior dos tres BER |
| 4 | `monteiro2018fakebr` | Avaliacao OOD (Sec. 4.2) | Fake.Br Corpus: corpus OOD politico-geral (n=7.200, prevalencia mercado 0,61%). Manchete D - TF-IDF NB e o mais robusto  |
| 5 | `garcia2022fakerecogna` | Avaliacao OOD (Sec. 4.2) | FakeRecogna: corpus OOD multi-veiculo brasileiro. Recorte 'economia UOL' (n=11.872) e onde FinBERT-PT-BR recupera vantag |
| 6 | `santana2018folha` | Corpus (Sec. 3.1) | Corpus de noticias da Folha de Sao Paulo (n=166.288). Nao tem paper canonico - dataset publico do Kaggle. Citar como rec |
| 7 | `dietterich1998` | Testes estatisticos (Sec. 3.5) | Fundamento metodologico do teste McNemar pareado para comparacao de classificadores supervisionados. Referencia obrigato |
| 8 | `demsar2006` | Testes estatisticos (Sec. 3.5) | Referencia canonica para comparacao estatistica de classificadores entre multiplas tarefas/modelos. Justifica explicitam |
| 9 | `guo2017calibration` | Metricas (Sec. 3.2) | Define Expected Calibration Error (ECE) e mostra que redes neurais modernas sao miscalibradas. Justifica reportar ECE al |
| 10 | `wolpert1992stacked` | Ensembles (Sec. 3.7) | Trabalho original de stacked generalization. Justifica o meta-classificador treinado em out-of-fold predictions (no noss |
| 11 | `bergstra2012random` | Busca de hiperparametros (Sec. 3.4) | Justifica metodologicamente o uso de RandomizedSearchCV em vez de grid search. Referencia obrigatoria quando se reporta  |
| 12 | `araci2019finbert` | Trabalhos relacionados (Sec. 2) | FinBERT original (ingles, Reuters/Financial PhraseBank). Antecedente direto do FinBERT-PT-BR e referencia obrigatoria ao |
| 13 | `yang2020finbert` | Trabalhos relacionados (Sec. 2) | Segundo FinBERT em ingles (Yang et al., 4.9B tokens financeiros). Junto com Araci, sustenta a hipotese 'pretraining fina |
| 14 | `lee2020biobert` | Trabalhos relacionados (Sec. 2) | Exemplar mais citado de domain-specific BERT (biomedico). Junto com SciBERT, ancora a discussao 'quando pretraining de d |
| 15 | `beltagy2019scibert` | Trabalhos relacionados (Sec. 2) | SciBERT: outro classico de domain-specific BERT. Complementa Lee 2020 BioBERT para enquadrar a tese. |
| 16 | `brown2020gpt3` | Trabalhos relacionados + Comparacao LLM (Sec. 2/4) | Origem do paradigma zero/few-shot via prompting em LLMs. Referencia obrigatoria para justificar avaliar Mistral/Qwen sem |
| 17 | `wang2012ngrams` | Modelos TF-IDF (Sec. 3.3) | Mostra empiricamente que NB-SVM com n-grams compete com modelos complexos em classificacao de texto. Sustenta o uso de T |
| 18 | `pires2023sabia` | LLMs PT-BR (Sec. 2 Trabalhos relacionados) | Sabia-7B: 1o LLM grande PT-BR (Maritaca). Continued pretraining de LLaMA em corpus brasileiro. Referencia de abertura qu |
| 19 | `almeida2024sabia2` | LLMs PT-BR (Sec. 2 Trabalhos relacionados) | Sabia-2: 2a geracao Maritaca, atinge GPT-4-level em benchmarks PT-BR. Estado-da-arte fechado/proprietary; referencia obr |
| 20 | `garcia2024bode` | LLMs PT-BR (Sec. 2 Trabalhos relacionados) | Bode: LLM PT-BR aberto (LLaMA-2 fine-tunado em Alpaca-PT). Util como comparavel aberto a Sabia (proprietary). Mesmo auto |
| 21 | `correa2025tucano` | LLMs PT-BR (Sec. 2 Trabalhos relacionados) | Tucano: LLM PT-BR pretreinado from-scratch (nao continued pretraining). Publicado em Patterns (Cell Press), referencia r |
| 22 | `rodrigues2023albertina` | LLMs PT-BR (Sec. 2 Trabalhos relacionados) | Albertina PT-* (PT-PT e PT-BR): DeBERTa-base continued pretraining em corpus portugues. Encoder, nao decoder - serve par |
| 23 | `santos2024gervasio` | LLMs PT-BR (Sec. 2 Trabalhos relacionados) | Gervasio PT-*: decoder aberto PT (Branco/PORTULAN). Complementa Albertina (encoder) com decoder. Util quando se quer cob |
| 24 | `larcher2023cabrita` | LLMs PT-BR (Sec. 2 Trabalhos relacionados) | Cabrita: LLaMA continued pretraining em PT-BR (3B tokens). Util como ponto historico - foi um dos primeiros experimentos |
| 25 | `lopes2024gloria` | LLMs PT-BR (Sec. 2 Trabalhos relacionados) | GlorIA: LLM PT (NOVA School Lisboa). 1.3B parametros, aberto. Foco em PT-PT mas relevante para o panorama luso-brasileir |
| 26 | `jiang2023mistral7b` | Modelos LLM avaliados (Sec. 3.3 / 4) | Mistral-7B-Instruct-v0.3 e um dos dois LLMs avaliados no paper (Manchete C). Tech report obrigatorio para citar. |
| 27 | `qwen2024qwen25` | Modelos LLM avaliados (Sec. 3.3 / 4) | Qwen2.5-7B-Instruct e o segundo LLM avaliado (Manchete C). Tech report do Alibaba. |
| 28 | `grattafiori2024llama3` | LLMs PT-BR / contexto (opcional) | Llama-3 Herd: referencia se o paper futuramente incluir Llama. Pelo stil_excluded_runs.md, Llama-3.1 foi removido do abs |

---

## 2. Entradas detalhadas

### `souza2020bertimbau`

**Referencia.** Souza, F.; Nogueira, R.; Lotufo, R. (2020). BERTimbau: pretrained BERT models for Brazilian Portuguese. BRACIS 2020, LNCS 12320.

**DOI.** `10.1007/978-3-030-61377-8_28` - https://doi.org/10.1007/978-3-030-61377-8_28

**Citado por (OpenAlex).** 587

**Onde no paper.** Modelos BERT (Sec. 3.3)

**Por que cita.** Encoder PT-BR geral; baseline e melhor modelo binario em-dominio (F1 0,8675).

```bibtex
@inproceedings{souza2020bertimbau,
  author = {Souza, F\'abio and Nogueira, Rodrigo and Lotufo, Roberto},
  title = {{BERT}imbau: pretrained {BERT} models for {B}razilian {P}ortuguese},
  booktitle = {Intelligent Systems - 9th Brazilian Conference, BRACIS},
  series = {Lecture Notes in Computer Science},
  volume = {12320},
  year = {2020},
  doi = {10.1007/978-3-030-61377-8_28}
}
```

### `santos2023finbert`

**Referencia.** Santos, L. L.; Bianchi, R. A. C.; Costa, A. H. R. (2023). FinBERT-PT-BR: Analise de Sentimentos de Textos em Portugues do Mercado Financeiro. II BWAIF/SBC, pp. 144-155.

**Onde no paper.** Modelos BERT (Sec. 3.3)

**Por que cita.** FinBERT-PT-BR: encoder financeiro PT-BR. Eixo central da Manchete A (pretraining financeiro nao compensa em downstream editorial).

```bibtex
@inproceedings{santos2023finbert,
  title = {FinBERT-PT-BR: An\'alise de Sentimentos de Textos em Portugu\^es do Mercado Financeiro},
  author = {Santos, Lucas L. and Bianchi, Reinaldo A. C. and Costa, Anna H. R.},
  booktitle = {Anais do II Brazilian Workshop on Artificial Intelligence in Finance},
  pages = {144--155},
  year = {2023},
  organization = {SBC}
}
```

### `pires2025deb3rta`

**Referencia.** Pires, H.; Paucar, L.; Carvalho, J. P. (2025). DeB3RTa: A Transformer-Based Model for the Portuguese Financial Domain. Big Data and Cognitive Computing 9(3):51.

**DOI.** `10.3390/bdcc9030051` - https://doi.org/10.3390/bdcc9030051

**Onde no paper.** Modelos BERT (Sec. 3.3)

**Por que cita.** DeB3RTa: 2o encoder financeiro PT-BR (DeBERTa-v2, mixed-domain pretraining). Replicacao da Manchete A; pior dos tres BERTs em-dominio (F1 0,7734).

```bibtex
@article{pires2025deb3rta,
  author = {Pires, Higo and Paucar, Leonardo and Carvalho, Jo\~ao Paulo},
  title = {{DeB3RTa}: A Transformer-Based Model for the {P}ortuguese Financial Domain},
  journal = {Big Data and Cognitive Computing},
  volume = {9},
  number = {3},
  pages = {51},
  year = {2025},
  doi = {10.3390/bdcc9030051}
}
```

### `monteiro2018fakebr`

**Referencia.** Monteiro, R. A.; Santos, R. L. S.; Pardo, T. A. S.; et al. (2018). Contributions to the Study of Fake News in Portuguese: New Corpus and Automatic Detection Results. PROPOR, LNCS 11122.

**DOI.** `10.1007/978-3-319-99722-3_33` - https://doi.org/10.1007/978-3-319-99722-3_33

**Citado por (OpenAlex).** 133

**Onde no paper.** Avaliacao OOD (Sec. 4.2)

**Por que cita.** Fake.Br Corpus: corpus OOD politico-geral (n=7.200, prevalencia mercado 0,61%). Manchete D - TF-IDF NB e o mais robusto fora de dominio politico.

```bibtex
@inproceedings{monteiro2018fakebr,
  author = {Monteiro, Rafael A. and Santos, Roney L. S. and Pardo, Thiago A. S. and de Almeida, Tiago A. and Ruiz, Evandro E. S. and Vale, Oto A.},
  title = {Contributions to the Study of Fake News in {P}ortuguese: New Corpus and Automatic Detection Results},
  booktitle = {Computational Processing of the Portuguese Language - PROPOR 2018},
  series = {Lecture Notes in Computer Science},
  volume = {11122},
  year = {2018},
  doi = {10.1007/978-3-319-99722-3_33}
}
```

### `garcia2022fakerecogna`

**Referencia.** Garcia, G. L.; Afonso, L. C. S.; Papa, J. P. (2022). FakeRecogna: A New Brazilian Corpus for Fake News Detection. PROPOR, LNCS 13208.

**DOI.** `10.1007/978-3-030-98305-5_6` - https://doi.org/10.1007/978-3-030-98305-5_6

**Citado por (OpenAlex).** 8

**Onde no paper.** Avaliacao OOD (Sec. 4.2)

**Por que cita.** FakeRecogna: corpus OOD multi-veiculo brasileiro. Recorte 'economia UOL' (n=11.872) e onde FinBERT-PT-BR recupera vantagem - evidencia aditiva da Manchete A/D.

```bibtex
@inproceedings{garcia2022fakerecogna,
  author = {Garcia, Gabriel Lino and Afonso, Luis C. S. and Papa, Jo\~ao Paulo},
  title = {{FakeRecogna}: A New {B}razilian Corpus for Fake News Detection},
  booktitle = {Computational Processing of the Portuguese Language - PROPOR 2022},
  series = {Lecture Notes in Computer Science},
  volume = {13208},
  year = {2022},
  doi = {10.1007/978-3-030-98305-5_6}
}
```

### `santana2018folha`

**Referencia.** Santana, M. (2018). News of the Site Folha de Sao Paulo (Brazilian Newspaper). Kaggle dataset.

**Onde no paper.** Corpus (Sec. 3.1)

**Por que cita.** Corpus de noticias da Folha de Sao Paulo (n=166.288). Nao tem paper canonico - dataset publico do Kaggle. Citar como recurso, nao como paper.

```bibtex
@misc{santana2018folha,
  author = {Santana, Marlesson},
  title = {News of the Site {F}olha de {S}\~ao {P}aulo ({B}razilian {N}ewspaper)},
  year = {2018},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/marlesson/news-of-the-site-folhauol}}
}
```

### `dietterich1998`

**Referencia.** Dietterich, T. G. (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. Neural Computation 10(7):1895-1923.

**DOI.** `10.1162/089976698300017197` - https://doi.org/10.1162/089976698300017197

**Citado por (OpenAlex).** 3603

**Onde no paper.** Testes estatisticos (Sec. 3.5)

**Por que cita.** Fundamento metodologico do teste McNemar pareado para comparacao de classificadores supervisionados. Referencia obrigatoria para a Tabela 4.

```bibtex
@article{dietterich1998,
  author = {Dietterich, Thomas G.},
  title = {Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms},
  journal = {Neural Computation},
  volume = {10},
  number = {7},
  pages = {1895--1923},
  year = {1998},
  doi = {10.1162/089976698300017197}
}
```

### `demsar2006`

**Referencia.** Demsar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. Journal of Machine Learning Research 7:1-30.

**Citado por (OpenAlex).** 11214

**Onde no paper.** Testes estatisticos (Sec. 3.5)

**Por que cita.** Referencia canonica para comparacao estatistica de classificadores entre multiplas tarefas/modelos. Justifica explicitamente a correcao de Bonferroni e alternativas (Friedman+Nemenyi).

```bibtex
@article{demsar2006,
  author = {Dem\v{s}ar, Janez},
  title = {Statistical Comparisons of Classifiers over Multiple Data Sets},
  journal = {Journal of Machine Learning Research},
  volume = {7},
  pages = {1--30},
  year = {2006}
}
```

### `guo2017calibration`

**Referencia.** Guo, C.; Pleiss, G.; Sun, Y.; Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML.

**DOI.** `10.48550/arxiv.1706.04599` - https://doi.org/10.48550/arxiv.1706.04599

**Citado por (OpenAlex).** 1727

**Onde no paper.** Metricas (Sec. 3.2)

**Por que cita.** Define Expected Calibration Error (ECE) e mostra que redes neurais modernas sao miscalibradas. Justifica reportar ECE alem de Brier no paper.

```bibtex
@inproceedings{guo2017calibration,
  author = {Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  title = {On Calibration of Modern Neural Networks},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning (ICML)},
  year = {2017},
  eprint = {1706.04599},
  archivePrefix = {arXiv}
}
```

### `wolpert1992stacked`

**Referencia.** Wolpert, D. H. (1992). Stacked generalization. Neural Networks 5(2):241-259.

**DOI.** `10.1016/s0893-6080(05)80023-1` - https://doi.org/10.1016/s0893-6080(05)80023-1

**Citado por (OpenAlex).** 7352

**Onde no paper.** Ensembles (Sec. 3.7)

**Por que cita.** Trabalho original de stacked generalization. Justifica o meta-classificador treinado em out-of-fold predictions (no nosso caso, validacao).

```bibtex
@article{wolpert1992stacked,
  author = {Wolpert, David H.},
  title = {Stacked generalization},
  journal = {Neural Networks},
  volume = {5},
  number = {2},
  pages = {241--259},
  year = {1992},
  doi = {10.1016/S0893-6080(05)80023-1}
}
```

### `bergstra2012random`

**Referencia.** Bergstra, J.; Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research 13:281-305.

**Citado por (OpenAlex).** 7928

**Onde no paper.** Busca de hiperparametros (Sec. 3.4)

**Por que cita.** Justifica metodologicamente o uso de RandomizedSearchCV em vez de grid search. Referencia obrigatoria quando se reporta n_trials=60.

```bibtex
@article{bergstra2012random,
  author = {Bergstra, James and Bengio, Yoshua},
  title = {Random Search for Hyper-Parameter Optimization},
  journal = {Journal of Machine Learning Research},
  volume = {13},
  pages = {281--305},
  year = {2012}
}
```

### `araci2019finbert`

**Referencia.** Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063.

**DOI.** `10.48550/arxiv.1908.10063` - https://doi.org/10.48550/arxiv.1908.10063

**Citado por (OpenAlex).** 160

**Onde no paper.** Trabalhos relacionados (Sec. 2)

**Por que cita.** FinBERT original (ingles, Reuters/Financial PhraseBank). Antecedente direto do FinBERT-PT-BR e referencia obrigatoria ao discutir pretraining financeiro.

```bibtex
@article{araci2019finbert,
  author = {Araci, Dogu},
  title = {{FinBERT}: Financial Sentiment Analysis with Pre-trained Language Models},
  journal = {arXiv preprint},
  eprint = {1908.10063},
  archivePrefix = {arXiv},
  year = {2019}
}
```

### `yang2020finbert`

**Referencia.** Yang, Y.; Uy, M. C. S.; Huang, A. (2020). FinBERT: A Pretrained Language Model for Financial Communications. arXiv:2006.08097.

**DOI.** `10.48550/arxiv.2006.08097` - https://doi.org/10.48550/arxiv.2006.08097

**Citado por (OpenAlex).** 176

**Onde no paper.** Trabalhos relacionados (Sec. 2)

**Por que cita.** Segundo FinBERT em ingles (Yang et al., 4.9B tokens financeiros). Junto com Araci, sustenta a hipotese 'pretraining financeiro ajuda em sentimento financeiro' que o paper PT-BR refuta para classificacao editorial.

```bibtex
@article{yang2020finbert,
  author = {Yang, Yi and Uy, Mark Christopher Siy and Huang, Allen},
  title = {{FinBERT}: A Pretrained Language Model for Financial Communications},
  journal = {arXiv preprint},
  eprint = {2006.08097},
  archivePrefix = {arXiv},
  year = {2020}
}
```

### `lee2020biobert`

**Referencia.** Lee, J.; Yoon, W.; Kim, S.; et al. (2020). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics 36(4):1234-1240.

**DOI.** `10.1093/bioinformatics/btz682` - https://doi.org/10.1093/bioinformatics/btz682

**Citado por (OpenAlex).** 6880

**Onde no paper.** Trabalhos relacionados (Sec. 2)

**Por que cita.** Exemplar mais citado de domain-specific BERT (biomedico). Junto com SciBERT, ancora a discussao 'quando pretraining de dominio compensa'.

```bibtex
@article{lee2020biobert,
  author = {Lee, Jinhyuk and Yoon, Wonjin and Kim, Sungdong and Kim, Donghyeon and Kim, Sunkyu and So, Chan Ho and Kang, Jaewoo},
  title = {{BioBERT}: a pre-trained biomedical language representation model for biomedical text mining},
  journal = {Bioinformatics},
  volume = {36},
  number = {4},
  pages = {1234--1240},
  year = {2020},
  doi = {10.1093/bioinformatics/btz682}
}
```

### `beltagy2019scibert`

**Referencia.** Beltagy, I.; Lo, K.; Cohan, A. (2019). SciBERT: A Pretrained Language Model for Scientific Text. EMNLP-IJCNLP.

**DOI.** `10.18653/v1/d19-1371` - https://doi.org/10.18653/v1/d19-1371

**Citado por (OpenAlex).** 2977

**Onde no paper.** Trabalhos relacionados (Sec. 2)

**Por que cita.** SciBERT: outro classico de domain-specific BERT. Complementa Lee 2020 BioBERT para enquadrar a tese.

```bibtex
@inproceedings{beltagy2019scibert,
  author = {Beltagy, Iz and Lo, Kyle and Cohan, Arman},
  title = {{SciBERT}: A Pretrained Language Model for Scientific Text},
  booktitle = {Proceedings of EMNLP-IJCNLP 2019},
  year = {2019},
  doi = {10.18653/v1/D19-1371}
}
```

### `brown2020gpt3`

**Referencia.** Brown, T. B.; Mann, B.; Ryder, N.; et al. (2020). Language Models are Few-Shot Learners. NeurIPS.

**DOI.** `10.48550/arxiv.2005.14165` - https://doi.org/10.48550/arxiv.2005.14165

**Citado por (OpenAlex).** 3029

**Onde no paper.** Trabalhos relacionados + Comparacao LLM (Sec. 2/4)

**Por que cita.** Origem do paradigma zero/few-shot via prompting em LLMs. Referencia obrigatoria para justificar avaliar Mistral/Qwen sem fine-tuning (Manchete C).

```bibtex
@inproceedings{brown2020gpt3,
  author = {Brown, Tom B. and Mann, Benjamin and Ryder, Nick and others},
  title = {Language Models are Few-Shot Learners},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2020},
  eprint = {2005.14165},
  archivePrefix = {arXiv}
}
```

### `wang2012ngrams`

**Referencia.** Wang, S.; Manning, C. D. (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. ACL.

**Citado por (OpenAlex).** 967

**Onde no paper.** Modelos TF-IDF (Sec. 3.3)

**Por que cita.** Mostra empiricamente que NB-SVM com n-grams compete com modelos complexos em classificacao de texto. Sustenta o uso de TF-IDF + LinearSVC/LogReg/NB como baselines fortes (nao stragglers).

```bibtex
@inproceedings{wang2012ngrams,
  author = {Wang, Sida and Manning, Christopher D.},
  title = {Baselines and Bigrams: Simple, Good Sentiment and Topic Classification},
  booktitle = {Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year = {2012}
}
```

### `pires2023sabia`

**Referencia.** Pires, R.; Abonizio, H.; Almeida, T. S.; Nogueira, R. (2023). Sabia: Portuguese Large Language Models. BRACIS 2023, LNCS 14197.

**DOI.** `10.1007/978-3-031-45392-2_15` - https://doi.org/10.1007/978-3-031-45392-2_15

**Citado por (OpenAlex).** 63

**Onde no paper.** LLMs PT-BR (Sec. 2 Trabalhos relacionados)

**Por que cita.** Sabia-7B: 1o LLM grande PT-BR (Maritaca). Continued pretraining de LLaMA em corpus brasileiro. Referencia de abertura quando se discute paisagem de LLM PT-BR.

```bibtex
@inproceedings{pires2023sabia,
  author = {Pires, Ramon and Abonizio, Hugo and Almeida, Thales Sales and Nogueira, Rodrigo},
  title = {{Sabi\'a}: {P}ortuguese Large Language Models},
  booktitle = {Intelligent Systems - 12th Brazilian Conference, BRACIS},
  series = {Lecture Notes in Computer Science},
  volume = {14197},
  year = {2023},
  doi = {10.1007/978-3-031-45392-2_15}
}
```

### `almeida2024sabia2`

**Referencia.** Almeida, T. S.; Abonizio, H.; Nogueira, R.; Pires, R. (2024). Sabia-2: A New Generation of Portuguese Large Language Models. arXiv:2403.09887.

**DOI.** `10.48550/arxiv.2403.09887` - https://doi.org/10.48550/arxiv.2403.09887

**Citado por (OpenAlex).** 4

**Onde no paper.** LLMs PT-BR (Sec. 2 Trabalhos relacionados)

**Por que cita.** Sabia-2: 2a geracao Maritaca, atinge GPT-4-level em benchmarks PT-BR. Estado-da-arte fechado/proprietary; referencia obrigatoria quando se compara LLM aberto vs SOTA.

```bibtex
@article{almeida2024sabia2,
  author = {Almeida, Thales Sales and Abonizio, Hugo and Nogueira, Rodrigo and Pires, Ramon},
  title = {{Sabi\'a-2}: A New Generation of Portuguese Large Language Models},
  journal = {arXiv preprint},
  eprint = {2403.09887},
  archivePrefix = {arXiv},
  year = {2024}
}
```

### `garcia2024bode`

**Referencia.** Garcia, G. L.; Paiola, P. H.; Morelli, L. H.; et al. (2024). Introducing Bode: A Fine-Tuned Large Language Model for Portuguese Prompt-Based Task. arXiv:2401.02909.

**DOI.** `10.48550/arxiv.2401.02909` - https://doi.org/10.48550/arxiv.2401.02909

**Citado por (OpenAlex).** 7

**Onde no paper.** LLMs PT-BR (Sec. 2 Trabalhos relacionados)

**Por que cita.** Bode: LLM PT-BR aberto (LLaMA-2 fine-tunado em Alpaca-PT). Util como comparavel aberto a Sabia (proprietary). Mesmo autor de FakeRecogna.

```bibtex
@article{garcia2024bode,
  author = {Garcia, Gabriel Lino and Paiola, Pedro Henrique and Morelli, Luis Henrique and Candido, Giovani and Candido J\'unior, Arnaldo and Jodas, Danilo Samuel and Afonso, Luis C. S. and Guilherme, Ivan Rizzo and Penteado, Bruno Elias and Papa, Jo\~ao Paulo},
  title = {Introducing Bode: A Fine-Tuned Large Language Model for {P}ortuguese Prompt-Based Task},
  journal = {arXiv preprint},
  eprint = {2401.02909},
  archivePrefix = {arXiv},
  year = {2024}
}
```

### `correa2025tucano`

**Referencia.** Correa, N. K.; Sen, A.; Falk, S.; et al. (2025). Tucano: Advancing neural text generation for Portuguese. Patterns 6(6):101325.

**DOI.** `10.1016/j.patter.2025.101325` - https://doi.org/10.1016/j.patter.2025.101325

**Citado por (OpenAlex).** 11

**Onde no paper.** LLMs PT-BR (Sec. 2 Trabalhos relacionados)

**Por que cita.** Tucano: LLM PT-BR pretreinado from-scratch (nao continued pretraining). Publicado em Patterns (Cell Press), referencia recente e auditavel.

```bibtex
@article{correa2025tucano,
  author = {Corr\^ea, Nicholas Kluge and Sen, Aniket and Falk, Sophia and Fatimah, Shiza},
  title = {Tucano: Advancing Neural Text Generation for {P}ortuguese},
  journal = {Patterns},
  volume = {6},
  number = {6},
  pages = {101325},
  year = {2025},
  doi = {10.1016/j.patter.2025.101325}
}
```

### `rodrigues2023albertina`

**Referencia.** Rodrigues, J.; Gomes, L.; Silva, J.; et al. (2023). Advancing Neural Encoding of Portuguese with Transformer Albertina PT-*. BRACIS 2023, LNCS 14195.

**DOI.** `10.1007/978-3-031-49008-8_35` - https://doi.org/10.1007/978-3-031-49008-8_35

**Citado por (OpenAlex).** 35

**Onde no paper.** LLMs PT-BR (Sec. 2 Trabalhos relacionados)

**Por que cita.** Albertina PT-* (PT-PT e PT-BR): DeBERTa-base continued pretraining em corpus portugues. Encoder, nao decoder - serve para enquadrar BERTimbau como referencia PT-BR concorrente.

```bibtex
@inproceedings{rodrigues2023albertina,
  author = {Rodrigues, Jo\~ao and Gomes, Lu\'is and Silva, Jo\~ao and Branco, Ant\'onio and Santos, Rodrigo and Cardoso, Henrique Lopes and Os\'orio, Tom\'as},
  title = {Advancing Neural Encoding of {P}ortuguese with Transformer {A}lbertina {PT-*}},
  booktitle = {Intelligent Systems - 12th Brazilian Conference, BRACIS},
  series = {Lecture Notes in Computer Science},
  volume = {14195},
  year = {2023},
  doi = {10.1007/978-3-031-49008-8_35}
}
```

### `santos2024gervasio`

**Referencia.** Santos, R.; Silva, J.; Gomes, L.; Rodrigues, J.; Branco, A. (2024). Advancing Generative AI for Portuguese with Open Decoder Gervasio PT*. arXiv:2402.18766.

**DOI.** `10.48550/arxiv.2402.18766` - https://doi.org/10.48550/arxiv.2402.18766

**Citado por (OpenAlex).** 2

**Onde no paper.** LLMs PT-BR (Sec. 2 Trabalhos relacionados)

**Por que cita.** Gervasio PT-*: decoder aberto PT (Branco/PORTULAN). Complementa Albertina (encoder) com decoder. Util quando se quer cobertura PT-PT alem de PT-BR.

```bibtex
@article{santos2024gervasio,
  author = {Santos, Rodrigo and Silva, Jo\~ao and Gomes, Lu\'is and Rodrigues, Jo\~ao and Branco, Ant\'onio},
  title = {Advancing Generative {AI} for {P}ortuguese with Open Decoder {G}erv\'asio {PT*}},
  journal = {arXiv preprint},
  eprint = {2402.18766},
  archivePrefix = {arXiv},
  year = {2024}
}
```

### `larcher2023cabrita`

**Referencia.** Larcher, C. H. N.; Piau, M.; Finardi, P.; et al. (2023). Cabrita: closing the gap for foreign languages. arXiv:2308.11878.

**DOI.** `10.48550/arxiv.2308.11878` - https://doi.org/10.48550/arxiv.2308.11878

**Citado por (OpenAlex).** 4

**Onde no paper.** LLMs PT-BR (Sec. 2 Trabalhos relacionados)

**Por que cita.** Cabrita: LLaMA continued pretraining em PT-BR (3B tokens). Util como ponto historico - foi um dos primeiros experimentos de adaptacao LLaMA para PT.

```bibtex
@article{larcher2023cabrita,
  author = {Larcher, Celio H. N. and Piau, Marcos and Finardi, Paulo and Gengo, Pedro and Esposito, Piero and Caridade, Vinicius},
  title = {Cabrita: Closing the Gap for Foreign Languages},
  journal = {arXiv preprint},
  eprint = {2308.11878},
  archivePrefix = {arXiv},
  year = {2023}
}
```

### `lopes2024gloria`

**Referencia.** Lopes, R.; Magalhaes, J.; Semedo, D. (2024). GlorIA - A Generative and Open Large Language Model for Portuguese. arXiv:2402.12969.

**DOI.** `10.48550/arxiv.2402.12969` - https://doi.org/10.48550/arxiv.2402.12969

**Citado por (OpenAlex).** 1

**Onde no paper.** LLMs PT-BR (Sec. 2 Trabalhos relacionados)

**Por que cita.** GlorIA: LLM PT (NOVA School Lisboa). 1.3B parametros, aberto. Foco em PT-PT mas relevante para o panorama luso-brasileiro.

```bibtex
@article{lopes2024gloria,
  author = {Lopes, Ricardo and Magalh\~aes, Jo\~ao and Semedo, David},
  title = {{Gl\'orIA} -- A Generative and Open Large Language Model for {P}ortuguese},
  journal = {arXiv preprint},
  eprint = {2402.12969},
  archivePrefix = {arXiv},
  year = {2024}
}
```

### `jiang2023mistral7b`

**Referencia.** Jiang, A. Q.; Sablayrolles, A.; Mensch, A.; et al. (2023). Mistral 7B. arXiv:2310.06825.

**DOI.** `10.48550/arxiv.2310.06825` - https://doi.org/10.48550/arxiv.2310.06825

**Citado por (OpenAlex).** 279

**Onde no paper.** Modelos LLM avaliados (Sec. 3.3 / 4)

**Por que cita.** Mistral-7B-Instruct-v0.3 e um dos dois LLMs avaliados no paper (Manchete C). Tech report obrigatorio para citar.

```bibtex
@article{jiang2023mistral7b,
  author = {Jiang, Albert Q. and Sablayrolles, Alexandre and Mensch, Arthur and Bamford, Chris and Chaplot, Devendra Singh and de las Casas, Diego and Bressand, Florian and Lengyel, Gianna and Lample, Guillaume and Saulnier, Lucile and others},
  title = {{Mistral 7B}},
  journal = {arXiv preprint},
  eprint = {2310.06825},
  archivePrefix = {arXiv},
  year = {2023}
}
```

### `qwen2024qwen25`

**Referencia.** Qwen Team; Yang, A.; et al. (2024). Qwen2.5 Technical Report. arXiv:2412.15115.

**DOI.** `10.48550/arxiv.2412.15115` - https://doi.org/10.48550/arxiv.2412.15115

**Citado por (OpenAlex).** 70

**Onde no paper.** Modelos LLM avaliados (Sec. 3.3 / 4)

**Por que cita.** Qwen2.5-7B-Instruct e o segundo LLM avaliado (Manchete C). Tech report do Alibaba.

```bibtex
@article{qwen2024qwen25,
  author = {{Qwen Team}},
  title = {{Qwen2.5} Technical Report},
  journal = {arXiv preprint},
  eprint = {2412.15115},
  archivePrefix = {arXiv},
  year = {2024}
}
```

### `grattafiori2024llama3`

**Referencia.** Grattafiori, A.; Dubey, A.; Jauhri, A.; et al. (2024). The Llama 3 Herd of Models. arXiv:2407.21783.

**DOI.** `10.48550/arxiv.2407.21783` - https://doi.org/10.48550/arxiv.2407.21783

**Citado por (OpenAlex).** 0

**Onde no paper.** LLMs PT-BR / contexto (opcional)

**Por que cita.** Llama-3 Herd: referencia se o paper futuramente incluir Llama. Pelo stil_excluded_runs.md, Llama-3.1 foi removido do abstract, mas a ref segue util para a dissertacao.

```bibtex
@article{grattafiori2024llama3,
  author = {Grattafiori, Aaron and Dubey, Abhimanyu and Jauhri, Abhinav and others},
  title = {The {Llama 3} Herd of Models},
  journal = {arXiv preprint},
  eprint = {2407.21783},
  archivePrefix = {arXiv},
  year = {2024}
}
```

---

## 3. Bloco BibTeX consolidado (copiar para references.bib)

```bibtex
@inproceedings{souza2020bertimbau,
  author = {Souza, F\'abio and Nogueira, Rodrigo and Lotufo, Roberto},
  title = {{BERT}imbau: pretrained {BERT} models for {B}razilian {P}ortuguese},
  booktitle = {Intelligent Systems - 9th Brazilian Conference, BRACIS},
  series = {Lecture Notes in Computer Science},
  volume = {12320},
  year = {2020},
  doi = {10.1007/978-3-030-61377-8_28}
}

@inproceedings{santos2023finbert,
  title = {FinBERT-PT-BR: An\'alise de Sentimentos de Textos em Portugu\^es do Mercado Financeiro},
  author = {Santos, Lucas L. and Bianchi, Reinaldo A. C. and Costa, Anna H. R.},
  booktitle = {Anais do II Brazilian Workshop on Artificial Intelligence in Finance},
  pages = {144--155},
  year = {2023},
  organization = {SBC}
}

@article{pires2025deb3rta,
  author = {Pires, Higo and Paucar, Leonardo and Carvalho, Jo\~ao Paulo},
  title = {{DeB3RTa}: A Transformer-Based Model for the {P}ortuguese Financial Domain},
  journal = {Big Data and Cognitive Computing},
  volume = {9},
  number = {3},
  pages = {51},
  year = {2025},
  doi = {10.3390/bdcc9030051}
}

@inproceedings{monteiro2018fakebr,
  author = {Monteiro, Rafael A. and Santos, Roney L. S. and Pardo, Thiago A. S. and de Almeida, Tiago A. and Ruiz, Evandro E. S. and Vale, Oto A.},
  title = {Contributions to the Study of Fake News in {P}ortuguese: New Corpus and Automatic Detection Results},
  booktitle = {Computational Processing of the Portuguese Language - PROPOR 2018},
  series = {Lecture Notes in Computer Science},
  volume = {11122},
  year = {2018},
  doi = {10.1007/978-3-319-99722-3_33}
}

@inproceedings{garcia2022fakerecogna,
  author = {Garcia, Gabriel Lino and Afonso, Luis C. S. and Papa, Jo\~ao Paulo},
  title = {{FakeRecogna}: A New {B}razilian Corpus for Fake News Detection},
  booktitle = {Computational Processing of the Portuguese Language - PROPOR 2022},
  series = {Lecture Notes in Computer Science},
  volume = {13208},
  year = {2022},
  doi = {10.1007/978-3-030-98305-5_6}
}

@misc{santana2018folha,
  author = {Santana, Marlesson},
  title = {News of the Site {F}olha de {S}\~ao {P}aulo ({B}razilian {N}ewspaper)},
  year = {2018},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/marlesson/news-of-the-site-folhauol}}
}

@article{dietterich1998,
  author = {Dietterich, Thomas G.},
  title = {Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms},
  journal = {Neural Computation},
  volume = {10},
  number = {7},
  pages = {1895--1923},
  year = {1998},
  doi = {10.1162/089976698300017197}
}

@article{demsar2006,
  author = {Dem\v{s}ar, Janez},
  title = {Statistical Comparisons of Classifiers over Multiple Data Sets},
  journal = {Journal of Machine Learning Research},
  volume = {7},
  pages = {1--30},
  year = {2006}
}

@inproceedings{guo2017calibration,
  author = {Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  title = {On Calibration of Modern Neural Networks},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning (ICML)},
  year = {2017},
  eprint = {1706.04599},
  archivePrefix = {arXiv}
}

@article{wolpert1992stacked,
  author = {Wolpert, David H.},
  title = {Stacked generalization},
  journal = {Neural Networks},
  volume = {5},
  number = {2},
  pages = {241--259},
  year = {1992},
  doi = {10.1016/S0893-6080(05)80023-1}
}

@article{bergstra2012random,
  author = {Bergstra, James and Bengio, Yoshua},
  title = {Random Search for Hyper-Parameter Optimization},
  journal = {Journal of Machine Learning Research},
  volume = {13},
  pages = {281--305},
  year = {2012}
}

@article{araci2019finbert,
  author = {Araci, Dogu},
  title = {{FinBERT}: Financial Sentiment Analysis with Pre-trained Language Models},
  journal = {arXiv preprint},
  eprint = {1908.10063},
  archivePrefix = {arXiv},
  year = {2019}
}

@article{yang2020finbert,
  author = {Yang, Yi and Uy, Mark Christopher Siy and Huang, Allen},
  title = {{FinBERT}: A Pretrained Language Model for Financial Communications},
  journal = {arXiv preprint},
  eprint = {2006.08097},
  archivePrefix = {arXiv},
  year = {2020}
}

@article{lee2020biobert,
  author = {Lee, Jinhyuk and Yoon, Wonjin and Kim, Sungdong and Kim, Donghyeon and Kim, Sunkyu and So, Chan Ho and Kang, Jaewoo},
  title = {{BioBERT}: a pre-trained biomedical language representation model for biomedical text mining},
  journal = {Bioinformatics},
  volume = {36},
  number = {4},
  pages = {1234--1240},
  year = {2020},
  doi = {10.1093/bioinformatics/btz682}
}

@inproceedings{beltagy2019scibert,
  author = {Beltagy, Iz and Lo, Kyle and Cohan, Arman},
  title = {{SciBERT}: A Pretrained Language Model for Scientific Text},
  booktitle = {Proceedings of EMNLP-IJCNLP 2019},
  year = {2019},
  doi = {10.18653/v1/D19-1371}
}

@inproceedings{brown2020gpt3,
  author = {Brown, Tom B. and Mann, Benjamin and Ryder, Nick and others},
  title = {Language Models are Few-Shot Learners},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2020},
  eprint = {2005.14165},
  archivePrefix = {arXiv}
}

@inproceedings{wang2012ngrams,
  author = {Wang, Sida and Manning, Christopher D.},
  title = {Baselines and Bigrams: Simple, Good Sentiment and Topic Classification},
  booktitle = {Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year = {2012}
}

@inproceedings{pires2023sabia,
  author = {Pires, Ramon and Abonizio, Hugo and Almeida, Thales Sales and Nogueira, Rodrigo},
  title = {{Sabi\'a}: {P}ortuguese Large Language Models},
  booktitle = {Intelligent Systems - 12th Brazilian Conference, BRACIS},
  series = {Lecture Notes in Computer Science},
  volume = {14197},
  year = {2023},
  doi = {10.1007/978-3-031-45392-2_15}
}

@article{almeida2024sabia2,
  author = {Almeida, Thales Sales and Abonizio, Hugo and Nogueira, Rodrigo and Pires, Ramon},
  title = {{Sabi\'a-2}: A New Generation of Portuguese Large Language Models},
  journal = {arXiv preprint},
  eprint = {2403.09887},
  archivePrefix = {arXiv},
  year = {2024}
}

@article{garcia2024bode,
  author = {Garcia, Gabriel Lino and Paiola, Pedro Henrique and Morelli, Luis Henrique and Candido, Giovani and Candido J\'unior, Arnaldo and Jodas, Danilo Samuel and Afonso, Luis C. S. and Guilherme, Ivan Rizzo and Penteado, Bruno Elias and Papa, Jo\~ao Paulo},
  title = {Introducing Bode: A Fine-Tuned Large Language Model for {P}ortuguese Prompt-Based Task},
  journal = {arXiv preprint},
  eprint = {2401.02909},
  archivePrefix = {arXiv},
  year = {2024}
}

@article{correa2025tucano,
  author = {Corr\^ea, Nicholas Kluge and Sen, Aniket and Falk, Sophia and Fatimah, Shiza},
  title = {Tucano: Advancing Neural Text Generation for {P}ortuguese},
  journal = {Patterns},
  volume = {6},
  number = {6},
  pages = {101325},
  year = {2025},
  doi = {10.1016/j.patter.2025.101325}
}

@inproceedings{rodrigues2023albertina,
  author = {Rodrigues, Jo\~ao and Gomes, Lu\'is and Silva, Jo\~ao and Branco, Ant\'onio and Santos, Rodrigo and Cardoso, Henrique Lopes and Os\'orio, Tom\'as},
  title = {Advancing Neural Encoding of {P}ortuguese with Transformer {A}lbertina {PT-*}},
  booktitle = {Intelligent Systems - 12th Brazilian Conference, BRACIS},
  series = {Lecture Notes in Computer Science},
  volume = {14195},
  year = {2023},
  doi = {10.1007/978-3-031-49008-8_35}
}

@article{santos2024gervasio,
  author = {Santos, Rodrigo and Silva, Jo\~ao and Gomes, Lu\'is and Rodrigues, Jo\~ao and Branco, Ant\'onio},
  title = {Advancing Generative {AI} for {P}ortuguese with Open Decoder {G}erv\'asio {PT*}},
  journal = {arXiv preprint},
  eprint = {2402.18766},
  archivePrefix = {arXiv},
  year = {2024}
}

@article{larcher2023cabrita,
  author = {Larcher, Celio H. N. and Piau, Marcos and Finardi, Paulo and Gengo, Pedro and Esposito, Piero and Caridade, Vinicius},
  title = {Cabrita: Closing the Gap for Foreign Languages},
  journal = {arXiv preprint},
  eprint = {2308.11878},
  archivePrefix = {arXiv},
  year = {2023}
}

@article{lopes2024gloria,
  author = {Lopes, Ricardo and Magalh\~aes, Jo\~ao and Semedo, David},
  title = {{Gl\'orIA} -- A Generative and Open Large Language Model for {P}ortuguese},
  journal = {arXiv preprint},
  eprint = {2402.12969},
  archivePrefix = {arXiv},
  year = {2024}
}

@article{jiang2023mistral7b,
  author = {Jiang, Albert Q. and Sablayrolles, Alexandre and Mensch, Arthur and Bamford, Chris and Chaplot, Devendra Singh and de las Casas, Diego and Bressand, Florian and Lengyel, Gianna and Lample, Guillaume and Saulnier, Lucile and others},
  title = {{Mistral 7B}},
  journal = {arXiv preprint},
  eprint = {2310.06825},
  archivePrefix = {arXiv},
  year = {2023}
}

@article{qwen2024qwen25,
  author = {{Qwen Team}},
  title = {{Qwen2.5} Technical Report},
  journal = {arXiv preprint},
  eprint = {2412.15115},
  archivePrefix = {arXiv},
  year = {2024}
}

@article{grattafiori2024llama3,
  author = {Grattafiori, Aaron and Dubey, Abhimanyu and Jauhri, Abhinav and others},
  title = {The {Llama 3} Herd of Models},
  journal = {arXiv preprint},
  eprint = {2407.21783},
  archivePrefix = {arXiv},
  year = {2024}
}

```

---

## 4. Notas de uso


**Mapeamento por manchete:**
- **Manchete A** (pretraining financeiro nao compensa em downstream editorial): `souza2020bertimbau`, `santos2023finbert`, `pires2025deb3rta`, `araci2019finbert`, `yang2020finbert`, `lee2020biobert`, `beltagy2019scibert`.
- **Manchete C** (LLM zero/few-shot perde para BERT fine-tunado): `brown2020gpt3` + os 3 modelos PT-BR.
- **Manchete D** (TF-IDF NB e mais robusto OOD em corpus politico): `monteiro2018fakebr`, `garcia2022fakerecogna`, `wang2012ngrams`.

**Metodologia:**
- McNemar pareado: `dietterich1998`
- Bonferroni / multiplas comparacoes: `demsar2006` (este e o canonico para ML, mais forte que citar Bonferroni original)
- Calibracao (ECE): `guo2017calibration`
- Stacking: `wolpert1992stacked`
- Random search: `bergstra2012random`

**Paisagem de LLMs PT-BR (Sec. 2 Trabalhos Relacionados):**
Recomendo um paragrafo unico cobrindo a paisagem, dividido entre:
- **Encoders PT-BR**: `souza2020bertimbau` (BERTimbau) e `rodrigues2023albertina` (Albertina) - os dois principais. BERTimbau e o avaliado; Albertina entra como "competidor PT-BR contemporaneo".
- **Decoders/LLMs PT-BR**: ordem cronologica em uma sentenca - `larcher2023cabrita` (2023, LLaMA continued), `pires2023sabia` (2023, Maritaca), `garcia2024bode` (2024, LLaMA-2 + Alpaca-PT), `almeida2024sabia2` (2024, Maritaca SOTA proprietary), `santos2024gervasio` + `rodrigues2023albertina` (PT-PT NOVA/PORTULAN), `lopes2024gloria` (PT-PT NOVA), `correa2025tucano` (2025, from-scratch, Patterns/Cell).
- **Modelos avaliados (estrangeiros, multilingual)**: `jiang2023mistral7b` e `qwen2024qwen25` - citar no momento de apresentar a Tabela 1.

**Justificativa metodologica para o paper STIL nao usar LLM PT-BR especifico:**
O paper avalia Mistral-7B-Instruct-v0.3 e Qwen2.5-7B-Instruct, NAO Sabia/Bode/Tucano. Vale uma frase honesta na sec. 4 explicando: "Restringimos a avaliacao LLM aos modelos abertos multilinguais mais usados na pratica (Mistral-7B, Qwen2.5-7B); a comparacao com LLMs PT-BR especializados (Sabia-2, Bode, Tucano) fica como trabalho futuro pois exigiria infraestrutura de inferencia separada para modelos proprietarios (Sabia-2) e adapter/LoRA-only models." Caso contrario o reviewer perguntara "por que nao Sabia?".

**O que ficou de fora (e por que):**
- *Maritalk*: existe como produto (Maritaca), nao como paper academico autonomo - citar via Sabia/Sabia-2 que sao os papers que sustentam o produto.
- *Llama-3.1 (especifico)*: pelo `stil_excluded_runs.md` foi removido do abstract (sem cards executados); incluido aqui como `grattafiori2024llama3` apenas para a dissertacao/trabalho futuro.
- *brWaC / Carolina*: corpora de pre-treinamento do BERTimbau/DeBERTa, citados indiretamente via `souza2020bertimbau`. Nao precisam de entrada propria a menos que o paper detalhe o corpus de pretraining.
- *Survey de fake news PT-BR*: nao foi buscado especificamente. Se quiser cobertura mais profunda de SOTA em fake news PT-BR alem de Fake.Br + FakeRecogna, vale uma busca extra.

**Lacunas a considerar:**
1. **Calibracao binaria especifica** - o paper mede Brier; o `guo2017calibration` cobre ECE. Para Brier o classico e Brier 1950 (Monthly Weather Review) - vale incluir se a secao 3.2 detalhar Brier.
2. **Bootstrap / intervalos de confianca** - se a Tabela 1/2 reporta IC bootstrap, falta uma ref (Efron & Tibshirani 1993). Verifique no notebook 41.
3. **Sklearn / Transformers / Trainer** - software citations. Geralmente vao em footnote, nao na biblio. Pedrosa et al. 2011 (scikit-learn) e Wolf et al. 2020 (HF Transformers) sao os classicos.

