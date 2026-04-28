"""Shared fixtures for all test modules."""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic corpus — 200 rows (25 mercado, 175 outros)
# Texts are realistic enough for downstream classifier evaluation.
# ---------------------------------------------------------------------------

_MERCADO_TEXTS = [
    "Banco Central elevou a taxa Selic para conter a inflacao que atinge o pais",
    "O dolar comercial fechou em alta e o mercado de cambio operou volatil",
    "Investidores reagem a decisao do Copom sobre os juros basicos da economia",
    "A bolsa de valores registrou queda de 2 porcento no pregao desta sexta",
    "O IPCA acumulado no ano supera a meta de inflacao definida pelo governo",
    "Tesouro Nacional anuncia leilao de titulos publicos com alta demanda",
    "O PIB brasileiro cresceu 0,8 porcento no terceiro trimestre do ano",
    "Bancos privados elevam taxas de juros do credito ao consumidor",
    "O mercado financeiro reagiu negativamente ao deficit fiscal do governo",
    "A balanca comercial registrou superavit recorde no semestre",
    "Fundos de investimento tiveram rentabilidade acima do CDI no mes",
    "O real se desvalorizou frente ao dolar com a crise de confianca",
    "Anuncio de privatizacao da estatal agitou o mercado de acoes",
    "A divida publica atingiu patamar recorde em relacao ao PIB",
    "Taxa de desemprego recuou mas informalidade segue alta na economia",
    "Exportacoes de commodities impulsionaram o saldo da balanca comercial",
    "Reforma tributaria promete simplificar impostos sobre consumo no pais",
    "O Banco Central vendeu reservas internacionais para conter o dolar",
    "Previdencia social apresenta rombo bilionario nas contas publicas",
    "Inflacao de alimentos pressiona o IPCA e preocupa o mercado",
    "Criptomoedas como bitcoin registram valorizacao expressiva no trimestre",
    "A Selic deve cair na proxima reuniao do Copom segundo analistas",
    "Credito imobiliario cresce com queda dos juros e aquece o setor",
    "Orcamento federal preve cortes em investimentos para cumprir o teto",
    "Mercado projeta crescimento do PIB acima do esperado para o proximo ano",
]

_OUTROS_TEXTS = [
    "O time de futebol venceu o campeonato regional apos uma temporada invicta",
    "Novo filme brasileiro concorre a premio em festival internacional de cinema",
    "Pesquisadores descobriram nova especie de planta na Amazonia ocidental",
    "Hospital publico inaugura ala de atendimento pediatrico na periferia",
    "Prefeitura anuncia obras de pavimentacao em bairros da zona norte",
    "Festival de musica reune artistas nacionais no parque da cidade",
    "Estudantes protestam contra cortes no orcamento da educacao publica",
    "Policia investiga quadrilha suspeita de roubo de carga em rodovias",
    "Temporal causa alagamentos e interdita vias em diversas regioes",
    "Vacina contra dengue sera distribuida em postos de saude do municipio",
    "Candidatos debatem propostas para seguranca publica em eleicoes municipais",
    "Nova linha de onibus liga o centro ao aeroporto com tarifa reduzida",
    "Feira literaria atrai milhares de visitantes ao centro de convencoes",
    "Cientistas brasileiros participam de missao internacional na Antartica",
    "Escola estadual ganha laboratorio de robotica com apoio de empresa",
    "Incendio florestal destroi hectares de vegetacao no cerrado mineiro",
    "Selecao brasileira convoca jogadores para amistosos preparatorios",
    "Museu reabre com exposicao sobre a historia da imigracao no Brasil",
    "Programa social amplia distribuicao de cestas basicas em comunidades",
    "Rodovia federal ganha novo trecho duplicado apos anos de obras",
    "Tribunal julga recurso de politico condenado por corrupcao passiva",
    "Aplicativo de transporte anuncia novas regras para motoristas parceiros",
    "Universidade publica abre inscricoes para cursos de extensao gratuitos",
    "Seca prolongada afeta reservatorios e ameaca abastecimento de agua",
    "Tecnologia de reconhecimento facial e adotada em estadios de futebol",
    "Voluntarios organizam mutirao de limpeza em praias do litoral",
    "Deputados aprovam projeto de lei sobre protecao de dados pessoais",
    "Exposicao de arte contemporanea abre no centro cultural da capital",
    "Surto de gripe lota emergencias de hospitais em diversas capitais",
    "Motoristas enfrentam congestionamento recorde em rodovias durante feriado",
    "Concurso publico oferece mil vagas para nivel medio e superior",
    "Campeonato de surf reune atletas em praia do litoral nordestino",
    "ONG promove campanha de adocao de animais abandonados no fim de semana",
    "Nova temporada de serie brasileira estreia em plataforma de streaming",
    "Agricultores pedem apoio do governo apos perdas com geada severa",
    "Operacao policial apreende toneladas de drogas em porto do litoral",
    "Projeto social ensina programacao para jovens de comunidades carentes",
    "Idosos recebem segunda dose da vacina em campanha de imunizacao",
    "Parque nacional registra numero recorde de visitantes no feriado",
    "Companhia aerea cancela voos por causa de mau tempo em aeroportos",
    "Prefeitura distribui material escolar para alunos da rede municipal",
    "Conferencia sobre mudancas climaticas reune especialistas de 40 paises",
    "Jovem atleta brasileiro conquista medalha em campeonato mundial",
    "Obra de saneamento basico beneficia milhares de moradores na regiao",
    "Festival gastronomico apresenta pratos tipicos de todas as regioes",
    "Teste do Enem registra numero recorde de inscritos neste ano",
    "Trem de alta velocidade comeca operacao experimental entre capitais",
    "Biblioteca publica amplia acervo com doacoes de editoras nacionais",
    "Enchente desaloja familias em municipios do interior do estado",
    "Clube de leitura ganha espaco em escolas publicas de todo o pais",
    "Campanha contra trabalho infantil e lancada pelo ministerio publico",
    "Pescadores artesanais pedem regulamentacao para proteger a atividade",
    "Usina solar e inaugurada no semiarido com capacidade para 50 mil casas",
    "Populacao participa de audiencia publica sobre plano diretor da cidade",
    "Casal de turistas e resgatado apos se perder em trilha na serra",
    "Grupo de teatro itinerante leva pecas a cidades do interior",
    "Startup brasileira desenvolve aplicativo para monitorar qualidade do ar",
    "Governador anuncia pacote de medidas para combater a violencia urbana",
    "Corrida de rua reune 10 mil participantes no aniversario da cidade",
    "Centro de pesquisa investe em inteligencia artificial para saude",
    "Associacao de moradores reivindica melhorias no transporte coletivo",
    "Musica sertaneja domina paradas de sucesso pelo terceiro mes seguido",
    "Restaurante comunitario amplia horario para atender trabalhadores noturnos",
    "Poluicao do rio preocupa ambientalistas e moradores ribeirinhos",
    "Programa de microondas estreia com novo apresentador na televisao aberta",
    "Operacao da receita federal combate contrabando na fronteira sul",
    "Artistas de rua se apresentam em festival cultural no centro historico",
    "Ministerio da saude lanca campanha contra obesidade infantil",
    "Time de basquete feminino conquista titulo inedito na liga nacional",
    "Projeto de horta comunitaria transforma terreno baldio em area verde",
    "Policia rodoviaria intensifica fiscalizacao durante operacao de feriado",
    "Mostra de documentarios exibe filmes sobre comunidades quilombolas",
    "Startup cria plataforma de ensino a distancia para escolas rurais",
    "Feira de ciencias em escola publica apresenta projetos inovadores",
    "Bombeiros resgatam familia ilhada apos forte chuva no municipio",
    "Orquestra sinfonica faz concerto gratuito em praca publica da capital",
    "Jornalista investigativo lanca livro sobre corrupcao no poder publico",
    "Academia ao ar livre e inaugurada em parque da zona sul",
    "Pais exigem mais seguranca em escolas apos incidente com alunos",
    "Torneio de xadrez escolar reune estudantes de 50 municipios",
    "Secretaria de saude alerta para aumento de casos de conjuntivite",
    "Projeto de lei propoe aumento da licenca paternidade para 30 dias",
    "Comunidade indigena recebe titulo de posse de terra ancestral",
    "Feira de adocao encontra lar para 200 animais em um unico dia",
    "Programa de intercambio envia estudantes brasileiros para universidades europeias",
    "Ciclovia inaugurada conecta bairros residenciais ao centro comercial",
    "Grupo de idosos participa de oficina de pintura em centro comunitario",
    "Prefeito assina decreto que proibe uso de canudos plasticos na cidade",
    "Expedição cientifica mapeia recifes de coral no litoral baiano",
    "Cooperativa de reciclagem gera renda para 300 familias na regiao",
    "Liga de esports atrai milhares de espectadores em arena na capital",
    "Novo parque aquatico e inaugurado como atracao turistica regional",
    "Alunos de escola tecnica desenvolvem prototipo de carro eletrico",
    "Festival de jazz internacional traz artistas de 15 paises ao Brasil",
    "Campanha de vacinacao contra poliomielite alcanca meta de cobertura",
    "Produtores rurais adotam tecnicas de agricultura regenerativa",
    "Maratona aquatica reune nadadores profissionais em represa do interior",
    "Centro de reabilitacao para dependentes quimicos e ampliado no estado",
    "Feira de emprego oferece 5 mil vagas em diversos setores na capital",
    "Rede de bibliotecas comunitarias e criada em favelas do Rio de Janeiro",
    "Exposicao fotografica retrata o cotidiano de comunidades ribeirinhas",
    "Programa de alimentacao escolar e premiado por organizacao internacional",
    "Trilha ecologica e revitalizada em parque estadual com apoio voluntario",
    "Seminario discute inclusao de pessoas com deficiencia no mercado de trabalho",
    "Clube de astronomia promove observacao de eclipse em praca publica",
    "ONG distribui kits de higiene para familias em situacao de rua",
    "Competicao de robotica estudantil classifica equipe para mundial",
    "Nova unidade de pronto atendimento e inaugurada na periferia da cidade",
    "Grupo de escoteiros realiza acampamento educativo em reserva ecologica",
    "Documentario sobre vida marinha e premiado em festival de cinema ambiental",
    "Associacao comercial promove liquidacao coletiva no centro da cidade",
    "Campeonato de futsal feminino ganha transmissao na televisao aberta",
    "Projeto de lei visa ampliar acesso a internet em areas rurais",
    "Feirantes recebem capacitacao em boas praticas de manipulacao de alimentos",
    "Parque de diversoes itinerante chega a cidades do interior do estado",
    "Conferencia internacional sobre direitos humanos acontece na capital federal",
    "Time de volei masculino perde final do campeonato sul-americano",
    "Horta vertical e instalada em escola como projeto de educacao ambiental",
    "Prefeitura inaugura centro de atendimento ao turista no centro historico",
    "Campanha de doacao de sangue busca repor estoques em hospitais publicos",
    "Festival de danca contemporanea reune companhias de todo o continente",
    "Pesquisa revela aumento no uso de bicicletas como transporte urbano",
    "Escola de samba inicia preparativos para o desfile do proximo carnaval",
    "Programa de mentoria conecta profissionais experientes a jovens universitarios",
    "Centro de convivencia para idosos oferece atividades culturais e esportivas",
    "Operacao da policia civil desarticula rede de falsificacao de documentos",
    "Mostra de cinema independente exibe producoes de cineastas estreantes",
    "Projeto de reflorestamento planta 100 mil arvores em area degradada",
    "Festival de comida de rua movimenta economia de bairro periferico",
    "Seminario sobre empreendedorismo feminino reune 500 participantes",
    "Corrida noturna beneficente arrecada fundos para hospital infantil",
    "Nova ciclofaixa e inaugurada ligando campus universitario ao metro",
    "Grupo de voluntarios reforma casas de familias em vulnerabilidade social",
    "Campeonato estadual de atletismo revela novos talentos para selecao",
    "Feira de artesanato indigena valoriza producao cultural de aldeias locais",
    "Programa de educacao financeira e implantado em escolas municipais",
    "Reserva biologica recebe certificacao internacional de conservacao",
    "Festival de teatro de bonecos encanta criancas em praca da cidade",
    "Projeto social leva aulas de musica classica a jovens da periferia",
    "Censo demografico revela mudancas no perfil da populacao brasileira",
    "Jardim botanico promove exposicao de orquideas raras e exoticas",
    "Competicao de natacao reune atletas de clubes de todo o estado",
    "Organizacao nao governamental capacita mulheres em tecnologia da informacao",
    "Prefeitura anuncia calendario de eventos culturais para o proximo semestre",
    "Museu de historia natural recebe acervo doado por colecionador privado",
    "Torneio de tenis de mesa adapta categorias para atletas paralimpicos",
    "Centro de acolhimento para migrantes e inaugurado na regiao central",
    "Programa de bolsas de estudo beneficia estudantes de baixa renda",
    "Curso gratuito de primeiros socorros e oferecido em centros comunitarios",
    "Festival de pipas colore o ceu do parque em evento para toda familia",
    "Cooperativa agricola investe em producao organica certificada",
    "Mutirao de saude oferece exames gratuitos em comunidade rural",
    "Projeto de lei regulamenta uso de patinetes eletricos nas calcadas",
    "Orquestra de camerata se apresenta em igreja historica do seculo XVIII",
    "Feira de profissoes orienta estudantes do ensino medio sobre carreiras",
    "Centro de triagem de residuos solidos amplia capacidade de processamento",
    "Grupo de capoeira angola realiza roda aberta no largo do pelourinho",
    "Programa habitacional entrega 500 apartamentos para familias sorteadas",
    "Corrida de aventura em montanha atrai competidores de todo o sudeste",
    "Conselho tutelar alerta para aumento de denuncias de maus tratos",
    "Feira de inovacao apresenta projetos de startups de base tecnologica",
    "Torneio de poesia falada reune jovens artistas em centro cultural",
    "Nova ponte sobre rio facilita acesso entre municipios vizinhos",
    "Campanha educativa orienta populacao sobre descarte correto de pilhas",
    "Coral infantil se apresenta em cerimonia de encerramento do ano letivo",
    "Defesa civil emite alerta de risco de deslizamento em encostas",
    "Grupo de escoteiros promove plantio de mudas nativas em area degradada",
    "Associacao de artesaos inaugura loja colaborativa no mercado municipal",
    "Programa de castração gratuita atende animais de comunidades carentes",
    "Competicao de gastronomia estudantil premia melhores receitas regionais",
    "Projeto de acessibilidade instala rampas e sinalizacao tatil em calcadas",
    "Festival de balonismo atrai turistas para cidade serrana do interior",
    "Biblioteca itinerante leva livros a comunidades sem acesso a leitura",
    "Campeonato amador de futebol de varzea define semifinalistas do torneio",
    "Oficina de teatro comunitario forma novos atores para grupo local",
]

assert len(_MERCADO_TEXTS) == 25
assert len(_OUTROS_TEXTS) == 175


@pytest.fixture
def synthetic_corpus() -> pd.DataFrame:
    """DataFrame with ~200 rows (25 mercado, 175 outros) for fast tests."""
    texts = _MERCADO_TEXTS + _OUTROS_TEXTS
    labels = [1] * len(_MERCADO_TEXTS) + [0] * len(_OUTROS_TEXTS)
    return pd.DataFrame({"text": texts, "label": labels})


@pytest.fixture
def synthetic_splits(synthetic_corpus):
    """Train/val/test derived from synthetic corpus with seed=42.

    The fixture seed (42) deliberately differs from the production global
    (seed=2026). Tests check structural properties (stratification, sizes,
    determinism), not specific row indices, so the value of the seed is
    arbitrary as long as it is fixed.

    Returns (train_df, val_df, test_df).
    """
    from economy_classifier.datasets import build_train_val_test_split

    return build_train_val_test_split(synthetic_corpus, seed=42)


@pytest.fixture
def known_predictions():
    """y_true, y_pred and y_score with hand-verifiable metrics.

    TP=2, FP=1, FN=1, TN=4
    precision = 2/3 ≈ 0.6667
    recall    = 2/3 ≈ 0.6667
    f1        = 2/3 ≈ 0.6667
    accuracy  = 6/8 = 0.75
    """
    y_true = pd.Series([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = pd.Series([1, 1, 0, 0, 0, 0, 1, 0])
    y_score = pd.Series([0.92, 0.85, 0.40, 0.10, 0.05, 0.12, 0.78, 0.20])
    return y_true, y_pred, y_score


@pytest.fixture
def multi_method_predictions():
    """Predictions from 7 methods over 20 examples, standard CSV format.

    Columns: index, y_true, y_pred, y_score, method
    """
    rng = np.random.RandomState(42)
    n = 20
    y_true = np.array([1] * 5 + [0] * 15)

    methods = [
        "logreg", "linearsvc", "nb",
        "bertimbau", "finbert_ptbr", "deb3rta_base",
        "ensemble_majority",
    ]

    rows = []
    for method in methods:
        scores = rng.uniform(0, 1, size=n)
        preds = (scores >= 0.5).astype(int)
        for i in range(n):
            rows.append({
                "index": i,
                "y_true": y_true[i],
                "y_pred": preds[i],
                "y_score": round(scores[i], 4),
                "method": method,
            })

    return pd.DataFrame(rows)
