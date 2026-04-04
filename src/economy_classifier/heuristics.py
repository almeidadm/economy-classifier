"""Transparent weighted-lexicon heuristics for economy text triage."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
import unicodedata


SIGNAL_WEIGHTS = {
    "nuclear": 4,
    "setorial": 3,
    "contextual": 2,
    "fraco": 1,
}

MARKET_SCORE_THRESHOLD = 2.6
AMBIGUOUS_SCORE_THRESHOLD = 1.0

THEME_RULES: tuple[dict[str, object], ...] = (
    {
        "theme": "macroeconomia",
        "signal": "nuclear",
        "weight": 4,
        "patterns": (
            "economia",
            "economia brasileira",
            "atividade economica",
            "crescimento economico",
            "financas",
            "mercado financeiro",
            "crise economica",
            "recessao",
            "pib",
            "inflacao",
            "ipca",
            "juros",
            "taxa selic",
            "juros futuros",
            "copom",
            "meta de inflacao",
            "politica monetaria",
            "dolar",
            "cambio",
            "desemprego",
            "salario minimo",
            "custo de vida",
        ),
    },
    {
        "theme": "tributacao_e_contas_publicas",
        "signal": "nuclear",
        "weight": 4,
        "patterns": (
            "impostos",
            "tributacao",
            "tributo",
            "imposto de renda",
            "reforma tributaria",
            "gasto publico",
            "divida publica",
            "deficit fiscal",
            "superavit",
            "orcamento",
            "arcabouco fiscal",
            "carga tributaria",
            "receita federal",
            "novo imposto",
            "aumento de imposto",
            "governo vai taxar",
            "governo vai cobrar",
            "imposto sobre heranca",
            "imposto sobre grandes fortunas",
        ),
    },
    {
        "theme": "sistema_financeiro_e_bancos",
        "signal": "setorial",
        "weight": 3,
        "patterns": (
            "banco central",
            "caixa economica",
            "banco do brasil",
            "bndes",
            "cvm",
            "tesouro direto",
            "poupanca",
            "credito",
            "cartao de credito",
            "juros do cartao",
            "rotativo do cartao",
            "limite do cartao",
            "spread bancario",
            "inadimplencia",
            "score de credito",
            "open finance",
            "fintech",
            "consorcio",
            "emprestimo",
            "financiamento",
            "renegociacao de dividas",
            "credito consignado",
            "pix",
            "bloqueio de conta",
            "confisco da poupanca",
            "moeda digital",
            "real digital",
            "drex",
            "conta bloqueada",
            "valores a receber",
            "registrato",
            "rombo nos bancos",
            "nubank faliu",
            "nubank quebrou",
            "caixa vai ser privatizada",
            "banco do brasil vai ser privatizado",
        ),
    },
    {
        "theme": "beneficios_previdencia_e_credito_popular",
        "signal": "contextual",
        "weight": 2,
        "patterns": (
            "fgts",
            "saque do fgts",
            "fim do fgts",
            "inss",
            "aposentadoria",
            "previdencia",
            "bolsa familia",
            "auxilio emergencial",
            "cadunico",
            "mei",
            "simples nacional",
            "compras internacionais",
            "taxacao internacional",
            "taxa das blusinhas",
            "fim da aposentadoria",
            "corte de beneficios",
            "14o salario",
            "14o salario inss",
            "revisao da vida toda",
            "abono salarial",
            "pis/pasep",
            "seguro-desemprego",
            "cnh social",
            "vale gas",
            "tarifa social",
            "desenrola brasil",
            "perdao de dividas",
            "limpar nome",
            "aumentar score",
            "cartao aprovado para negativado",
            "emprestimo para negativado",
            "restituicao via pix",
        ),
    },
    {
        "theme": "investimentos_e_cripto",
        "signal": "setorial",
        "weight": 3,
        "patterns": (
            "criptomoeda",
            "bitcoin",
            "token",
            "ibovespa",
            "bolsa de valores",
            "mercado de capitais",
            "renda fixa",
            "renda variavel",
            "cdb",
            "cdi",
            "lci",
            "lca",
            "tesouro selic",
            "tesouro ipca",
            "tesouro prefixado",
            "debenture",
            "debentures",
            "fundo de investimento",
            "fundos de investimento",
            "fundo imobiliario",
            "fundos imobiliarios",
            "fii",
            "etf",
            "oferta publica inicial",
            "ipo",
            "follow on",
            "bookbuilding",
            "gestao de recursos",
            "investidor institucional",
            "capital aberto",
            "lucro liquido",
            "receita liquida",
            "resultado trimestral",
            "balanco trimestral",
            "guidance",
            "fluxo de caixa",
            "ebitda",
            "margem ebitda",
            "fusao",
            "aquisicao",
            "fusoes e aquisicoes",
            "valuation",
            "dividend yield",
            "spread de credito",
            "acoes da petrobras",
            "lucro da petrobras",
            "dividendos",
            "moeda unica mercosul",
            "peso real",
            "moeda do brics",
        ),
    },
    {
        "theme": "fraudes_e_desinformacao_financeira",
        "signal": "contextual",
        "weight": 2,
        "patterns": (
            "piramide financeira",
            "golpe financeiro",
            "fraude bancaria",
            "investimento falso",
            "rendimento garantido",
            "robo do pix",
            "jogo do tigrinho",
            "fortune tiger",
            "bets",
            "apostas esportivas",
            "lucro rapido",
            "renda extra garantida",
            "ganhar dinheiro avaliando",
            "avaliador de marcas",
            "dinheiro assistindo videos",
            "cashback em dobro",
            "multiplicador de pix",
            "golpe do falso boleto",
            "falsa central de atendimento",
            "aplicativo clonado",
            "falso emprego",
            "vaga de emprego home office",
            "taxacao do whatsapp",
            "taxacao da poupanca",
            "governo vai confiscar",
            "confisco de bens",
            "fim do dinheiro fisico",
            "faz o l imposto",
            "heranca bloqueada",
            "conta bloqueada na caixa",
            "cadastro suspenso",
            "atualizacao cadastral obrigatoria",
            "clique aqui para resgatar",
        ),
    },
)

TERM_OVERRIDES: dict[str, dict[str, object]] = {
    "cambio": {"theme": "macroeconomia", "signal": "fraco", "weight": 1},
    "pix": {"theme": "sistema_financeiro_e_bancos", "signal": "fraco", "weight": 1},
    "credito": {"theme": "sistema_financeiro_e_bancos", "signal": "fraco", "weight": 1},
    "token": {"theme": "investimentos_e_cripto", "signal": "contextual", "weight": 2},
    "economia": {"theme": "macroeconomia", "signal": "contextual", "weight": 2},
    "piramide financeira": {
        "theme": "fraudes_e_desinformacao_financeira",
        "signal": "setorial",
        "weight": 3,
    },
    "golpe financeiro": {
        "theme": "fraudes_e_desinformacao_financeira",
        "signal": "setorial",
        "weight": 3,
    },
    "aplicativo clonado": {
        "theme": "fraudes_e_desinformacao_financeira",
        "signal": "fraco",
        "weight": 1,
    },
    "vaga de emprego home office": {
        "theme": "fraudes_e_desinformacao_financeira",
        "signal": "fraco",
        "weight": 1,
    },
}

NEGATIVE_CONTEXT_RULES: dict[str, dict[str, object]] = {
    "cambio": {
        "negative_contexts": ("carro", "motor", "embreagem", "marcha", "oficina", "automotivo"),
        "positive_contexts": ("dolar", "banco central", "cotacao", "moeda", "mercado financeiro"),
        "penalty": -4.0,
        "reason": "cambio_contexto_automotivo",
    },
    "credito": {
        "negative_contexts": ("autor", "autoria", "foto", "legenda", "disciplina", "carga horaria"),
        "positive_contexts": ("banco", "emprestimo", "consignado", "financiamento", "score"),
        "penalty": -3.0,
        "reason": "credito_contexto_editorial_ou_academico",
    },
    "token": {
        "negative_contexts": ("api", "jwt", "login", "autenticacao", "credencial", "endpoint"),
        "positive_contexts": ("criptomoeda", "blockchain", "investimento", "bitcoin"),
        "penalty": -3.0,
        "reason": "token_contexto_tecnico",
    },
}

POLITICAL_SOCIAL_CONTEXTS = (
    "eleicao", "eleicoes", "eleitoral", "candidato", "candidatos",
    "campanha", "presidencia", "presidente", "prefeitura", "prefeito",
    "vereador", "governador", "deputado", "deputados", "senador",
    "senadores", "partido", "camara", "senado", "congresso",
    "debate", "entrevista", "politico", "politicos", "governo",
    "ministro", "ministra", "corrupcao", "propina", "projeto de lei", "plenario",
)

ECONOMIA_NON_FINANCIAL_CONTEXTS = (
    "agua", "energia", "bateria", "modo economia", "jogos",
    "videogame", "gasolina", "combustivel", "internet",
)

TECH_RH_CONTEXTS = (
    "api", "jwt", "login", "backend", "frontend", "servidor",
    "sistema", "curriculo", "recrutamento", "vaga", "processo seletivo",
    "desenvolvedor", "autenticacao",
)

TECH_RH_TRIGGER_TERMS = frozenset({
    "aplicativo clonado",
    "falso emprego",
    "vaga de emprego home office",
    "token",
})

# Embedded keyword list (195 terms from the canonical palavras-chave.txt)
DEFAULT_KEYWORDS: tuple[str, ...] = (
    "economia", "finanças", "mercado financeiro", "crise econômica", "recessão",
    "pib", "inflação", "ipca", "juros", "taxa selic", "dólar", "câmbio",
    "desemprego", "salário mínimo", "custo de vida", "impostos", "tributação",
    "imposto de renda", "reforma tributária", "gasto público", "dívida pública",
    "déficit fiscal", "superávit", "orçamento", "banco central", "receita federal",
    "caixa econômica", "banco do brasil", "bndes", "cvm", "tesouro direto",
    "poupança", "crédito", "empréstimo", "financiamento", "crédito consignado",
    "pix", "imposto sobre pix", "taxação do pix", "bloqueio de conta",
    "confisco da poupança", "moeda digital", "real digital", "drex",
    "fgts", "saque do fgts", "fim do fgts", "inss", "aposentadoria",
    "previdência", "bolsa família", "auxílio emergencial", "cadúnico", "mei",
    "simples nacional", "compras internacionais", "taxação internacional",
    "taxa das blusinhas", "criptomoeda", "bitcoin", "token",
    "pirâmide financeira", "golpe financeiro", "fraude bancária",
    "investimento falso", "rendimento garantido", "novo imposto",
    "aumento de imposto", "governo vai taxar", "governo vai cobrar",
    "economia quebrou", "banco faliu", "inflação disparou", "dólar disparou",
    "fim da aposentadoria", "corte de benefícios", "conta bloqueada",
    "dinheiro esquecido", "valores a receber", "registrato",
    "14º salário", "14º salário inss", "revisão da vida toda",
    "abono salarial", "pis/pasep", "seguro-desemprego", "cnh social",
    "vale gás", "tarifa social", "desenrola brasil", "perdão de dívidas",
    "limpar nome", "aumentar score", "cartão aprovado para negativado",
    "empréstimo para negativado", "restituição via pix", "robô do pix",
    "jogo do tigrinho", "fortune tiger", "bets", "apostas esportivas",
    "lucro rápido", "renda extra garantida", "ganhar dinheiro avaliando",
    "avaliador de marcas", "dinheiro assistindo vídeos", "cashback em dobro",
    "multiplicador de pix", "golpe do falso boleto",
    "falsa central de atendimento", "aplicativo clonado", "falso emprego",
    "vaga de emprego home office", "imposto sobre herança",
    "imposto sobre grandes fortunas", "taxação do whatsapp",
    "taxação da poupança", "governo vai confiscar", "confisco de bens",
    "fim do dinheiro físico", "moeda única mercosul", "peso real",
    "moeda do brics", "rombo nos bancos", "nubank faliu", "nubank quebrou",
    "caixa vai ser privatizada", "banco do brasil vai ser privatizado",
    "ações da petrobras", "lucro da petrobras", "dividendos",
    "faz o l imposto", "herança bloqueada", "conta bloqueada na caixa",
    "cadastro suspenso", "atualização cadastral obrigatória",
    "clique aqui para resgatar", "economia brasileira",
    "atividade economica", "crescimento economico", "politica monetaria",
    "copom", "meta de inflacao", "juros futuros", "bolsa de valores",
    "ibovespa", "mercado de capitais", "renda fixa", "renda variavel",
    "cdb", "cdi", "lci", "lca", "tesouro selic", "tesouro ipca",
    "tesouro prefixado", "debenture", "debentures", "fundo de investimento",
    "fundos de investimento", "fundo imobiliario", "fundos imobiliarios",
    "fii", "etf", "oferta publica inicial", "ipo", "follow on",
    "bookbuilding", "gestao de recursos", "investidor institucional",
    "capital aberto", "lucro liquido", "receita liquida",
    "resultado trimestral", "balanco trimestral", "guidance",
    "fluxo de caixa", "ebitda", "margem ebitda", "fusao", "aquisicao",
    "fusoes e aquisicoes", "valuation", "dividend yield",
    "spread de credito", "cartao de credito", "juros do cartao",
    "rotativo do cartao", "limite do cartao", "renegociacao de dividas",
    "inadimplencia", "score de credito", "fintech", "open finance",
    "consorcio", "spread bancario",
)


@dataclass(slots=True)
class CatalogEntry:
    """Structured keyword metadata used to score incoming texts."""

    term: str
    normalized_term: str
    theme: str
    signal: str
    weight: int
    source: str


def normalize_text(text: str) -> str:
    """Normalize casing, accents and spacing for keyword matching."""
    lower_text = text.lower().strip().replace("º", "o").replace("ª", "a")
    without_accents = "".join(
        character
        for character in unicodedata.normalize("NFKD", lower_text)
        if not unicodedata.combining(character)
    )
    cleaned = re.sub(r"[^a-z0-9/]+", " ", without_accents)
    return re.sub(r"\s+", " ", cleaned).strip()


def _classify_term(term: str) -> CatalogEntry:
    normalized_term = normalize_text(term)
    override = TERM_OVERRIDES.get(normalized_term)
    if override is not None:
        return CatalogEntry(
            term=term,
            normalized_term=normalized_term,
            theme=str(override["theme"]),
            signal=str(override["signal"]),
            weight=int(override["weight"]),
            source="override",
        )

    for rule in THEME_RULES:
        patterns = tuple(str(pattern) for pattern in rule["patterns"])
        if any(pattern in normalized_term for pattern in patterns):
            return CatalogEntry(
                term=term,
                normalized_term=normalized_term,
                theme=str(rule["theme"]),
                signal=str(rule["signal"]),
                weight=int(rule["weight"]),
                source="rule",
            )

    return CatalogEntry(
        term=term,
        normalized_term=normalized_term,
        theme="sinais_genericos",
        signal="fraco",
        weight=1,
        source="fallback",
    )


def build_default_catalog() -> list[CatalogEntry]:
    """Build the structured catalog from the embedded keyword list."""
    return [_classify_term(term) for term in DEFAULT_KEYWORDS]


def _compile_term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


def _count_context_hits(normalized_text: str, contexts: tuple[str, ...] | list[str]) -> int:
    return sum(1 for context in contexts if _compile_term_pattern(str(context)).search(normalized_text))


def _has_strong_signal(signal_counts: Counter[str]) -> bool:
    return signal_counts["nuclear"] >= 1 or signal_counts["setorial"] >= 2


def _append_penalty_reason(
    penalty_reasons: list[dict[str, object]],
    *,
    rule: str,
    penalty: float,
    hits: int,
) -> None:
    penalty_reasons.append({
        "rule": rule,
        "penalty": round(float(penalty), 4),
        "hits": hits,
    })


def _local_negative_adjustment(
    normalized_text: str,
    term_counter: Counter[str],
    *,
    penalty_reasons: list[dict[str, object]],
) -> float:
    total_penalty = 0.0
    for term, rule in NEGATIVE_CONTEXT_RULES.items():
        if term_counter.get(term, 0) == 0:
            continue
        negative_hits = _count_context_hits(normalized_text, tuple(rule["negative_contexts"]))
        positive_hits = _count_context_hits(normalized_text, tuple(rule["positive_contexts"]))
        if negative_hits and not positive_hits:
            penalty = float(rule["penalty"]) - (0.5 * max(0, negative_hits - 1))
            total_penalty += penalty
            _append_penalty_reason(
                penalty_reasons,
                rule=str(rule["reason"]),
                penalty=penalty,
                hits=negative_hits,
            )
    return total_penalty


def _global_negative_adjustment(
    normalized_text: str,
    *,
    term_counter: Counter[str],
    signal_counts: Counter[str],
    valid_themes: set[str],
    raw_score: float,
    penalty_reasons: list[dict[str, object]],
) -> float:
    total_penalty = 0.0
    has_strong = _has_strong_signal(signal_counts)
    has_market_theme = any(
        theme in {"sistema_financeiro_e_bancos", "investimentos_e_cripto"} for theme in valid_themes
    )

    political_hits = _count_context_hits(normalized_text, POLITICAL_SOCIAL_CONTEXTS)
    if political_hits >= 2:
        penalty = -4.5
        if has_market_theme:
            penalty += 1.5
        if has_strong:
            penalty += 1.0
        penalty -= 0.5 * min(political_hits - 2, 4)
        total_penalty += penalty
        _append_penalty_reason(penalty_reasons, rule="densidade_politico_social", penalty=penalty, hits=political_hits)

    economia_hits = _count_context_hits(normalized_text, ECONOMIA_NON_FINANCIAL_CONTEXTS)
    if term_counter.get("economia", 0) > 0 and economia_hits >= 1 and not has_strong and raw_score <= 6:
        penalty = -3.0 - (0.5 * min(economia_hits - 1, 2))
        total_penalty += penalty
        _append_penalty_reason(penalty_reasons, rule="economia_contexto_nao_financeiro", penalty=penalty, hits=economia_hits)

    tech_rh_hits = _count_context_hits(normalized_text, TECH_RH_CONTEXTS)
    trigger_hits = sum(term_counter.get(term, 0) for term in TECH_RH_TRIGGER_TERMS)
    if trigger_hits >= 1 and tech_rh_hits >= 2 and not has_strong:
        penalty = -2.5 - (0.25 * min(tech_rh_hits - 2, 4))
        total_penalty += penalty
        _append_penalty_reason(penalty_reasons, rule="contexto_tecnico_ou_rh", penalty=penalty, hits=tech_rh_hits)

    return total_penalty


def classify_score(score: float, *, has_strong_signal: bool) -> str:
    """Map the normalized score to the public 3-band heuristic output."""
    if score >= MARKET_SCORE_THRESHOLD and has_strong_signal:
        return "mercado"
    if score >= AMBIGUOUS_SCORE_THRESHOLD:
        return "ambiguo"
    return "outros"


def score_text(text: str, entries: list[CatalogEntry]) -> dict[str, object]:
    """Score a text using the weighted lexicon catalog."""
    normalized_text = normalize_text(text)
    matches: list[dict[str, object]] = []
    signal_counts: Counter[str] = Counter()
    theme_counts: Counter[str] = Counter()
    term_counter: Counter[str] = Counter()
    penalty_reasons: list[dict[str, object]] = []
    raw_score = 0.0

    for entry in entries:
        occurrences = len(_compile_term_pattern(entry.normalized_term).findall(normalized_text))
        if occurrences == 0:
            continue
        capped = min(occurrences, 2)
        contribution = float(entry.weight * capped)
        matches.append({
            "term": entry.term,
            "normalized_term": entry.normalized_term,
            "theme": entry.theme,
            "signal": entry.signal,
            "weight": entry.weight,
            "occurrences": occurrences,
            "contribution": contribution,
        })
        raw_score += contribution
        signal_counts[entry.signal] += 1
        theme_counts[entry.theme] += 1
        term_counter[entry.normalized_term] += occurrences

    valid_themes = {t for t, c in theme_counts.items() if c > 0 and t != "sinais_genericos"}
    has_strong = _has_strong_signal(signal_counts)
    diversity_bonus = float(max(0, len(valid_themes) - 1) if has_strong else 0)

    negative_adjustment = _local_negative_adjustment(
        normalized_text, term_counter, penalty_reasons=penalty_reasons,
    )
    negative_adjustment += _global_negative_adjustment(
        normalized_text,
        term_counter=term_counter,
        signal_counts=signal_counts,
        valid_themes=valid_themes,
        raw_score=raw_score,
        penalty_reasons=penalty_reasons,
    )

    adjusted_score = raw_score + diversity_bonus + negative_adjustment
    word_count = len(normalized_text.split()) if normalized_text else 0
    normalization_factor = max(1.0, math.log1p(word_count))
    final_score = adjusted_score / normalization_factor

    return {
        "score": round(float(final_score), 4),
        "raw_score": round(float(raw_score), 4),
        "adjusted_score": round(float(adjusted_score), 4),
        "diversity_bonus": round(float(diversity_bonus), 4),
        "negative_adjustment": round(float(negative_adjustment), 4),
        "normalization_factor": round(float(normalization_factor), 4),
        "word_count": word_count,
        "classification": classify_score(final_score, has_strong_signal=has_strong),
        "terms_found": matches,
        "themes_found": sorted(valid_themes),
        "signal_counts": dict(signal_counts),
        "normalized_text": normalized_text,
        "penalty_reasons": penalty_reasons,
    }
