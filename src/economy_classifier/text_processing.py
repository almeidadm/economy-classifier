"""Pre-processamento de texto — remocao de stopwords PT-BR.

Carrega a lista de stopwords do NLTK materializada em ``data/stopwords_pt.txt``
(ver ``scripts/generate_stopwords_pt.py``) e expoe dois caminhos de uso:

- :func:`load_stopwords_pt` / :func:`tfidf_stop_words` — para o
  ``TfidfVectorizer``, que aceita ``stop_words=`` e remove os tokens *apos*
  ``lowercase`` + ``strip_accents='unicode'``. A lista e normalizada no mesmo
  espaco de tokens para casar exatamente (evita o aviso de inconsistencia do
  sklearn e o bug de stopwords acentuadas que nunca casariam).
- :func:`strip_stopwords` — para BERT, cujo tokenizador subword nao tem
  ``stop_words``. A remocao acontece *upstream*, na string crua, antes da
  tokenizacao.

Nenhuma dependencia de runtime sobre o NLTK: o pipeline le apenas o ``.txt``
versionado (reprodutivel e offline-friendly no Colab).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

_DATA_FILE = Path(__file__).resolve().parent / "data" / "stopwords_pt.txt"

# Tokens-palavra (letras/digitos/_), Unicode-aware. \b...\b preserva a pontuacao
# e os espacos ao redor para que :func:`strip_stopwords` remova apenas as
# palavras-stopword, sem mexer no resto.
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _normalize(token: str) -> str:
    """Lowercase + remocao de acentos, espelhando ``strip_accents='unicode'``.

    Replica a normalizacao do ``TfidfVectorizer`` (NFKD + descarte de marcas de
    combinacao) para que a comparacao de stopwords ocorra no mesmo espaco.
    """
    decomposed = unicodedata.normalize("NFKD", token.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


@lru_cache(maxsize=1)
def load_stopwords_pt() -> frozenset[str]:
    """Conjunto normalizado de stopwords PT-BR (cacheado).

    Le ``data/stopwords_pt.txt`` (ignorando linhas em branco e comentarios
    iniciados por ``#``) e normaliza cada entrada para o espaco de tokens do
    ``TfidfVectorizer`` (lowercase, sem acento).
    """
    if not _DATA_FILE.exists():
        raise FileNotFoundError(
            f"Lista de stopwords nao encontrada em {_DATA_FILE}. "
            "Gere-a com: uv run --with nltk python scripts/generate_stopwords_pt.py"
        )
    words: set[str] = set()
    for raw_line in _DATA_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        normalized = _normalize(line)
        if normalized:
            words.add(normalized)
    if not words:
        raise ValueError(f"Lista de stopwords vazia em {_DATA_FILE}")
    return frozenset(words)


def tfidf_stop_words() -> list[str]:
    """Lista de stopwords no formato esperado por ``TfidfVectorizer(stop_words=...)``.

    Ja normalizada (lowercase, sem acento), portanto consistente com
    ``strip_accents='unicode'`` + ``lowercase`` — nao dispara o aviso de
    inconsistencia do sklearn nem deixa stopwords acentuadas escaparem da remocao.
    """
    return sorted(load_stopwords_pt())


def strip_stopwords(text: str, stopwords: frozenset[str] | None = None) -> str:
    """Remove stopwords PT-BR de uma string crua (uso upstream do BERT).

    Cada token-palavra e comparado na forma normalizada (lowercase, sem acento);
    tokens que nao sao stopword mantem sua forma de superficie original (caixa e
    acentos preservados para modelos *cased*). Pontuacao e espacos remanescentes
    sao mantidos e os espacos sao colapsados ao final. Idempotente.
    """
    if stopwords is None:
        stopwords = load_stopwords_pt()

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return "" if _normalize(token) in stopwords else token

    stripped = _WORD_RE.sub(_replace, text)
    return re.sub(r"\s+", " ", stripped).strip()
