"""Tests for economy_classifier.text_processing — remocao de stopwords PT-BR."""

import unicodedata

from economy_classifier.text_processing import (
    load_stopwords_pt,
    strip_stopwords,
    tfidf_stop_words,
)


def test_load_stopwords_nonempty_and_normalized():
    sw = load_stopwords_pt()
    assert isinstance(sw, frozenset)
    assert len(sw) > 100  # NLTK PT tem ~207 termos
    for word in sw:
        assert word == word.lower()                 # lowercase
        assert word.isascii()                        # acentos removidos (NFKD)
        assert not any(unicodedata.combining(c) for c in word)
    # Stopwords comuns presentes na forma normalizada.
    assert {"de", "que", "para"} <= sw


def test_accented_stopwords_are_matchable():
    """Acentuadas ('nao', 'esta', 'tambem') precisam casar apos strip_accents."""
    sw = load_stopwords_pt()
    assert "nao" in sw     # de "nao"/"não"
    assert "esta" in sw    # de "esta"/"está"
    assert "tambem" in sw  # de "também"


def test_load_stopwords_is_cached():
    assert load_stopwords_pt() is load_stopwords_pt()


def test_tfidf_stop_words_is_sorted_list():
    words = tfidf_stop_words()
    assert isinstance(words, list)
    assert words == sorted(words)
    assert set(words) == set(load_stopwords_pt())


def test_strip_stopwords_removes_only_stopword_tokens():
    assert strip_stopwords("o mercado de acoes subiu") == "mercado acoes subiu"


def test_strip_stopwords_handles_accents_and_preserves_surface_form():
    # "não"/"está" sao stopwords (acentuadas); "Mercado" mantem caixa e acento.
    assert strip_stopwords("O Mercado não está caro") == "Mercado caro"


def test_strip_stopwords_preserves_punctuation():
    assert strip_stopwords("Hoje, o mercado caiu.") == "Hoje, mercado caiu."


def test_strip_stopwords_is_idempotent():
    text = "O mercado de acoes nao parou de subir hoje"
    once = strip_stopwords(text)
    assert strip_stopwords(once) == once


def test_strip_stopwords_all_stopwords_collapses_to_empty():
    assert strip_stopwords("de a o que e para") == ""
    assert strip_stopwords("") == ""
