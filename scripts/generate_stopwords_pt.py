"""Gera a lista de stopwords PT-BR versionada a partir do corpus do NLTK.

Script *one-shot* de proveniencia: o pipeline em runtime NAO importa NLTK — ele
le o arquivo materializado ``src/economy_classifier/data/stopwords_pt.txt``. Este
gerador existe apenas para documentar de onde a lista veio e permitir
reproduzi-la deterministicamente. Rode-o quando quiser atualizar a lista:

    uv run --with nltk python scripts/generate_stopwords_pt.py

O arquivo de saida e congelado no git, garantindo reprodutibilidade exata
independente da versao do NLTK instalada (CLAUDE.md: "Reproducibilidade por
construcao") e funcionamento offline no Colab (sem ``nltk.download`` em runtime).
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "src" / "economy_classifier" / "data" / "stopwords_pt.txt"
)


def main() -> None:
    import nltk
    from nltk.corpus import stopwords

    with tempfile.TemporaryDirectory() as download_dir:
        nltk.download("stopwords", download_dir=download_dir, quiet=True)
        nltk.data.path.insert(0, download_dir)
        words = sorted(set(stopwords.words("portuguese")))
        nltk_version = nltk.__version__

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Stopwords PT-BR — fonte: NLTK corpus `stopwords` (portuguese).",
        f"# Materializada em {date.today().isoformat()} a partir de nltk=={nltk_version}.",
        f"# Total: {len(words)} termos. Uma stopword por linha; linhas iniciadas",
        "# por '#' sao comentarios. Regerar com: scripts/generate_stopwords_pt.py",
        "# As formas sao acentuadas/originais; a normalizacao (lowercase +",
        "# strip_accents=unicode) e aplicada em load_stopwords_pt() para casar",
        "# com o espaco de tokens do TfidfVectorizer (strip_accents='unicode').",
    ]
    OUTPUT.write_text("\n".join([*header, *words, ""]), encoding="utf-8")
    print(f"Escreveu {len(words)} stopwords em {OUTPUT}")


if __name__ == "__main__":
    main()
