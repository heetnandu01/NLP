"""Utility helpers for the opinion mining project."""

from __future__ import annotations

import re
import string
from typing import Iterable

import nltk
from nltk.corpus import stopwords


def ensure_nltk_resources() -> None:
    """Download the NLTK resources needed by the project if missing."""
    nltk.download("stopwords", quiet=True)


_STOP_WORDS: set[str] | None = None


def get_stop_words() -> set[str]:
    """Return the English stop words set used for preprocessing."""
    global _STOP_WORDS

    if _STOP_WORDS is not None:
        return _STOP_WORDS

    ensure_nltk_resources()

    try:
        _STOP_WORDS = set(stopwords.words("english"))
    except LookupError:
        # Custom stop words that exclude negation words important for sentiment
        _STOP_WORDS = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "but",
            "by",
            "for",
            "if",
            "in",
            "into",
            "it",
            "of",
            "on",
            "or",
            "such",
            "that",
            "the",
            "their",
            "then",
            "there",
            "these",
            "they",
            "this",
            "to",
            "will",
            "with",
            "you",
        }

    return _STOP_WORDS


PUNCT_TRANSLATION_TABLE = str.maketrans("", "", string.punctuation)


def preprocess_text(text: object) -> str:
    """Lowercase text, remove punctuation, and drop stop words.

    The function keeps the implementation intentionally simple so it is easy to
    understand for beginners and suitable for a small assignment project.
    """
    if text is None:
        return ""

    cleaned_text = str(text).lower()
    cleaned_text = cleaned_text.translate(PUNCT_TRANSLATION_TABLE)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    stop_words = get_stop_words()
    tokens = [word for word in cleaned_text.split() if word not in stop_words]
    return " ".join(tokens)


def preprocess_corpus(texts: Iterable[object]) -> list[str]:
    """Apply preprocessing to a collection of texts."""
    return [preprocess_text(text) for text in texts]
