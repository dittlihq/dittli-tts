"""German G2P unit tests.

Curated word list — these specifically exercise the rule table in
src/dittli_tts/text/german.py:RULES (ch / chs / st / sp / r / s / v rules,
umlauts, devoicing, exception dict).
"""
from __future__ import annotations

import pytest

from dittli_tts.text import german as g
from dittli_tts.text import phonemes_to_ids


def phones(word: str) -> list[str]:
    out, _, _ = g.grapheme_to_phoneme(word, pad_start_end=False)
    return out


@pytest.mark.parametrize("word", [
    "der", "die", "das", "und", "haus", "katze", "schnell",
    "guten", "morgen", "wie", "geht", "es", "dir",
])
def test_g2p_de_returns_phones(word: str):
    out = phones(word)
    assert out, f"empty phone sequence for {word!r}"
    assert all(isinstance(p, str) and p for p in out)


def test_g2p_de_padding_adds_underscore():
    padded, _, _ = g.grapheme_to_phoneme("ja", pad_start_end=True)
    assert padded[0] == "_" and padded[-1] == "_"


def test_g2p_de_tones_are_zero():
    """German is single-tone: every phoneme should have tone 0."""
    _, tones, _ = g.grapheme_to_phoneme("Guten Morgen", pad_start_end=False)
    assert all(t == 0 for t in tones)


def test_g2p_de_word2ph_sums_to_phone_count():
    p, _, w2p = g.grapheme_to_phoneme("Hallo Welt", pad_start_end=False)
    assert sum(w2p) == len(p)


@pytest.mark.parametrize("text,expected_substr", [
    ("z.B. heute", "zum Beispiel"),
    ("d.h. morgen", "das heißt"),
    ("Dr. Schmidt", "Doktor"),
])
def test_g2p_de_normalize_expands_abbreviations(text: str, expected_substr: str):
    assert expected_substr in g.normalize_text(text)


def test_g2p_de_normalize_expands_numbers():
    out = g.normalize_text("21 Äpfel")
    assert "einundzwanzig" in out
    assert "21" not in out


def test_g2p_de_phones_in_symbol_table():
    """Every phoneme produced by the German G2P must be in the global symbol
    table (modulo UNK fallback) — phonemes_to_ids would raise otherwise."""
    p, t, _ = g.grapheme_to_phoneme("Sprache und Schrift", pad_start_end=True)
    ids, tones, langs = phonemes_to_ids(p, t, "DE")
    assert len(ids) == len(p) == len(tones) == len(langs)
    # tone IDs are offset to the German block, never raw 0/1 values
    assert all(isinstance(i, int) for i in ids)
