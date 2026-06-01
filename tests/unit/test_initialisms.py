"""Initialism / acronym expansion in the G2P front-end.

Regression tests for the bug where consecutive uppercase letters ("TV",
"FBI", ...) were lowercased and fed to the word-level G2P, producing
near-empty or garbled audio (e.g. "TV" -> ['t', 'f'] in German).
"""

from __future__ import annotations

import pytest

from dittli_tts.text import get_g2p
from dittli_tts.text.initialisms import (
    DE_LETTER_NAMES,
    EN_LETTER_NAMES,
    expand_initialisms,
)

# Vowel phoneme symbols per language — every spoken letter name must contain at
# least one of these, otherwise the synthesiser produces near-silent audio.
_EN_VOWELS = {"aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"}
_DE_VOWELS = set("aeiouäöüɛɔœøyʏʊɪ") | {"aː", "eː", "iː", "oː", "uː", "ɛː", "yː", "øː", "aʊ", "aɪ", "ɔʏ", "ə", "ɐ"}


def test_expand_leaves_normal_text_untouched():
    assert expand_initialisms("Hello world", "EN") == "Hello world"
    # Capitalised German nouns are single capitals (runs of 1) — untouched.
    assert expand_initialisms("Die Serie", "DE") == "Die Serie"


def test_expand_spells_out_uppercase_runs():
    assert expand_initialisms("TV", "EN").split() == ["tee", "vee"]
    assert expand_initialisms("TV", "DE").split() == ["te", "vau"]
    # Embedded run followed by a hyphenated word ("TV-Serie").
    assert expand_initialisms("TV-Serie", "DE").split() == ["te", "vau", "-Serie"]


@pytest.mark.parametrize("lang", ["EN", "DE"])
def test_every_letter_name_yields_a_vowel(lang: str):
    """Each spelled-out letter must phonemise to at least one vowel."""
    normalize, g2p = get_g2p(lang)
    table = EN_LETTER_NAMES if lang == "EN" else DE_LETTER_NAMES
    vowels = _EN_VOWELS if lang == "EN" else _DE_VOWELS
    for letter, name in table.items():
        phones = [p for p in g2p(normalize(name))[0] if p != "_"]
        assert any(p in vowels for p in phones), f"{lang} {letter!r} -> {name!r} gave {phones}"


@pytest.mark.parametrize(
    "lang,acronym",
    [("EN", "FBI"), ("EN", "GPU"), ("EN", "TV"), ("DE", "TV"), ("DE", "USB"), ("DE", "ABC")],
)
def test_acronyms_phonemise_with_vowels(lang: str, acronym: str):
    """The reported bug: acronyms used to produce vowel-less, near-silent output."""
    normalize, g2p = get_g2p(lang)
    vowels = _EN_VOWELS if lang == "EN" else _DE_VOWELS
    phones = [p for p in g2p(normalize(acronym))[0] if p != "_"]
    assert sum(p in vowels for p in phones) >= 2, f"{acronym} -> {phones}"


def test_tv_serie_keeps_the_word_part():
    """Full regression for "TV-Serie": acronym spelled out, "Serie" intact."""
    normalize, g2p = get_g2p("DE")
    phones = [p for p in g2p(normalize("TV-Serie"))[0] if p != "_"]
    # te-vau (t e f a ʊ) followed by serie (s e ʁ i ː)
    assert phones == ["t", "e", "f", "a", "ʊ", "s", "e", "ʁ", "i", "ː"]
