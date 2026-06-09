"""Number- and abbreviation-normalization unit tests.

These touch only pure functions, no model or tokenizer.
"""

from __future__ import annotations

import pytest

from dittli_tts.text.german_utils.abbreviations import expand_abbreviations as expand_de
from dittli_tts.text.german_utils.number_norm import (
    normalize_numbers as normalize_numbers_de,
)
from dittli_tts.text.german_utils.number_norm import (
    number_to_words,
)


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "null"),
        (1, "eins"),
        (7, "sieben"),
        (10, "zehn"),
        (11, "elf"),
        (12, "zwölf"),
        (16, "sechzehn"),
        (20, "zwanzig"),
        (21, "einundzwanzig"),
        (42, "zweiundvierzig"),
        (100, "einhundert"),
        (101, "einhunderteins"),
        (999, "neunhundertneunundneunzig"),
        (1_000, "eintausend"),
        (1_234, "eintausendzweihundertvierunddreißig"),
        (1_000_000, "eine Million"),
        (2_000_000, "zwei Millionen"),
        (1_000_000_000, "eine Milliarde"),
    ],
)
def test_german_number_to_words(n: int, expected: str):
    assert number_to_words(n) == expected


def test_german_number_to_words_negative():
    assert number_to_words(-5).startswith("minus ")


def test_german_normalize_numbers_in_text():
    assert "21" not in normalize_numbers_de("Ich habe 21 Äpfel")
    assert "einundzwanzig" in normalize_numbers_de("Ich habe 21 Äpfel")


@pytest.mark.parametrize(
    "inp,must_contain",
    [
        ("z.B. heute", "zum Beispiel"),
        ("z. B. heute", "zum Beispiel"),
        ("d.h. morgen", "das heißt"),
        ("Dr. Müller", "Doktor"),
        ("Prof. Schmidt", "Professor"),
        ("ca. 10", "circa"),
        ("etc.", "et cetera"),
        ("usw.", "und so weiter"),
        ("Str. 5", "Straße"),
    ],
)
def test_german_abbreviations(inp: str, must_contain: str):
    out = expand_de(inp)
    assert must_contain in out


def test_english_number_norm():
    """English number normalization uses inflect; skip if unavailable."""
    pytest.importorskip("inflect")
    from dittli_tts.text.english_utils.number_norm import normalize_numbers

    assert "5" not in normalize_numbers("I have 5 apples")
    assert "five" in normalize_numbers("I have 5 apples")
