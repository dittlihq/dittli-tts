"""English G2P unit tests.

`english.py` needs `g2p_en` (which transitively pulls nltk's cmudict);
the suite skips if that toolchain isn't available.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def en():
    pytest.importorskip("g2p_en")
    from dittli_tts.text import english

    return english


@pytest.mark.parametrize("word", ["hello", "world", "weather", "today"])
def test_g2p_en_returns_phones(en, word: str):
    out, _, _ = en.grapheme_to_phoneme(word, pad_start_end=False)
    assert out
    assert all(isinstance(p, str) and p for p in out)


def test_g2p_en_padding(en):
    padded, _, _ = en.grapheme_to_phoneme("ok", pad_start_end=True)
    assert padded[0] == "_" and padded[-1] == "_"


def test_g2p_en_normalize_lowercases_and_expands_numbers(en):
    out = en.normalize_text("It's 5 PM.")
    assert out == out.lower()
    assert "5" not in out
    assert "five" in out


def test_g2p_en_word2ph_sums_to_phone_count(en):
    p, _, w2p = en.grapheme_to_phoneme("hello world", pad_start_end=False)
    assert sum(w2p) == len(p)
