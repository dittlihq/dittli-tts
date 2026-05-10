"""Symbol-table and id-mapping invariants."""
from __future__ import annotations

import pytest

from dittli_tts.nn import commons
from dittli_tts.text import phonemes_to_ids
from dittli_tts.text.symbols import (
    language_id_map,
    language_tone_start_map,
    symbols,
)


def test_symbol_table_has_pad_and_unk():
    assert "_" in symbols
    assert "UNK" in symbols
    assert symbols.index("_") == 0  # pad must be id 0 (used for blanks)


def test_symbol_table_unique():
    assert len(symbols) == len(set(symbols))


def test_language_maps_consistent():
    assert "EN" in language_id_map and "DE" in language_id_map
    assert "EN" in language_tone_start_map and "DE" in language_tone_start_map


@pytest.mark.parametrize("lang", ["EN", "DE"])
def test_phonemes_to_ids_roundtrip(lang: str):
    """Every symbol maps back to its index, tones get language-offset."""
    ids, tones, langs = phonemes_to_ids(["_", "UNK"], [0, 0], lang)
    assert ids == [symbols.index("_"), symbols.index("UNK")]
    assert all(t == language_tone_start_map[lang] for t in tones)
    assert all(l == language_id_map[lang] for l in langs)


def test_phonemes_to_ids_unknown_falls_back_to_unk():
    ids, _, _ = phonemes_to_ids(["definitely-not-a-symbol"], [0], "EN")
    assert ids == [symbols.index("UNK")]


def test_insert_blanks():
    assert commons.insert_blanks([], 0) == [0]
    assert commons.insert_blanks([1], 0) == [0, 1, 0]
    assert commons.insert_blanks([1, 2, 3], 0) == [0, 1, 0, 2, 0, 3, 0]
