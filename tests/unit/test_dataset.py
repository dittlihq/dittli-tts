"""Unit tests for dataset helper functions."""

from __future__ import annotations

from dittli_tts.data.dataset import _phones_path


def test_phones_path_is_language_tagged():
    en = _phones_path("/data/wav/clip.wav", "EN")
    de = _phones_path("/data/wav/clip.wav", "DE")
    assert en != de, "EN and DE must produce different cache paths"
    assert "en" in en
    assert "de" in de


def test_phones_path_language_tag_is_lowercase():
    assert _phones_path("/a.wav", "EN").endswith(".en.phones.pt")
    assert _phones_path("/a.wav", "DE").endswith(".de.phones.pt")
