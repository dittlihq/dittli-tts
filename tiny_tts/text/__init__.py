from .symbols import *


_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def phonemes_to_ids(cleaned_text, tones, language, symbol_to_id=None):
    """Converts a list of phoneme symbols to a sequence of integer IDs."""
    symbol_to_id_map = symbol_to_id if symbol_to_id else _symbol_to_id
    unk_id = symbol_to_id_map.get("UNK")
    if unk_id is None:
        phones = [symbol_to_id_map[symbol] for symbol in cleaned_text]
    else:
        phones = [symbol_to_id_map.get(symbol, unk_id) for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for _ in phones]
    return phones, tones, lang_ids


def get_g2p(language: str):
    """Return the (normalize_text, grapheme_to_phoneme) pair for a language code."""
    lang = language.upper()
    if lang == "EN":
        from .english import normalize_text, grapheme_to_phoneme
        return normalize_text, grapheme_to_phoneme
    if lang == "DE":
        from .german import normalize_text, grapheme_to_phoneme
        return normalize_text, grapheme_to_phoneme
    raise ValueError(f"No G2P registered for language {language!r}.")
