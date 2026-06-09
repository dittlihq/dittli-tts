"""Initialism / acronym expansion for the G2P front-end.

Consecutive uppercase letters (e.g. "TV", "FBI", "USB") are initialisms that
should be spoken letter-by-letter ("tee-vee", "ef-bee-eye"). Without this step
they are lowercased and fed to the word-level G2P as if they were ordinary
words, which produces near-empty or garbled audio:

    "TV"  -> rules/CMU on "tv"  -> ['t', 'f'] (DE) / ['t','iy','v','iy'] (EN by luck)
    "FBI" -> CMU/neural on "fbi" -> ['b', 'ay'] ("bye")
    "GPU" -> CMU/neural on "gpu" -> ['jh', 'uw'] ("joo")

The fix runs *before* the pipeline lowercases the text: each run of two or more
uppercase letters is replaced with the space-separated letter-name spelling for
the language, which the existing G2P then pronounces correctly as words.

Trade-off: acronyms that are conventionally read as a single word (NASA, NATO)
are also spelled out ("en-ay-es-ay"). Spelling out is the safer default — it is
intelligible, where the current behaviour is silent/garbled — and the vast
majority of all-uppercase tokens users type are true initialisms.

This module is the single source of truth for the letter tables; the JS packages
(`@dittli/tts-en`, `@dittli/tts-de`) carry hand-ported copies kept in sync via
the parity tests.
"""

from __future__ import annotations

import re

# Letter-name spellings chosen so the per-language G2P pronounces each one as the
# spoken name of the letter (verified to yield vowel-bearing phoneme strings).
EN_LETTER_NAMES = {
    "A": "ay",
    "B": "bee",
    "C": "see",
    "D": "dee",
    "E": "ee",
    "F": "ef",
    "G": "jee",
    "H": "aitch",
    "I": "eye",
    "J": "jay",
    "K": "kay",
    "L": "el",
    "M": "em",
    "N": "en",
    "O": "oh",
    "P": "pee",
    "Q": "cue",
    "R": "ar",
    "S": "ess",
    "T": "tee",
    "U": "you",
    "V": "vee",
    "W": "double you",
    "X": "ex",
    "Y": "why",
    "Z": "zee",
}

DE_LETTER_NAMES = {
    "A": "a",
    "B": "be",
    "C": "ze",
    "D": "de",
    "E": "e",
    "F": "ef",
    "G": "ge",
    "H": "ha",
    "I": "i",
    "J": "jot",
    "K": "ka",
    "L": "el",
    "M": "em",
    "N": "en",
    "O": "o",
    "P": "pe",
    "Q": "ku",
    "R": "er",
    "S": "es",
    "T": "te",
    "U": "u",
    "V": "vau",
    "W": "we",
    "X": "iks",
    "Y": "ypsilon",
    "Z": "zet",
    "Ä": "ä",
    "Ö": "ö",
    "Ü": "ü",
}

_LETTER_TABLES = {"EN": EN_LETTER_NAMES, "DE": DE_LETTER_NAMES}

# A run of two or more uppercase letters. German adds the umlauts; both include
# ASCII A-Z. Single capitals (sentence-initial words, capitalised German nouns)
# are left untouched.
_RUN_RE = {
    "EN": re.compile(r"[A-Z]{2,}"),
    "DE": re.compile(r"[A-ZÄÖÜ]{2,}"),
}


def expand_initialisms(text: str, language: str) -> str:
    """Replace runs of >=2 uppercase letters with their spoken letter names.

    Must run before the pipeline lowercases the text. Surrounding spaces are
    inserted so each letter name tokenises as its own word.
    """
    lang = language.upper()
    table = _LETTER_TABLES.get(lang)
    if table is None:
        return text
    pattern = _RUN_RE[lang]

    def _replace(match: re.Match[str]) -> str:
        spelled = " ".join(table.get(ch, ch) for ch in match.group())
        return f" {spelled} "

    return pattern.sub(_replace, text)
