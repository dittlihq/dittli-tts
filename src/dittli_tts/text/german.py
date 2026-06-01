"""Rule-based German G2P.

Public API mirrors dittli_tts.text.english:
    grapheme_to_phoneme(text)  -> (phones, tones, word2ph)
    normalize_text(text)       -> str

The JS port (src/node/g2p_de.js) is a 1:1 translation of this file. The
RULES table and EXCEPTION_DICT below are the source of truth — they are
emitted as JSON (scripts/gen_de_rules.py) so both implementations can stay
in sync.

Output phonemes use IPA tokens that already exist in dittli_tts.text.symbols
(after appending "yː" to de_symbols). All German tones are 0 (single-tone).
"""

from __future__ import annotations

from .german_utils.abbreviations import expand_abbreviations
from .german_utils.number_norm import normalize_numbers
from .initialisms import expand_initialisms
from .symbols import symbols as _SYMBOLS

# Common loanwords whose pronunciation deviates from German rules.
EXCEPTION_DICT: dict[str, list[str]] = {
    "café": ["k", "a", "f", "e", "ː"],
    "cafe": ["k", "a", "f", "e", "ː"],
    "computer": ["k", "ɔ", "m", "p", "j", "u", "ː", "t", "ɐ"],
    "virus": ["v", "i", "ː", "r", "ʊ", "s"],
    "vase": ["v", "a", "ː", "z", "ə"],
    "violine": ["v", "i", "o", "l", "i", "ː", "n", "ə"],
    "vital": ["v", "i", "t", "a", "ː", "l"],
    "vegan": ["v", "e", "ː", "g", "a", "ː", "n"],
    "genre": ["ʒ", "ã", "ʁ", "ə"],  # ã not in symbols → will UNK; acceptable
    "team": ["t", "i", "ː", "m"],
    "email": ["i", "ː", "m", "e", "ː", "l"],
    "internet": ["ɪ", "n", "t", "ɐ", "n", "ɛ", "t"],
    "online": ["ɔ", "n", "l", "a", "ɪ", "n"],
    "chaos": ["k", "a", "ː", "ɔ", "s"],
    "charakter": ["k", "a", "ʁ", "a", "k", "t", "ɐ"],
    "china": ["ç", "i", "ː", "n", "a"],
    "chemie": ["ç", "e", "m", "i", "ː"],
    "und": ["ʊ", "n", "t"],  # final-devoicing -d → -t
    "ist": ["ɪ", "s", "t"],
    "die": ["d", "i", "ː"],
    "der": ["d", "e", "ː", "ɐ"],
    "das": ["d", "a", "s"],
    "ein": ["a", "ɪ", "n"],
    "eine": ["a", "ɪ", "n", "ə"],
    "nicht": ["n", "ɪ", "ç", "t"],
    "ich": ["ɪ", "ç"],
    "auch": ["a", "ʊ", "x"],
    "sie": ["z", "i", "ː"],
    "er": ["e", "ː", "ɐ"],
    "wir": ["v", "i", "ː", "ɐ"],
    "es": ["ɛ", "s"],
    "in": ["ɪ", "n"],
    "mit": ["m", "ɪ", "t"],
    "von": ["f", "ɔ", "n"],
    "zu": ["ts", "u", "ː"],
    "den": ["d", "e", "ː", "n"],
    "dem": ["d", "e", "ː", "m"],
}


# Common stem-prefixes used to detect syllable onsets for st-/sp- → ʃt-/ʃp-
PREFIXES = (
    "ab",
    "an",
    "auf",
    "aus",
    "be",
    "ein",
    "ent",
    "er",
    "ge",
    "miss",
    "mit",
    "nach",
    "ver",
    "vor",
    "weg",
    "zer",
    "zu",
    "über",
    "unter",
)

# Loanword v-list (v → /v/ instead of default /f/)
LOANWORD_V_FRAGMENTS = (
    "vase",
    "virus",
    "violin",
    "vital",
    "vakuum",
    "vegan",
    "vegetar",
    "vibri",
    "video",
    "visit",
    "vulkan",
    "vulgar",
    "vulkan",
    "veterin",
    "vampir",
    "vanille",
    "vatikan",
    "venedig",
    "vers",
    "veto",
)

# Vowels used in context-sensitive lookups
_BACK_VOWELS = "aou"
_FRONT_VOWELS = "eiäöü"
_ALL_VOWELS = "aeiouäöüy"


# ----------------------------------------------------------------------
# Context-sensitive callbacks. Each takes (word, i) where i is the index
# of the matched pattern's first char, and returns a list[str] of phonemes.
# ----------------------------------------------------------------------
def ch_rule(word: str, i: int) -> list[str]:
    if i == 0:
        # word-initial: most loanwords -> /k/. (China, Chemie are in EXCEPTION_DICT.)
        return ["k"]
    # check au- diphthong first (au+ch → ach-laut)
    if i >= 2 and word[i - 2 : i] == "au":
        return ["x"]
    prev = word[i - 1]
    if prev in _BACK_VOWELS:
        return ["x"]
    return ["ç"]


def chs_rule(word: str, i: int) -> list[str]:
    # ks (Wachs, Fuchs) when chs = morpheme-ending kluster.
    # Heuristic: trust /ks/ at end of word or before 't', else fall back to ch+s.
    if i + 3 == len(word) or (i + 3 < len(word) and word[i + 3] == "t"):
        return ["k", "s"]
    return ch_rule(word, i) + ["s"]


def st_rule(word: str, i: int) -> list[str]:
    if i == 0:
        return ["ʃ", "t"]
    for p in PREFIXES:
        if word.startswith(p) and i == len(p):
            return ["ʃ", "t"]
    return ["s", "t"]


def sp_rule(word: str, i: int) -> list[str]:
    if i == 0:
        return ["ʃ", "p"]
    for p in PREFIXES:
        if word.startswith(p) and i == len(p):
            return ["ʃ", "p"]
    return ["s", "p"]


def r_rule(word: str, i: int) -> list[str]:
    # word-final r preceded by a vowel → vocalic ɐ
    if i == len(word) - 1 and i > 0 and word[i - 1] in _ALL_VOWELS:
        return ["ɐ"]
    # -er / -ern / -ers at word end → vocalic
    if word.endswith("er") and i == len(word) - 1:
        return ["ɐ"]
    return ["ʁ"]


def s_rule(word: str, i: int) -> list[str]:
    # voiced /z/ at word start before a vowel, or between vowels
    if i == 0 and i + 1 < len(word) and word[i + 1] in _ALL_VOWELS:
        return ["z"]
    if 0 < i < len(word) - 1 and word[i - 1] in _ALL_VOWELS and word[i + 1] in _ALL_VOWELS:
        return ["z"]
    return ["s"]


def v_rule(word: str, i: int) -> list[str]:
    if any(frag in word for frag in LOANWORD_V_FRAGMENTS):
        return ["v"]
    return ["f"]


# Map of name → callable for JSON serialization
_CALLBACKS = {
    "ch": ch_rule,
    "chs": chs_rule,
    "st": st_rule,
    "sp": sp_rule,
    "r": r_rule,
    "s": s_rule,
    "v": v_rule,
}


# ----------------------------------------------------------------------
# Rule table. Order MUST be longest-prefix-first within each length class
# because the scanner tries entries in order. Output is either:
#   - list[str]  : literal phoneme tokens to emit
#   - str        : name of a callback in _CALLBACKS
# ----------------------------------------------------------------------
RULES: list[tuple[str, list[str] | str]] = [
    # 4 chars
    ("tsch", ["t", "ʃ"]),
    ("dsch", ["d", "ʒ"]),
    # 3 chars
    ("sch", ["ʃ"]),
    ("chs", "chs"),  # context-sensitive
    ("ung", ["ʊ", "ŋ"]),  # -ung suffix common
    # 2 chars — diphthongs and digraphs
    ("ei", ["a", "ɪ"]),
    ("ai", ["a", "ɪ"]),
    ("ay", ["a", "ɪ"]),
    ("ey", ["a", "ɪ"]),
    ("au", ["a", "ʊ"]),
    ("eu", ["ɔ", "ʏ"]),
    ("äu", ["ɔ", "ʏ"]),
    ("ie", ["i", "ː"]),
    # 2 chars — silent-h marks long vowel (must come before bare vowels)
    ("ah", ["a", "ː"]),
    ("eh", ["e", "ː"]),
    ("ih", ["i", "ː"]),
    ("oh", ["o", "ː"]),
    ("uh", ["u", "ː"]),
    ("äh", ["ɛ", "ː"]),
    ("öh", ["ø", "ː"]),
    ("üh", ["yː"]),
    # 2 chars — doubled vowels (long)
    ("aa", ["a", "ː"]),
    ("ee", ["e", "ː"]),
    ("oo", ["o", "ː"]),
    # 2 chars — consonant clusters
    ("ch", "ch"),  # context-sensitive
    ("ck", ["k"]),
    ("ng", ["ŋ"]),
    ("nk", ["ŋ", "k"]),
    ("pf", ["p", "f"]),
    ("ph", ["f"]),
    ("qu", ["k", "v"]),
    ("st", "st"),  # context-sensitive
    ("sp", "sp"),  # context-sensitive
    ("tz", ["ts"]),
    ("ts", ["ts"]),
    # doubled consonants → single (preceding vowel is short by orthography)
    ("bb", ["b"]),
    ("dd", ["d"]),
    ("ff", ["f"]),
    ("gg", ["g"]),
    ("ll", ["l"]),
    ("mm", ["m"]),
    ("nn", ["n"]),
    ("pp", ["p"]),
    ("rr", ["ʁ"]),
    ("ss", ["s"]),
    ("tt", ["t"]),
    # single chars
    ("ä", ["ɛ"]),
    ("ö", ["œ"]),
    ("ü", ["ʏ"]),
    ("ß", ["s"]),
    ("a", ["a"]),
    ("b", ["b"]),
    ("c", ["k"]),
    ("d", ["d"]),
    ("e", ["e"]),
    ("f", ["f"]),
    ("g", ["g"]),
    ("h", ["h"]),
    ("i", ["i"]),
    ("j", ["j"]),
    ("k", ["k"]),
    ("l", ["l"]),
    ("m", ["m"]),
    ("n", ["n"]),
    ("o", ["o"]),
    ("p", ["p"]),
    ("q", ["k"]),
    ("r", "r"),  # context-sensitive
    ("s", "s"),  # context-sensitive
    ("t", ["t"]),
    ("u", ["u"]),
    ("v", "v"),  # context-sensitive
    ("w", ["v"]),
    ("x", ["k", "s"]),
    ("y", ["yː"]),
    ("z", ["ts"]),
]


def _apply_rules(word: str) -> list[str]:
    """Greedy longest-match scanner over RULES."""
    word = word.lower()
    out: list[str] = []
    i = 0
    n = len(word)
    while i < n:
        matched = False
        for pattern, action in RULES:
            plen = len(pattern)
            if i + plen <= n and word[i : i + plen] == pattern:
                if isinstance(action, str):
                    out.extend(_CALLBACKS[action](word, i))
                else:
                    out.extend(action)
                i += plen
                matched = True
                break
        if not matched:
            # Unknown char — drop. (Could emit "UNK" but that pollutes audio.)
            i += 1
    return out


def _map_phoneme(ph: str) -> str:
    """Replicate english.map_phoneme: keep if in symbol table, else UNK."""
    rep_map = {"\n": ".", "...": "…", "v": "V"}
    if ph in rep_map:
        ph = rep_map[ph]
    if ph in _SYMBOLS:
        return ph
    return "UNK"


def normalize_text(text: str) -> str:
    text = expand_initialisms(text, "DE")
    text = expand_abbreviations(text)
    text = normalize_numbers(text)
    return text


def grapheme_to_phoneme(text: str, pad_start_end: bool = True):
    """Returns (phones, tones, word2ph) — same shape as english.grapheme_to_phoneme."""
    text = normalize_text(text)
    # Tokenize on whitespace; punctuation is split off and emitted as its own token.
    raw_words = text.split()
    phones: list[str] = []
    tones: list[int] = []
    word2ph: list[int] = []

    for raw in raw_words:
        # split leading and trailing punctuation
        lead, core, trail = _split_punct(raw)

        for ch in lead:
            mapped = _map_phoneme(ch)
            phones.append(mapped)
            tones.append(0)
            word2ph.append(1)

        if core:
            lower = core.lower()
            if lower in EXCEPTION_DICT:
                ph_list = list(EXCEPTION_DICT[lower])
            else:
                ph_list = _apply_rules(lower)
            ph_list = [_map_phoneme(p) for p in ph_list]
            phones.extend(ph_list)
            tones.extend([0] * len(ph_list))
            word2ph.append(len(ph_list))

        for ch in trail:
            mapped = _map_phoneme(ch)
            phones.append(mapped)
            tones.append(0)
            word2ph.append(1)

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


# Characters that can appear inside German words. Anything else is punctuation.
_WORD_CHARS = set("abcdefghijklmnopqrstuvwxyzäöüß0123456789'-")


def _split_punct(token: str) -> tuple[str, str, str]:
    """Split a whitespace-delimited token into (leading_punct, core, trailing_punct)."""
    lower = token.lower()
    n = len(lower)
    a = 0
    while a < n and lower[a] not in _WORD_CHARS:
        a += 1
    b = n
    while b > a and lower[b - 1] not in _WORD_CHARS:
        b -= 1
    return token[:a], token[a:b], token[b:]
