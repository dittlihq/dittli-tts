"""German number-to-words for 0..999_999_999.

German has its quirks:
- 21 = einundzwanzig (one-and-twenty)
- "ein" before "und" (eins drops the s when joined)
- 100 = (ein)hundert; 1000 = (ein)tausend
- Long scale: 1_000_000 = eine Million, 1_000_000_000 = eine Milliarde
"""

import re

_ONES = [
    "null",
    "eins",
    "zwei",
    "drei",
    "vier",
    "fünf",
    "sechs",
    "sieben",
    "acht",
    "neun",
]
_TEENS = {
    10: "zehn",
    11: "elf",
    12: "zwölf",
    13: "dreizehn",
    14: "vierzehn",
    15: "fünfzehn",
    16: "sechzehn",
    17: "siebzehn",
    18: "achtzehn",
    19: "neunzehn",
}
_TENS = {
    20: "zwanzig",
    30: "dreißig",
    40: "vierzig",
    50: "fünfzig",
    60: "sechzig",
    70: "siebzig",
    80: "achtzig",
    90: "neunzig",
}


def _under_hundred(n: int) -> str:
    if n < 10:
        return _ONES[n]
    if n in _TEENS:
        return _TEENS[n]
    tens = (n // 10) * 10
    ones = n % 10
    if ones == 0:
        return _TENS[tens]
    ones_word = "ein" if ones == 1 else _ONES[ones]
    return f"{ones_word}und{_TENS[tens]}"


def _under_thousand(n: int) -> str:
    if n < 100:
        return _under_hundred(n)
    hundreds = n // 100
    rest = n % 100
    head = "ein" if hundreds == 1 else _ONES[hundreds]
    if rest == 0:
        return f"{head}hundert"
    return f"{head}hundert{_under_hundred(rest)}"


def _under_million(n: int) -> str:
    if n < 1000:
        return _under_thousand(n)
    thousands = n // 1000
    rest = n % 1000
    head = "ein" if thousands == 1 else _under_thousand(thousands)
    if rest == 0:
        return f"{head}tausend"
    return f"{head}tausend{_under_thousand(rest)}"


def number_to_words(n: int) -> str:
    if n < 0:
        return f"minus {number_to_words(-n)}"
    if n < 1_000_000:
        return _under_million(n)
    if n < 1_000_000_000:
        millions = n // 1_000_000
        rest = n % 1_000_000
        m_word = "eine Million" if millions == 1 else f"{_under_million(millions)} Millionen"
        if rest == 0:
            return m_word
        return f"{m_word} {_under_million(rest)}"
    billions = n // 1_000_000_000
    rest = n % 1_000_000_000
    b_word = "eine Milliarde" if billions == 1 else f"{_under_million(billions)} Milliarden"
    if rest == 0:
        return b_word
    return f"{b_word} {_under_million(rest)}"


_INT_RE = re.compile(r"-?\d+")


def normalize_numbers(text: str) -> str:
    """Replace standalone integers with their German word form."""
    return _INT_RE.sub(lambda m: number_to_words(int(m.group())), text)
