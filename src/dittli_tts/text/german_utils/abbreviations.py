"""German abbreviation expansion. Order matters — longer patterns first."""

import re

# (pattern, expansion) — pattern is a regex that must match the abbreviation
# in context. Word boundaries handled via lookarounds.
_ABBREVIATIONS = [
    (r"\bz\.\s?B\.", "zum Beispiel"),
    (r"\bd\.\s?h\.", "das heißt"),
    (r"\bu\.\s?a\.", "unter anderem"),
    (r"\bu\.\s?ä\.", "und ähnliches"),
    (r"\bs\.\s?o\.", "siehe oben"),
    (r"\bs\.\s?u\.", "siehe unten"),
    (r"\bbzw\.", "beziehungsweise"),
    (r"\busw\.", "und so weiter"),
    (r"\bevtl\.", "eventuell"),
    (r"\bggf\.", "gegebenenfalls"),
    (r"\bca\.", "circa"),
    (r"\betc\.", "et cetera"),
    (r"\bggü\.", "gegenüber"),
    (r"\bDr\.", "Doktor"),
    (r"\bProf\.", "Professor"),
    (r"\bDipl\.", "Diplom"),
    (r"\bHr\.", "Herr"),
    (r"\bFr\.", "Frau"),
    (r"\bNr\.", "Nummer"),
    (r"\bSt\.", "Sankt"),
    (r"\bStr\.", "Straße"),
    (r"\bMio\.", "Millionen"),
    (r"\bMrd\.", "Milliarden"),
    (r"\bTsd\.", "Tausend"),
    (r"\bAbb\.", "Abbildung"),
    (r"\bz\.\s?Z\.", "zur Zeit"),
    (r"\bn\.\s?Chr\.", "nach Christus"),
    (r"\bv\.\s?Chr\.", "vor Christus"),
]


def expand_abbreviations(text: str) -> str:
    for pattern, expansion in _ABBREVIATIONS:
        text = re.sub(pattern, expansion, text)
    return text
