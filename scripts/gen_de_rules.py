"""Emit packages/tts-de/src/g2p_de_rules.json from the Python German G2P module.

The JS port (packages/tts-de/src/g2p_de.js) reads this JSON so it cannot
drift from the Python source of truth. Re-run after every edit to german.py /
abbreviations.py / number_norm.py.

The top-level `dittli_tts/__init__.py` eagerly imports torch, which we don't
want as a build-time dependency for emitting the JSON. We register an empty
package shim before importing the text submodule.
"""

from __future__ import annotations

import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _install_pkg_shim(name: str, dir_path: str) -> None:
    """Register `name` as a package whose __init__ body is a no-op."""
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [dir_path]
    sys.modules[name] = pkg


_install_pkg_shim("dittli_tts", os.path.join(ROOT, "src", "dittli_tts"))

from dittli_tts.text import german as g  # noqa: E402
from dittli_tts.text.german_utils.abbreviations import _ABBREVIATIONS  # noqa: E402


def serialize_rules() -> list:
    out = []
    for pattern, action in g.RULES:
        if isinstance(action, str):
            out.append([pattern, {"callback": action}])
        else:
            out.append([pattern, list(action)])
    return out


def main(out_path: str) -> None:
    payload = {
        "exceptions": {k: list(v) for k, v in g.EXCEPTION_DICT.items()},
        "rules": serialize_rules(),
        "prefixes": list(g.PREFIXES),
        "loanword_v_fragments": list(g.LOANWORD_V_FRAGMENTS),
        "abbreviations": [[p, e] for p, e in _ABBREVIATIONS],
        "back_vowels": g._BACK_VOWELS,
        "front_vowels": g._FRONT_VOWELS,
        "all_vowels": g._ALL_VOWELS,
        "word_chars": "".join(sorted(g._WORD_CHARS)),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(
        f"Wrote {out_path}: "
        f"{len(payload['rules'])} rules, "
        f"{len(payload['exceptions'])} exceptions, "
        f"{len(payload['abbreviations'])} abbreviations."
    )


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "packages", "tts-de", "src", "g2p_de_rules.json")
    main(out)
