"""Emit ONNX-model sidecar metadata JSON files for English and German.

Each sidecar carries the exact symbol list, language ID and tone offset the
model was trained against. The JS runtime reads this file (next to the
.onnx) so inference no longer relies on hard-coded constants.

Usage:
    python scripts/gen_metadata.py [out_dir]
"""
from __future__ import annotations

import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _install_pkg_shim(name: str, dir_path: str) -> None:
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [dir_path]
    sys.modules[name] = pkg


_install_pkg_shim("tiny_tts", os.path.join(ROOT, "tiny_tts"))

from tiny_tts.text.symbols import (
    language_id_map,
    language_tone_start_map,
)
from tiny_tts.text.symbols import (  # noqa: E402
    symbols as new_symbols,
)
from tiny_tts.utils.config import SAMPLING_RATE  # noqa: E402


def build(language: str, phoneme_set: str, spk2id: dict, symbols: list) -> dict:
    return {
        "language": language.lower(),
        "language_id": language_id_map[language.upper()],
        "tone_offset": language_tone_start_map[language.upper()],
        "sample_rate": SAMPLING_RATE,
        "symbols": list(symbols),
        "phoneme_set": phoneme_set,
        "n_speakers": len(spk2id),
        "spk2id": spk2id,
    }


def _load_old_en_symbols() -> list:
    path = os.path.join(ROOT, "checkpoints", "symbols_v1_en.txt")
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # The currently shipped English ONNX was trained against the 219-symbol
    # list, before German extended the union. The English sidecar must
    # reference that older list so phoneme IDs line up.
    en_symbols = _load_old_en_symbols()
    en = build("EN", "english_v1", {"MALE": 0}, en_symbols)

    # The German checkpoint will be trained against the new (extended) list,
    # which is what `from tiny_tts.text.symbols import symbols` returns.
    de = build("DE", "german_v1", {"THORSTEN": 0}, list(new_symbols))

    en_path = os.path.join(out_dir, "tinytts-en.json")
    de_path = os.path.join(out_dir, "tinytts-de.json")

    with open(en_path, "w", encoding="utf-8") as f:
        json.dump(en, f, ensure_ascii=False, indent=2)
    with open(de_path, "w", encoding="utf-8") as f:
        json.dump(de, f, ensure_ascii=False, indent=2)

    print(f"Wrote {en_path} (lang_id={en['language_id']}, tone_offset={en['tone_offset']}, "
          f"n_symbols={len(en['symbols'])})")
    print(f"Wrote {de_path} (lang_id={de['language_id']}, tone_offset={de['tone_offset']}, "
          f"n_symbols={len(de['symbols'])})")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models")
    main(out)
