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
sys.path.insert(0, os.path.join(ROOT, "src"))


def _install_pkg_shim(name: str, dir_path: str) -> None:
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [dir_path]
    sys.modules[name] = pkg


_install_pkg_shim("dittli_tts", os.path.join(ROOT, "src", "dittli_tts"))

from dittli_tts.text.symbols import (
    language_id_map,
    language_tone_start_map,
)
from dittli_tts.text.symbols import (  # noqa: E402
    symbols as new_symbols,
)
from dittli_tts.utils.config import SAMPLING_RATE  # noqa: E402


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


def _resolve_en_symbols(en_checkpoint: str | None = None) -> list:
    """Use the checkpoint's actual vocab size to pick the right symbol list."""
    if en_checkpoint:
        import torch
        ckpt = torch.load(en_checkpoint, map_location="cpu", weights_only=False)
        n_vocab = ckpt["model"]["enc_p.emb.weight"].shape[0]
        if n_vocab == len(list(new_symbols)):
            return list(new_symbols)
    snap = os.path.join(ROOT, "checkpoints", "symbols_v1_en.txt")
    if os.path.exists(snap):
        with open(snap, encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]
    return list(new_symbols)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", nargs="?", default=None, help="(unused, kept for compat)")
    parser.add_argument("--en-checkpoint", default=None,
                        help="Path to English G.pth to detect actual vocab size")
    args = parser.parse_args()

    targets = {
        "EN": (
            os.path.join(ROOT, "packages", "tts-en", "assets", "en", "metadata.json"),
            "english_v1",
            {"MALE": 0},
            _resolve_en_symbols(args.en_checkpoint),
        ),
        "DE": (
            os.path.join(ROOT, "packages", "tts-de", "assets", "de", "metadata.json"),
            "german_v1",
            {"THORSTEN": 0},
            list(new_symbols),
        ),
    }

    for lang, (out_path, phoneme_set, spk2id, symbols) in targets.items():
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        meta = build(lang, phoneme_set, spk2id, symbols)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Wrote {out_path} (lang_id={meta['language_id']}, "
              f"tone_offset={meta['tone_offset']}, n_symbols={len(meta['symbols'])})")


if __name__ == "__main__":
    main()
