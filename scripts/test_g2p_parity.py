"""Cross-check Python and JS German G2P over a word list.

Runs the Python implementation in-process, then shells out to Node to run
the JS port over the same words, and asserts that the phoneme sequences
match for every word.

Usage:
    python scripts/test_g2p_parity.py [path/to/wordlist.txt]
"""
from __future__ import annotations

import json
import os
import subprocess
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

from tiny_tts.text import german as g  # noqa: E402


def python_phones(word: str) -> list[str]:
    """Phones the Python G2P would emit for a single word, no padding."""
    phones, _, _ = g.grapheme_to_phoneme(word, pad_start_end=False)
    return phones


def run_js(words: list[str]) -> dict[str, list[str]]:
    """Spawn a one-shot Node process that prints phones per word as JSON."""
    runner = os.path.join(ROOT, "scripts", "_run_js_g2p.js")
    cmd = ["node", runner]
    proc = subprocess.run(
        cmd,
        input=json.dumps(words),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)


def main(wordlist_path: str) -> int:
    with open(wordlist_path, encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    print(f"Comparing G2P on {len(words)} words ...")

    py_results = {w: python_phones(w) for w in words}
    js_results = run_js(words)

    mismatches: list[tuple[str, list[str], list[str]]] = []
    for w in words:
        py = py_results[w]
        js = js_results.get(w, [])
        if py != js:
            mismatches.append((w, py, js))

    matched = len(words) - len(mismatches)
    print(f"Matched: {matched} / {len(words)}")
    if mismatches:
        print("First 20 mismatches:")
        for w, py, js in mismatches[:20]:
            print(f"  {w!r}")
            print(f"    py: {py}")
            print(f"    js: {js}")
        return 1
    print("PASS — all phoneme sequences match.")
    return 0


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "scripts", "de_test_words.txt"
    )
    sys.exit(main(path))
