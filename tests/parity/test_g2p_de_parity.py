"""Cross-check Python and JS German G2P over the curated word list.

Drift between src/dittli_tts/text/german.py and src/node/g2p_de.js is
the most common silent training/inference bug in this repo, so this
runs on every `pytest` invocation (no opt-in marker).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from dittli_tts.text import german as g

REPO_ROOT = Path(__file__).resolve().parents[2]
WORDLIST = REPO_ROOT / "scripts" / "de_test_words.txt"
JS_RUNNER = REPO_ROOT / "scripts" / "_run_js_g2p.js"


def python_phones(word: str) -> list[str]:
    phones, _, _ = g.grapheme_to_phoneme(word, pad_start_end=False)
    return phones


@pytest.fixture(scope="module")
def words() -> list[str]:
    if not WORDLIST.exists():
        pytest.skip(f"missing {WORDLIST}")
    return [line.strip() for line in WORDLIST.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def js_results(node_bin: str, words: list[str]) -> dict[str, list[str]]:
    if not JS_RUNNER.exists():
        pytest.skip(f"missing {JS_RUNNER}")
    proc = subprocess.run(
        [node_bin, str(JS_RUNNER)],
        input=json.dumps(words),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)


@pytest.mark.node
def test_de_g2p_python_matches_js(words: list[str], js_results: dict[str, list[str]]):
    """Every word's Python and JS phoneme sequences must match exactly."""
    mismatches: list[tuple[str, list[str], list[str]]] = []
    for w in words:
        py = python_phones(w)
        js = js_results.get(w, [])
        if py != js:
            mismatches.append((w, py, js))

    if mismatches:
        msg_lines = [f"{len(mismatches)}/{len(words)} words mismatch:"]
        for w, py, js in mismatches[:10]:
            msg_lines.append(f"  {w!r}\n    py: {py}\n    js: {js}")
        pytest.fail("\n".join(msg_lines))
