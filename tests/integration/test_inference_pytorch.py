"""End-to-end PyTorch inference: load the committed English G.pth, synth a
short phrase, assert the wav output is finite audio of plausible length.

Heavy: loads ~20 MB checkpoint + bert tokenizer cache. The fixture is
session-scoped so the load cost is amortised across the suite.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def synthesize_fn():
    pytest.importorskip("g2p_en")
    pytest.importorskip("soundfile")
    import nltk

    for resource in ("taggers/averaged_perceptron_tagger_eng",
                     "taggers/averaged_perceptron_tagger",
                     "corpora/cmudict"):
        try:
            nltk.data.find(resource)
        except LookupError:
            pytest.skip(f"missing NLTK resource {resource!r}; "
                        f"run `python -c 'import nltk; nltk.download(\"all\")'`")
    from dittli_tts.inference.engine import synthesize
    return synthesize


def test_synthesize_english_writes_finite_wav(synthesize_fn, pytorch_engine, out_wav: Path):
    import numpy as np
    import soundfile as sf

    model, device = pytorch_engine
    text = "Hello world, this is a short test."
    synthesize_fn(text, str(out_wav), model, speaker="MALE", device=device, lang="EN")

    assert out_wav.exists(), "synthesize() did not write the output wav"
    audio, sr = sf.read(out_wav, dtype="float32")
    assert sr == 44100
    assert audio.ndim == 1
    assert np.isfinite(audio).all()
    # Reasonable length: 1.5..30 s for the phrase above
    assert 1.5 * sr < audio.shape[0] < 30 * sr


def test_synthesize_speed_changes_length(synthesize_fn, pytorch_engine, tmp_path: Path):
    """speed > 1 should produce shorter audio than speed < 1 for the same text."""
    import soundfile as sf

    model, device = pytorch_engine
    text = "Quick brown fox."

    fast = tmp_path / "fast.wav"
    slow = tmp_path / "slow.wav"
    synthesize_fn(text, str(fast), model, speaker="MALE", device=device, lang="EN", speed=1.5)
    synthesize_fn(text, str(slow), model, speaker="MALE", device=device, lang="EN", speed=0.7)

    fast_n = sf.read(fast, dtype="float32")[0].shape[0]
    slow_n = sf.read(slow, dtype="float32")[0].shape[0]
    assert fast_n < slow_n, f"speed=1.5 produced {fast_n} samples, speed=0.7 produced {slow_n}"
