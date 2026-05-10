"""End-to-end ONNX inference (CPU). Skipped unless an ONNX export exists.

Run `python -m dittli_tts.inference.export --checkpoint checkpoints/G.pth
--out models/dittli.onnx --lang EN` first to produce one.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def onnx_path(repo_root: Path) -> Path:
    p = repo_root / "models" / "dittli.onnx"
    if not p.exists():
        pytest.skip(f"missing ONNX export at {p}")
    pytest.importorskip("onnxruntime")
    return p


@pytest.mark.onnx
def test_onnx_synthesize(onnx_path: Path, out_wav: Path):
    import numpy as np
    import soundfile as sf

    from dittli_tts.inference.onnx import OnnxDittliTTS

    tts = OnnxDittliTTS(onnx_path=str(onnx_path))
    tts.speak("Hello world.", output_path=str(out_wav))

    audio, sr = sf.read(out_wav, dtype="float32")
    assert sr == 44100
    assert audio.ndim == 1
    assert np.isfinite(audio).all()
    assert audio.shape[0] > 0
