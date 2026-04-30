"""End-to-end ONNX inference (CPU). Skipped unless an ONNX export exists.

Run `python -m dittli_tts.inference.export --checkpoint checkpoints/G.pth
--out onnx/dittli.onnx --lang EN` first to produce one. The directory
shape (4 separate .onnx files) is what OnnxDittliTTS expects.
"""
from __future__ import annotations

from pathlib import Path

import pytest

EXPECTED = ["text_encoder.onnx", "duration_predictor.onnx", "flow.onnx", "decoder.onnx"]


@pytest.fixture(scope="module")
def onnx_dir(repo_root: Path) -> Path:
    d = repo_root / "onnx"
    if not d.exists() or not all((d / f).exists() for f in EXPECTED):
        pytest.skip(f"missing ONNX export under {d} (need: {', '.join(EXPECTED)})")
    pytest.importorskip("onnxruntime")
    return d


@pytest.mark.onnx
def test_onnx_synthesize(onnx_dir: Path, out_wav: Path):
    import numpy as np
    import soundfile as sf

    from dittli_tts.inference.onnx import OnnxDittliTTS

    tts = OnnxDittliTTS(onnx_dir=str(onnx_dir))
    tts.speak("Hello world.", output_path=str(out_wav))

    audio, sr = sf.read(out_wav, dtype="float32")
    assert sr == 44100
    assert audio.ndim == 1
    assert np.isfinite(audio).all()
    assert audio.shape[0] > 0
