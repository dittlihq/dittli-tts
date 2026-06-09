"""Contract test: export.py output → OnnxDittliTTS input round-trip.

Uses a random-weight model (no checkpoint needed) to verify that what
the export pipeline writes is exactly what OnnxDittliTTS can load.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def exported_onnx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a tiny random-weight model, export to ONNX, and write the sidecar."""
    import torch

    from dittli_tts.inference.export import _build_metadata, _OnnxWrapper, _resolve_symbols
    from dittli_tts.models.synthesizer import VoiceSynthesizer
    from dittli_tts.text.symbols import language_id_map, symbols
    from dittli_tts.utils.config import (
        MODEL_PARAMS,
        N_SPEAKERS,
        SEGMENT_FRAMES,
        SPEC_CHANNELS,
    )

    tmp = tmp_path_factory.mktemp("onnx_contract")
    onnx_path = tmp / "dittli.onnx"

    model = VoiceSynthesizer(
        len(symbols),
        SPEC_CHANNELS,
        SEGMENT_FRAMES,
        n_speakers=N_SPEAKERS,
        **MODEL_PARAMS,
    )
    model.eval()
    wrapper = _OnnxWrapper(model)
    wrapper.eval()

    seq_len = 8
    dummy = (
        torch.randint(0, 50, (1, seq_len), dtype=torch.long),
        torch.LongTensor([seq_len]),
        torch.LongTensor([0]),
        torch.randint(0, 5, (1, seq_len), dtype=torch.long),
        torch.full((1, seq_len), language_id_map["EN"], dtype=torch.long),
        torch.zeros((1, 1024, seq_len)),
        torch.zeros((1, 768, seq_len)),
        torch.FloatTensor([0.667]),
        torch.FloatTensor([0.8]),
        torch.FloatTensor([1.0]),
    )

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            dynamo=False,
            do_constant_folding=True,
            input_names=[
                "x",
                "x_lengths",
                "sid",
                "tone",
                "language",
                "bert",
                "ja_bert",
                "noise_scale",
                "noise_scale_w",
                "length_scale",
            ],
            output_names=["audio"],
            dynamic_axes={
                "x": {0: "batch_size", 1: "text_length"},
                "tone": {0: "batch_size", 1: "text_length"},
                "language": {0: "batch_size", 1: "text_length"},
                "bert": {0: "batch_size", 2: "text_length"},
                "ja_bert": {0: "batch_size", 2: "text_length"},
                "audio": {0: "batch_size", 2: "audio_length"},
            },
        )

    n_vocab = model.enc_p.emb.weight.shape[0]
    syms = _resolve_symbols("EN", n_vocab)
    meta = _build_metadata("EN", {"MALE": 0}, syms)
    onnx_path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")

    return onnx_path


@pytest.mark.onnx
def test_export_produces_loadable_onnx(exported_onnx: Path):
    """The exported file must be parseable by onnxruntime."""
    import onnxruntime as ort

    assert exported_onnx.exists()
    # Will raise if the graph is malformed
    sess = ort.InferenceSession(str(exported_onnx), providers=["CPUExecutionProvider"])
    input_names = {inp.name for inp in sess.get_inputs()}
    assert {
        "x",
        "x_lengths",
        "sid",
        "tone",
        "language",
        "bert",
        "ja_bert",
        "noise_scale",
        "noise_scale_w",
        "length_scale",
    } == input_names
    output_names = {out.name for out in sess.get_outputs()}
    assert output_names == {"audio"}


@pytest.mark.onnx
def test_sidecar_has_required_fields(exported_onnx: Path):
    """Sidecar written by export must satisfy every field OnnxDittliTTS (JS side) requires."""
    sidecar = exported_onnx.with_suffix(".json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    for field in ["language", "language_id", "tone_offset", "sample_rate", "symbols"]:
        assert field in meta, f"sidecar missing required field: {field!r}"
    assert isinstance(meta["symbols"], list) and len(meta["symbols"]) > 0


@pytest.mark.onnx
def test_onnx_inference_produces_finite_audio(exported_onnx: Path, out_wav: Path):
    """OnnxDittliTTS can load the export and synthesize finite audio."""
    pytest.importorskip("g2p_en")
    import numpy as np
    import soundfile as sf

    from dittli_tts.inference.onnx import OnnxDittliTTS

    tts = OnnxDittliTTS(onnx_path=str(exported_onnx))
    tts.speak("Hello world.", output_path=str(out_wav))

    audio, sr = sf.read(out_wav, dtype="float32")
    assert sr == 44100
    assert audio.ndim == 1
    assert np.isfinite(audio).all()
    assert audio.shape[0] > 0
