"""Smoke-test the Node CLI: spawn `node packages/tts-core/bin/cli.js`, assert
it writes a wav file. Skipped if Node, npm dependencies, or an ONNX model
are missing.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def cli_path(repo_root: Path) -> Path:
    p = repo_root / "packages" / "tts-core" / "bin" / "cli.js"
    assert p.exists(), f"missing CLI entry point at {p}"
    return p


@pytest.fixture(scope="module")
def en_onnx_model(repo_root: Path) -> Path:
    p = repo_root / "models" / "dittli-en.onnx"
    if not p.exists():
        pytest.skip(f"missing ONNX model at {p}; run dittli_tts.inference.export")
    return p


def _wav_duration(path: Path) -> float:
    """Read the RIFF header and compute duration without extra deps."""
    with open(path, "rb") as f:
        data = f.read()
    # Validate RIFF/WAVE header
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", "not a RIFF/WAVE file"
    sample_rate, _ = struct.unpack("<II", data[24:32])
    bits_per_sample = struct.unpack("<H", data[34:36])[0]
    # data chunk size — search for "data" subchunk
    idx = data.find(b"data")
    assert idx > 0
    data_size = struct.unpack("<I", data[idx + 4: idx + 8])[0]
    n_samples = data_size / (bits_per_sample // 8)
    return n_samples / sample_rate


@pytest.mark.node
def test_node_cli_writes_wav(
    repo_root: Path,
    node_bin: str,
    node_modules_installed: Path,
    cli_path: Path,
    en_onnx_model: Path,
    tmp_path: Path,
):
    from tests.conftest import run

    out = tmp_path / "node_out.wav"
    run([
        node_bin, str(cli_path),
        "Hello world.",
        "--model", str(en_onnx_model),
        "-o", str(out),
    ], cwd=repo_root)

    assert out.exists() and out.stat().st_size > 1024, "CLI did not produce a wav"
    duration = _wav_duration(out)
    assert 0.3 < duration < 30, f"unexpected wav duration: {duration:.2f} s"
