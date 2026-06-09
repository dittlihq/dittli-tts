"""Shared pytest fixtures.

Heavy fixtures (PyTorch model load, ONNX session) are session-scoped so the
~10-30 s checkpoint load is paid once for the whole run, not per test.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ── Paths ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def english_checkpoint(repo_root: Path) -> Path:
    """Path to the English G.pth committed at checkpoints/G.pth."""
    p = repo_root / "checkpoints" / "G.pth"
    if not p.exists():
        pytest.skip(f"missing {p}; restore from git or run setup_de_data.sh")
    return p


@pytest.fixture(scope="session")
def thorsten_metadata(repo_root: Path) -> Path:
    p = repo_root / "data" / "thorsten" / "metadata.csv"
    if not p.exists():
        pytest.skip("Thorsten dataset not downloaded; run scripts/setup_de_data.sh")
    return p


@pytest.fixture(scope="session")
def thorsten_wavs(repo_root: Path) -> Path:
    p = repo_root / "data" / "thorsten" / "wavs"
    if not p.exists():
        pytest.skip("Thorsten wavs not downloaded; run scripts/setup_de_data.sh")
    return p


# ── Tooling probes ──────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def node_bin() -> str:
    """Resolved absolute path to `node`, or skip the test if unavailable."""
    n = shutil.which("node")
    if not n:
        pytest.skip("node executable not on PATH")
    return n


@pytest.fixture(scope="session")
def node_modules_installed(repo_root: Path, node_bin: str) -> Path:
    """Skip if `npm install` hasn't been run (onnxruntime-node missing)."""
    nm = repo_root / "node_modules"
    if not (nm / "onnxruntime-node").exists():
        pytest.skip("npm dependencies not installed; run `npm install`")
    return nm


# ── Heavy session-scoped engine ─────────────────────────────────────────


@pytest.fixture(scope="session")
def pytorch_engine(english_checkpoint: Path):
    """Load the English G.pth once for all tests that need PyTorch inference."""
    import torch

    from dittli_tts.inference.engine import load_engine

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_engine(str(english_checkpoint), device=device), device


# ── Per-test scratch dir ────────────────────────────────────────────────


@pytest.fixture
def out_wav(tmp_path: Path) -> Path:
    return tmp_path / "out.wav"


# ── Subprocess helper ───────────────────────────────────────────────────


def run(cmd: list[str], cwd: Path | None = None, **kw) -> subprocess.CompletedProcess:
    """subprocess.run wrapper that captures + asserts success and surfaces stderr."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
        **kw,
    )
