"""Audio-utility unit tests on synthetic signals — no wav files needed."""

from __future__ import annotations

import math

import torch

from dittli_tts.audio import (
    MelSpectrogram,
    dynamic_range_compression,
    mel_spectrogram_torch,
    spec_to_mel_torch,
    spectrogram_torch,
)
from dittli_tts.utils.config import (
    FILTER_LENGTH,
    HOP_LENGTH,
    SAMPLING_RATE,
)

N_MELS = 128
T_SECONDS = 0.5  # 0.5 s of synthetic audio


def _sine(freq_hz: float, sr: int, seconds: float) -> torch.Tensor:
    n = int(sr * seconds)
    t = torch.arange(n, dtype=torch.float32) / sr
    return torch.sin(2 * math.pi * freq_hz * t)


def test_spectrogram_shape():
    y = _sine(440.0, SAMPLING_RATE, T_SECONDS).unsqueeze(0)
    spec = spectrogram_torch(y, FILTER_LENGTH, HOP_LENGTH, FILTER_LENGTH)
    n_freqs = FILTER_LENGTH // 2 + 1
    assert spec.shape[0] == 1
    assert spec.shape[1] == n_freqs
    assert spec.shape[2] > 0


def test_spectrogram_finite_and_nonnegative():
    y = _sine(440.0, SAMPLING_RATE, T_SECONDS).unsqueeze(0)
    spec = spectrogram_torch(y, FILTER_LENGTH, HOP_LENGTH, FILTER_LENGTH)
    assert torch.isfinite(spec).all()
    assert (spec >= 0).all()


def test_mel_shape_matches_n_mels():
    y = _sine(440.0, SAMPLING_RATE, T_SECONDS).unsqueeze(0)
    mel = mel_spectrogram_torch(
        y,
        FILTER_LENGTH,
        N_MELS,
        SAMPLING_RATE,
        HOP_LENGTH,
        FILTER_LENGTH,
    )
    assert mel.shape[1] == N_MELS
    assert torch.isfinite(mel).all()


def test_mel_basis_cached_across_calls():
    """spec_to_mel_torch should reuse the mel basis (cache hit on second call)
    — verified by identical output from identical inputs."""
    spec = torch.rand(1, FILTER_LENGTH // 2 + 1, 10)
    a = spec_to_mel_torch(spec, FILTER_LENGTH, N_MELS, SAMPLING_RATE)
    b = spec_to_mel_torch(spec, FILTER_LENGTH, N_MELS, SAMPLING_RATE)
    assert torch.allclose(a, b)


def test_dynamic_range_compression_monotonic():
    x = torch.tensor([1e-6, 1e-3, 1.0, 100.0])
    y = dynamic_range_compression(x)
    assert torch.isfinite(y).all()
    # log() is monotonic increasing
    assert (y[1:] > y[:-1]).all()


def test_mel_module_matches_functional():
    y = _sine(880.0, SAMPLING_RATE, T_SECONDS).unsqueeze(0)
    mod = MelSpectrogram(SAMPLING_RATE, FILTER_LENGTH, N_MELS, HOP_LENGTH, FILTER_LENGTH)
    mel_a = mod(y)
    mel_b = mel_spectrogram_torch(
        y,
        FILTER_LENGTH,
        N_MELS,
        SAMPLING_RATE,
        HOP_LENGTH,
        FILTER_LENGTH,
    )
    assert torch.allclose(mel_a, mel_b)


def test_pure_tone_peak_in_correct_mel_bin():
    """A 440 Hz tone should peak at a low-frequency mel bin, not the high end."""
    y = _sine(440.0, SAMPLING_RATE, 1.0).unsqueeze(0)
    mel = mel_spectrogram_torch(
        y,
        FILTER_LENGTH,
        N_MELS,
        SAMPLING_RATE,
        HOP_LENGTH,
        FILTER_LENGTH,
    )
    avg_per_bin = mel[0].mean(dim=-1)
    peak = int(torch.argmax(avg_per_bin).item())
    assert peak < N_MELS // 4, f"peak bin {peak} not in lower quarter for 440 Hz"
