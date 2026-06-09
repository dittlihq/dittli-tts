"""STFT + mel utilities for training and dataset preprocessing.

Uses torchaudio for spectrograms and mel filterbanks. Hyperparameters come
from dittli_tts.utils.config so the values match what the model was trained
with originally (44.1 kHz, n_fft=2048, hop=512, n_mels=128).
"""

from __future__ import annotations

import soundfile as sf
import torch
import torchaudio.functional as AF
from torch import nn


def load_audio(path: str, sr: int) -> torch.Tensor:
    """Load and (if needed) resample a wav to mono `sr`. Returns [T]."""
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # [C, T]
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if file_sr != sr:
        wav = AF.resample(wav, file_sr, sr)
    return wav.squeeze(0)


def dynamic_range_compression(x, C=1.0, eps=1e-5):
    return torch.log(torch.clamp(x, min=eps) * C)


_HANN_WINDOW_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _hann(win_size: int, device: torch.device) -> torch.Tensor:
    key = (win_size, str(device))
    if key not in _HANN_WINDOW_CACHE:
        _HANN_WINDOW_CACHE[key] = torch.hann_window(win_size).to(device)
    return _HANN_WINDOW_CACHE[key]


def spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
) -> torch.Tensor:
    """Linear magnitude spectrogram. `y` shape [B, T] or [T]."""
    if y.dim() == 1:
        y = y.unsqueeze(0)
    if y.dim() == 3:
        y = y.squeeze(1)
    pad = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=_hann(win_size, y.device),
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(spec.real**2 + spec.imag**2 + 1e-9)
    return spec


_MEL_BASIS_CACHE: dict[tuple, torch.Tensor] = {}


def _mel_basis(n_fft: int, n_mels: int, sr: int, fmin: float, fmax: float, device: torch.device) -> torch.Tensor:
    key = (n_fft, n_mels, sr, fmin, fmax, str(device))
    if key not in _MEL_BASIS_CACHE:
        mel = AF.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=float(fmin),
            f_max=float(fmax) if fmax else float(sr) / 2.0,
            n_mels=n_mels,
            sample_rate=sr,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)
        # torchaudio returns [n_freqs, n_mels]; we want [n_mels, n_freqs]
        _MEL_BASIS_CACHE[key] = mel.T.contiguous()
    return _MEL_BASIS_CACHE[key]


def spec_to_mel_torch(
    spec: torch.Tensor,
    n_fft: int,
    n_mels: int,
    sr: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> torch.Tensor:
    """Linear spec [B, n_freqs, T] → log-mel [B, n_mels, T]."""
    basis = _mel_basis(n_fft, n_mels, sr, fmin, fmax or sr / 2.0, spec.device)
    mel = torch.matmul(basis, spec)
    return dynamic_range_compression(mel)


def mel_spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    n_mels: int,
    sr: int,
    hop_size: int,
    win_size: int,
    fmin: float = 0.0,
    fmax: float | None = None,
    center: bool = False,
) -> torch.Tensor:
    spec = spectrogram_torch(y, n_fft, hop_size, win_size, center=center)
    return spec_to_mel_torch(spec, n_fft, n_mels, sr, fmin, fmax)


class MelSpectrogram(nn.Module):
    """Convenience module that wraps mel_spectrogram_torch."""

    def __init__(
        self,
        sr: int,
        n_fft: int,
        n_mels: int,
        hop_size: int,
        win_size: int,
        fmin: float = 0.0,
        fmax: float | None = None,
    ):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return mel_spectrogram_torch(
            y,
            self.n_fft,
            self.n_mels,
            self.sr,
            self.hop_size,
            self.win_size,
            self.fmin,
            self.fmax,
        )


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int) -> torch.Tensor:
    """Slice [B, ..., T] at start indices `ids_str` of length `segment_size`."""
    return commons_extract(x, ids_str, segment_size)


def commons_extract(x, ids_str, segment_size):
    """Match commons.extract_segments but accept any leading dims."""
    if x.dim() == 2:
        # treat as [B, T]
        ret = torch.zeros_like(x[:, :segment_size])
        for i in range(x.size(0)):
            idx = max(0, int(ids_str[i].item()))
            seg = x[i, idx : idx + segment_size]
            ret[i, : seg.size(0)] = seg
        return ret
    ret = torch.zeros_like(x[..., :segment_size])
    for i in range(x.size(0)):
        idx = max(0, int(ids_str[i].item()))
        seg = x[i, ..., idx : idx + segment_size]
        ret[i, ..., : seg.size(-1)] = seg
    return ret
