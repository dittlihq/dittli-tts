"""Dataset loader for German TTS fine-tuning on Thorsten Voice.

Expects a metadata file with `filename|transcript` pairs (Thorsten format)
and a folder of wavs alongside it. Pre-computes spectrogram and phoneme IDs
on disk so training is I/O bound on the spec tensor only.
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset

from dittli_tts.audio import load_audio, spectrogram_torch
from dittli_tts.nn import commons
from dittli_tts.text import phonemes_to_ids
from dittli_tts.text.english import grapheme_to_phoneme as en_g2p
from dittli_tts.text.english import normalize_text as en_normalize
from dittli_tts.text.german import grapheme_to_phoneme as de_g2p
from dittli_tts.utils.config import (
    ADD_BLANK,
    FILTER_LENGTH,
    HOP_LENGTH,
    SAMPLING_RATE,
)


def _read_metadata(path: str) -> list[tuple[str, str]]:
    """Read a metadata.csv with `filename|transcript[|...]` lines."""
    rows: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            rows.append((parts[0], parts[1]))
    return rows


def _spec_path(wav_path: str) -> str:
    return wav_path + ".spec.pt"


def _phones_path(wav_path: str, language: str = "DE") -> str:
    return wav_path + f".{language.lower()}.phones.pt"


def compute_and_cache(
    wav_path: str,
    text: str,
    sr: int = SAMPLING_RATE,
    n_fft: int = FILTER_LENGTH,
    hop: int = HOP_LENGTH,
    language: str = "DE",
) -> None:
    """Compute spec + phoneme IDs for a single utterance and cache to disk."""
    wav = load_audio(wav_path, sr).unsqueeze(0)  # [1, T]
    spec = spectrogram_torch(wav, n_fft, hop, n_fft, center=False).squeeze(0)
    torch.save(spec.cpu(), _spec_path(wav_path))

    if language == "EN":
        phones, tones, _ = en_g2p(en_normalize(text))
    else:
        phones, tones, _ = de_g2p(text)
    phone_ids, tone_ids, lang_ids = phonemes_to_ids(phones, tones, language)
    if ADD_BLANK:
        phone_ids = commons.insert_blanks(phone_ids, 0)
        tone_ids = commons.insert_blanks(tone_ids, 0)
        lang_ids = commons.insert_blanks(lang_ids, 0)
    torch.save(
        {
            "phone_ids": torch.LongTensor(phone_ids),
            "tone_ids": torch.LongTensor(tone_ids),
            "lang_ids": torch.LongTensor(lang_ids),
        },
        _phones_path(wav_path, language),
    )


class ThorstenDataset(Dataset):
    """Single-speaker German dataset.

    Args:
        metadata_path: path to metadata.csv (`filename|transcript`).
        wavs_dir:      directory containing the wav files referenced.
        sr:            target sample rate.
        speaker_id:    integer ID for the (single) speaker.
        bert_dim, ja_bert_dim: BERT feature dims (zero-filled for German).
        max_spec_len:  filter out utterances longer than this many spec frames
                       (helps avoid OOM on long captions).
        require_cache: if True, error when precomputed cache files are missing
                       (useful in training to fail loud); if False, fall back
                       to recomputing on the fly.
    """

    def __init__(
        self,
        metadata_path: str,
        wavs_dir: str,
        sr: int = SAMPLING_RATE,
        n_fft: int = FILTER_LENGTH,
        hop: int = HOP_LENGTH,
        speaker_id: int = 0,
        bert_dim: int = 1024,
        ja_bert_dim: int = 768,
        max_spec_len: int = 1500,
        require_cache: bool = True,
        language: str = "DE",
    ):
        self.metadata_path = metadata_path
        self.wavs_dir = wavs_dir
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.speaker_id = speaker_id
        self.bert_dim = bert_dim
        self.ja_bert_dim = ja_bert_dim
        self.max_spec_len = max_spec_len
        self.require_cache = require_cache
        self.language = language

        rows = _read_metadata(metadata_path)
        self.items: list[tuple[str, str]] = []
        for filename, transcript in rows:
            wav_path = self._resolve(filename)
            if not os.path.exists(wav_path):
                continue
            if max_spec_len:
                sp = _spec_path(wav_path)
                if os.path.exists(sp):
                    spec = torch.load(sp, map_location="cpu", weights_only=True)
                    if spec.shape[1] > max_spec_len:
                        continue
            self.items.append((wav_path, transcript))

    def _resolve(self, filename: str) -> str:
        # Thorsten files may or may not include the .wav extension.
        if not filename.endswith(".wav"):
            filename = filename + ".wav"
        return os.path.join(self.wavs_dir, filename)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        wav_path, transcript = self.items[idx]

        spec_path = _spec_path(wav_path)
        ph_path = _phones_path(wav_path, self.language)
        if not os.path.exists(spec_path) or not os.path.exists(ph_path):
            if self.require_cache:
                raise FileNotFoundError(
                    f"Missing cached features for {wav_path}. Run dittli_tts.data.preprocess first."
                )
            compute_and_cache(wav_path, transcript, self.sr, self.n_fft, self.hop, self.language)

        spec = torch.load(spec_path, map_location="cpu", weights_only=True)
        ph = torch.load(ph_path, map_location="cpu", weights_only=True)
        wav = load_audio(wav_path, self.sr)

        return {
            "phone_ids": ph["phone_ids"],
            "tone_ids": ph["tone_ids"],
            "lang_ids": ph["lang_ids"],
            "spec": spec,
            "wav": wav,
            "sid": torch.LongTensor([self.speaker_id])[0],
        }


def collate(batch: list[dict], bert_dim: int = 1024, ja_bert_dim: int = 768):
    """Pad variable-length sequences. Returns a dict of batched tensors."""
    B = len(batch)

    x_lens = torch.LongTensor([b["phone_ids"].size(0) for b in batch])
    y_lens = torch.LongTensor([b["spec"].size(1) for b in batch])
    wav_lens = torch.LongTensor([b["wav"].size(0) for b in batch])

    max_x = int(x_lens.max())
    max_y = int(y_lens.max())
    max_w = int(wav_lens.max())
    n_freqs = batch[0]["spec"].size(0)

    x = torch.zeros(B, max_x, dtype=torch.long)
    tone = torch.zeros(B, max_x, dtype=torch.long)
    lang = torch.zeros(B, max_x, dtype=torch.long)
    spec = torch.zeros(B, n_freqs, max_y, dtype=batch[0]["spec"].dtype)
    wav = torch.zeros(B, 1, max_w, dtype=batch[0]["wav"].dtype)
    sid = torch.zeros(B, dtype=torch.long)
    bert = torch.zeros(B, bert_dim, max_x, dtype=torch.float)
    ja_bert = torch.zeros(B, ja_bert_dim, max_x, dtype=torch.float)

    for i, b in enumerate(batch):
        nx = b["phone_ids"].size(0)
        ny = b["spec"].size(1)
        nw = b["wav"].size(0)
        x[i, :nx] = b["phone_ids"]
        tone[i, :nx] = b["tone_ids"]
        lang[i, :nx] = b["lang_ids"]
        spec[i, :, :ny] = b["spec"]
        wav[i, 0, :nw] = b["wav"]
        sid[i] = b["sid"]

    return {
        "x": x,
        "x_lengths": x_lens,
        "tone": tone,
        "language": lang,
        "spec": spec,
        "spec_lengths": y_lens,
        "wav": wav,
        "wav_lengths": wav_lens,
        "sid": sid,
        "bert": bert,
        "ja_bert": ja_bert,
    }
