"""Pre-compute spectrograms and phoneme IDs for a Thorsten-Voice-style corpus.

Usage:
    python -m dittli_tts.data.preprocess \
        --metadata path/to/metadata.csv \
        --wavs-dir path/to/wavs/

The script writes `{wav}.spec.pt` and `{wav}.phones.pt` next to each wav.
Failures are surfaced with the offending file path so a few bad rows don't
silently corrupt the cache.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch

from dittli_tts.data.dataset import (
    _read_metadata,
    _spec_path,
    _phones_path,
    compute_and_cache,
)
from dittli_tts.utils.config import SAMPLING_RATE, FILTER_LENGTH, HOP_LENGTH


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--wavs-dir", required=True)
    p.add_argument("--sr", type=int, default=SAMPLING_RATE)
    p.add_argument("--n-fft", type=int, default=FILTER_LENGTH)
    p.add_argument("--hop", type=int, default=HOP_LENGTH)
    p.add_argument("--language", default="DE", choices=["DE", "EN"],
                   help="Language for G2P (default: DE).")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if cache files already exist.")
    args = p.parse_args()

    rows = _read_metadata(args.metadata)
    print(f"Found {len(rows)} rows in {args.metadata}.")

    ok, skipped, failed = 0, 0, 0
    for i, (filename, transcript) in enumerate(rows):
        if not filename.endswith(".wav"):
            filename = filename + ".wav"
        wav_path = os.path.join(args.wavs_dir, filename)
        if not os.path.exists(wav_path):
            print(f"[{i}] missing wav: {wav_path}", file=sys.stderr)
            failed += 1
            continue
        if (
            not args.force
            and os.path.exists(_spec_path(wav_path))
            and os.path.exists(_phones_path(wav_path))
        ):
            skipped += 1
            continue
        try:
            compute_and_cache(wav_path, transcript, args.sr, args.n_fft, args.hop, args.language)
            ok += 1
        except Exception as e:  # surface root cause and continue
            failed += 1
            print(f"[{i}] FAILED {wav_path}: {e}", file=sys.stderr)
            traceback.print_exc()
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} / {len(rows)} processed (ok={ok}, skip={skipped}, fail={failed})")

    print(f"Done. ok={ok}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
