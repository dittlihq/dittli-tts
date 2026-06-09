"""Train the English DittliTTS model from scratch.

Usage:
    python scripts/train_en.py \
        --metadata data/ljspeech/metadata.csv \
        --wavs-dir  data/ljspeech/wavs/ \
        --ckpt-dir  checkpoints_en/ \
        [--init-ckpt checkpoints/G.pth]   # optional warm-start

Omit --init-ckpt to train from random initialisation.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dittli_tts.training.trainer import Trainer, TrainerConfig
from dittli_tts.utils.train_config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LOG_INTERVAL,
    N_SPEAKERS_EN,
    SAVE_INTERVAL,
    TOTAL_STEPS,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--wavs-dir", required=True)
    p.add_argument("--ckpt-dir", default="checkpoints_en")
    p.add_argument("--init-ckpt", default=None, help="Optional checkpoint for warm-starting (e.g. checkpoints/G.pth).")
    p.add_argument("--max-steps", type=int, default=TOTAL_STEPS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    p.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    init_g = None
    if args.init_ckpt:
        if os.path.exists(args.init_ckpt):
            init_g = args.init_ckpt
            print(f"[train_en] warm-starting from {args.init_ckpt}")
        else:
            print(f"[train_en] WARN: --init-ckpt {args.init_ckpt} not found; training from scratch.")

    cfg = TrainerConfig(
        metadata_path=args.metadata,
        wavs_dir=args.wavs_dir,
        ckpt_dir=args.ckpt_dir,
        init_g_ckpt=init_g,
        n_speakers=N_SPEAKERS_EN,
        total_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        device=args.device,
        amp=not args.no_amp,
        language="EN",
    )
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
