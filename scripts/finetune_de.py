"""Fine-tune the English DittliTTS checkpoint on Thorsten Voice (German).

Steps:
1. Load the existing English G.pth.
2. Load the snapshot of the *English-era* symbol list
   (`checkpoints/symbols_v1_en.txt`) and remap the phoneme embedding rows so
   that overlapping symbols stay aligned with the new (German-extended)
   symbol table.
3. Save the remapped checkpoint to a temporary file and hand it to
   `Trainer.run` as `init_g_ckpt`.

Usage:
    python scripts/finetune_de.py \
        --metadata /path/to/thorsten/metadata.csv \
        --wavs-dir /path/to/thorsten/wavs/ \
        --english-ckpt checkpoints/G.pth \
        --ckpt-dir checkpoints_de/ \
        [--max-steps 100]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dittli_tts.text.symbols import symbols as new_symbols
from dittli_tts.training.trainer import Trainer, TrainerConfig
from dittli_tts.utils.remap_checkpoint import (
    load_old_symbols,
    remap_state_dict,
)
from dittli_tts.utils.train_config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LOG_INTERVAL,
    N_SPEAKERS_DE,
    SAVE_INTERVAL,
    TOTAL_STEPS,
)


def remap_and_save(english_ckpt: str, old_symbols_file: str, out_path: str) -> str:
    """Produce a checkpoint compatible with the German-extended symbol table."""
    old_symbols = load_old_symbols(old_symbols_file)
    if len(old_symbols) == len(new_symbols):
        # Symbol list is unchanged — no remapping needed.
        return english_ckpt

    ckpt = torch.load(english_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    state = remap_state_dict(state, old_symbols, list(new_symbols))
    out = {"model": state, "step": 0}
    torch.save(out, out_path)
    print(f"[finetune] wrote remapped checkpoint to {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--wavs-dir", required=True)
    p.add_argument("--english-ckpt", default="checkpoints/G.pth")
    p.add_argument("--old-symbols", default=os.path.join(ROOT, "checkpoints", "symbols_v1_en.txt"))
    p.add_argument("--ckpt-dir", default="checkpoints_de")
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
    if args.english_ckpt and os.path.exists(args.english_ckpt):
        if not os.path.exists(args.old_symbols):
            print(
                f"[finetune] WARN: {args.old_symbols} missing — fine-tuning "
                f"with symbol mismatch will skip the embedding row."
            )
            init_g = args.english_ckpt
        else:
            tmp_ckpt = os.path.join(args.ckpt_dir, "_g_remapped.pth")
            init_g = remap_and_save(args.english_ckpt, args.old_symbols, tmp_ckpt)
    else:
        print(f"[finetune] no English checkpoint at {args.english_ckpt} — training from scratch.")

    cfg = TrainerConfig(
        metadata_path=args.metadata,
        wavs_dir=args.wavs_dir,
        ckpt_dir=args.ckpt_dir,
        init_g_ckpt=init_g,
        n_speakers=N_SPEAKERS_DE,
        total_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        device=args.device,
        amp=not args.no_amp,
    )
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
