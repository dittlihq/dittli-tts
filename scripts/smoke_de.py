"""End-to-end smoke test of the German training pipeline.

Runs on CPU. Reads metadata, builds the dataset, fetches a tiny batch via
the collator, and runs ONE forward pass through `VoiceSynthesizer.forward()`
plus a discriminator step. Verifies losses are finite. Does NOT save a
checkpoint.

Useful for catching shape / device / dtype mistakes before paying for GPU
time. Takes ~30 s on a laptop.

Usage:
    python scripts/smoke_de.py \
        --metadata data/thorsten/metadata.csv \
        --wavs-dir data/thorsten/wavs
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dittli_tts.audio import (
    commons_extract,
    mel_spectrogram_torch,
    spec_to_mel_torch,
)
from dittli_tts.data.dataset import ThorstenDataset, collate, compute_and_cache
from dittli_tts.losses import (
    discriminator_loss,
    feature_matching_loss,
    generator_loss,
    kl_loss,
    mel_loss,
)
from dittli_tts.models.discriminator import MultiPeriodDiscriminator
from dittli_tts.models.synthesizer import VoiceSynthesizer
from dittli_tts.text.symbols import symbols
from dittli_tts.utils.config import (
    FILTER_LENGTH,
    HOP_LENGTH,
    MODEL_PARAMS,
    SAMPLING_RATE,
    SEGMENT_FRAMES,
    SPEC_CHANNELS,
)
from dittli_tts.utils.train_config import (
    C_DUR,
    C_KL,
    C_MEL,
    F_MAX,
    F_MIN,
    N_MELS,
)


def _ensure_cache(dataset: ThorstenDataset, n: int) -> None:
    """Make sure the first `n` rows have their .spec.pt and .phones.pt
    artifacts on disk. Safe to call repeatedly."""
    items = dataset.items[:n]
    for wav_path, transcript in items:
        spec_path = wav_path + ".spec.pt"
        ph_path = wav_path + ".phones.pt"
        if not (os.path.exists(spec_path) and os.path.exists(ph_path)):
            print(f"[smoke] precomputing features for {os.path.basename(wav_path)}")
            compute_and_cache(wav_path, transcript)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--wavs-dir", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[smoke] device={device}")

    # 1) Dataset + collate
    print("[smoke] building dataset ...")
    dataset = ThorstenDataset(args.metadata, args.wavs_dir, require_cache=False)
    if len(dataset) == 0:
        sys.exit(f"[smoke] no rows in {args.metadata}; check the path")
    print(f"[smoke] dataset size: {len(dataset)} utterances")

    _ensure_cache(dataset, args.batch_size)
    batch = collate([dataset[i] for i in range(args.batch_size)])
    print(f"[smoke] batched shapes — "
          f"x={tuple(batch['x'].shape)}, spec={tuple(batch['spec'].shape)}, "
          f"wav={tuple(batch['wav'].shape)}")

    # Move to device
    for k in ("x", "x_lengths", "tone", "language", "spec", "spec_lengths",
              "wav", "wav_lengths", "sid", "bert", "ja_bert"):
        batch[k] = batch[k].to(device)

    # 2) Build models
    print("[smoke] building generator + discriminator ...")
    net_g = VoiceSynthesizer(
        len(symbols), SPEC_CHANNELS, SEGMENT_FRAMES,
        n_speakers=1, **MODEL_PARAMS,
    ).to(device)
    net_d = MultiPeriodDiscriminator().to(device)
    n_g = sum(p.numel() for p in net_g.parameters()) / 1e6
    n_d = sum(p.numel() for p in net_d.parameters()) / 1e6
    print(f"[smoke] params — G={n_g:.2f}M, D={n_d:.2f}M")

    # 3) Generator forward pass
    print("[smoke] generator forward ...")
    (o, l_dur_sdp, l_dur_dp, attn,
     ids_slice, x_mask, y_mask,
     (z, z_p, m_p_exp, logs_p_exp, m_q, logs_q)) = net_g(
        batch["x"], batch["x_lengths"],
        batch["spec"], batch["spec_lengths"],
        batch["sid"], batch["tone"], batch["language"],
        batch["bert"], batch["ja_bert"],
    )
    print(f"[smoke] gen output shapes — o={tuple(o.shape)}, "
          f"attn={tuple(attn.shape)}, ids_slice={tuple(ids_slice.shape)}")
    assert torch.isfinite(o).all(), "generator output is not finite"

    # 4) Mel + slice
    mel_y_full = spec_to_mel_torch(batch["spec"], FILTER_LENGTH, N_MELS,
                                   SAMPLING_RATE, F_MIN, F_MAX)
    mel_y_slice = commons_extract(mel_y_full, ids_slice, SEGMENT_FRAMES)
    mel_o = mel_spectrogram_torch(
        o.squeeze(1), FILTER_LENGTH, N_MELS, SAMPLING_RATE,
        HOP_LENGTH, FILTER_LENGTH, F_MIN, F_MAX,
    )
    sample_ids = ids_slice * HOP_LENGTH
    sample_size = SEGMENT_FRAMES * HOP_LENGTH
    wav_slice = commons_extract(batch["wav"], sample_ids, sample_size)
    print(f"[smoke] mel shapes — mel_o={tuple(mel_o.shape)}, "
          f"mel_y_slice={tuple(mel_y_slice.shape)}, "
          f"wav_slice={tuple(wav_slice.shape)}")

    # 5) Discriminator step
    print("[smoke] discriminator step ...")
    y_d_rs, y_d_gs, _, _ = net_d(wav_slice, o.detach())
    loss_d, _, _ = discriminator_loss(y_d_rs, y_d_gs)
    assert math.isfinite(loss_d.item()), f"loss_d not finite: {loss_d.item()}"
    print(f"[smoke] loss_d={loss_d.item():.4f}")

    # 6) Generator loss bundle
    print("[smoke] generator losses ...")
    y_d_rs2, y_d_gs2, fmap_rs, fmap_gs = net_d(wav_slice, o)
    loss_fm = feature_matching_loss(fmap_rs, fmap_gs)
    loss_gen, _ = generator_loss(y_d_gs2)
    loss_mel_v = mel_loss(mel_o, mel_y_slice) * C_MEL
    loss_kl_v = kl_loss(z_p, logs_q, m_p_exp, logs_p_exp, y_mask) * C_KL
    loss_dur = (l_dur_sdp + l_dur_dp) * C_DUR
    loss_g_total = loss_gen + loss_fm + loss_mel_v + loss_kl_v + loss_dur

    for name, val in [
        ("loss_gen (adv)", loss_gen),
        ("loss_fm", loss_fm),
        ("loss_mel", loss_mel_v),
        ("loss_kl", loss_kl_v),
        ("loss_dur", loss_dur),
        ("loss_g_total", loss_g_total),
    ]:
        v = float(val.item())
        ok = math.isfinite(v)
        print(f"  {name:<18} = {v:+.4f}  {'OK' if ok else 'NOT FINITE'}")
        assert ok, f"{name} is not finite ({v})"

    # 7) Backward pass to confirm grads compute
    print("[smoke] backward ...")
    (loss_g_total).backward()
    print("[smoke] backward OK.")

    print("[smoke] PASS — pipeline is healthy.")


if __name__ == "__main__":
    main()
