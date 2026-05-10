"""End-to-end smoke test of the German training pipeline.

Runs on CPU. Reads metadata, builds the dataset, fetches a tiny batch
via the collator, and runs ONE forward pass through
`VoiceSynthesizer.forward()` plus a discriminator step. Verifies losses
are finite. Does NOT save a checkpoint.

Requires the Thorsten dataset (~3.7 GB) — guarded by the `slow` marker
and the `thorsten_*` fixtures that skip if the data isn't downloaded.

    pytest -m slow
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def _ensure_cache(dataset, n: int) -> None:
    from dittli_tts.data.dataset import _phones_path, compute_and_cache

    for wav_path, transcript in dataset.items[:n]:
        spec = wav_path + ".spec.pt"
        ph = _phones_path(wav_path, dataset.language)
        if not (os.path.exists(spec) and os.path.exists(ph)):
            compute_and_cache(wav_path, transcript, language=dataset.language)


def test_training_step_produces_finite_losses(
    thorsten_metadata: Path,
    thorsten_wavs: Path,
):
    import torch

    from dittli_tts.audio import (
        commons_extract,
        mel_spectrogram_torch,
        spec_to_mel_torch,
    )
    from dittli_tts.data.dataset import ThorstenDataset, collate
    from dittli_tts.models.discriminator import MultiPeriodDiscriminator
    from dittli_tts.models.synthesizer import VoiceSynthesizer
    from dittli_tts.text.symbols import symbols
    from dittli_tts.training.losses import (
        discriminator_loss,
        feature_matching_loss,
        generator_loss,
        kl_loss,
        mel_loss,
    )
    from dittli_tts.utils.config import (
        FILTER_LENGTH,
        HOP_LENGTH,
        MODEL_PARAMS,
        SAMPLING_RATE,
        SEGMENT_FRAMES,
        SPEC_CHANNELS,
    )
    from dittli_tts.utils.train_config import C_DUR, C_KL, C_MEL, F_MAX, F_MIN, N_MELS

    device = torch.device("cpu")
    batch_size = 2

    dataset = ThorstenDataset(str(thorsten_metadata), str(thorsten_wavs), require_cache=False)
    assert len(dataset) > 0, f"no rows in {thorsten_metadata}"

    _ensure_cache(dataset, batch_size)
    batch = collate([dataset[i] for i in range(batch_size)])
    for k in ("x", "x_lengths", "tone", "language", "spec", "spec_lengths",
              "wav", "wav_lengths", "sid", "bert", "ja_bert"):
        batch[k] = batch[k].to(device)

    net_g = VoiceSynthesizer(
        len(symbols), SPEC_CHANNELS, SEGMENT_FRAMES,
        n_speakers=1, **MODEL_PARAMS,
    ).to(device)
    net_d = MultiPeriodDiscriminator().to(device)

    (o, l_dur_sdp, l_dur_dp, _attn,
     ids_slice, _x_mask, y_mask,
     (_z, z_p, m_p_exp, logs_p_exp, _m_q, logs_q)) = net_g(
        batch["x"], batch["x_lengths"],
        batch["spec"], batch["spec_lengths"],
        batch["sid"], batch["tone"], batch["language"],
        batch["bert"], batch["ja_bert"],
    )
    assert torch.isfinite(o).all()

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

    y_d_rs, y_d_gs, _, _ = net_d(wav_slice, o.detach())
    loss_d, _, _ = discriminator_loss(y_d_rs, y_d_gs)
    assert math.isfinite(loss_d.item())

    y_d_rs2, y_d_gs2, fmap_rs, fmap_gs = net_d(wav_slice, o)
    loss_fm = feature_matching_loss(fmap_rs, fmap_gs)
    loss_gen, _ = generator_loss(y_d_gs2)
    loss_mel_v = mel_loss(mel_o, mel_y_slice) * C_MEL
    loss_kl_v = kl_loss(z_p, logs_q, m_p_exp, logs_p_exp, y_mask) * C_KL
    loss_dur = (l_dur_sdp + l_dur_dp) * C_DUR
    loss_g_total = loss_gen + loss_fm + loss_mel_v + loss_kl_v + loss_dur

    for name, val in [
        ("loss_gen", loss_gen),
        ("loss_fm", loss_fm),
        ("loss_mel", loss_mel_v),
        ("loss_kl", loss_kl_v),
        ("loss_dur", loss_dur),
        ("loss_g_total", loss_g_total),
    ]:
        assert math.isfinite(val.item()), f"{name} not finite"

    loss_g_total.backward()
