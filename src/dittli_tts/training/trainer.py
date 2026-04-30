"""Single-GPU training loop for VoiceSynthesizer + HiFi-GAN discriminator.

Run via `scripts/finetune_de.py` for the German fine-tune (the script handles
loading the English checkpoint and remapping the embedding before calling
`Trainer.run`).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dittli_tts.audio import (
    commons_extract,
    mel_spectrogram_torch,
    spec_to_mel_torch,
)
from dittli_tts.data.dataset import ThorstenDataset, collate
from dittli_tts.training.losses import (
    discriminator_loss,
    feature_matching_loss,
    generator_loss,
    kl_loss,
    mel_loss,
)
from dittli_tts.models.discriminator import MultiPeriodDiscriminator
from dittli_tts.models.synthesizer import VoiceSynthesizer
from dittli_tts.nn import commons
from dittli_tts.text.symbols import symbols
from dittli_tts.utils.config import (
    FILTER_LENGTH,
    HOP_LENGTH,
    MODEL_PARAMS,
    SAMPLING_RATE,
    SPEC_CHANNELS,
)
from dittli_tts.utils.train_config import (
    BATCH_SIZE,
    BETAS,
    C_DUR,
    C_KL,
    C_MEL,
    EPS,
    F_MAX,
    F_MIN,
    GRAD_CLIP,
    LEARNING_RATE,
    LOG_INTERVAL,
    LR_DECAY,
    N_MELS,
    NUM_WORKERS,
    SAVE_INTERVAL,
    SEGMENT_SIZE,
)


@dataclass
class TrainerConfig:
    metadata_path: str
    wavs_dir: str
    ckpt_dir: str
    init_g_ckpt: str | None = None      # English G.pth to fine-tune from
    init_d_ckpt: str | None = None
    n_speakers: int = 1
    total_steps: int = 100_000
    batch_size: int = BATCH_SIZE
    segment_size: int = SEGMENT_SIZE
    learning_rate: float = LEARNING_RATE
    save_interval: int = SAVE_INTERVAL
    log_interval: int = LOG_INTERVAL
    num_workers: int = NUM_WORKERS
    device: str = "cuda"
    amp: bool = True


def _slice_wav_for_segment(wav: torch.Tensor, ids_slice: torch.Tensor, segment_size: int):
    """Slice the raw wav waveform to match a spec-frame segment."""
    sample_ids = ids_slice * HOP_LENGTH
    sample_size = segment_size * HOP_LENGTH
    return commons_extract(wav, sample_ids, sample_size)


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
        self.device = device

        self.dataset = ThorstenDataset(cfg.metadata_path, cfg.wavs_dir)
        self.loader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=lambda b: collate(b),
            pin_memory=device.type == "cuda",
            drop_last=True,
        )

        self.net_g = VoiceSynthesizer(
            len(symbols),
            SPEC_CHANNELS,
            cfg.segment_size,
            n_speakers=cfg.n_speakers,
            **MODEL_PARAMS,
        ).to(device)

        self.net_d = MultiPeriodDiscriminator().to(device)

        self.opt_g = AdamW(
            self.net_g.parameters(),
            lr=cfg.learning_rate,
            betas=BETAS,
            eps=EPS,
        )
        self.opt_d = AdamW(
            self.net_d.parameters(),
            lr=cfg.learning_rate,
            betas=BETAS,
            eps=EPS,
        )

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.opt_g, gamma=LR_DECAY)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.opt_d, gamma=LR_DECAY)

        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

        self.step = 0
        if cfg.init_g_ckpt:
            self._load_g(cfg.init_g_ckpt)
        if cfg.init_d_ckpt:
            self._load_d(cfg.init_d_ckpt)

    def _load_g(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        own = self.net_g.state_dict()
        loaded, skipped = 0, 0
        new_state = {}
        for k, v in state.items():
            key = k[7:] if k.startswith("module.") else k
            if key in own and v.shape == own[key].shape:
                new_state[key] = v
                loaded += 1
            else:
                skipped += 1
        self.net_g.load_state_dict(new_state, strict=False)
        print(f"[trainer] generator: loaded {loaded} tensors, skipped {skipped}.")

    def _load_d(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        try:
            self.net_d.load_state_dict(state, strict=False)
            print(f"[trainer] discriminator: loaded from {path}.")
        except Exception as e:
            print(f"[trainer] discriminator: failed to load ({e}); starting fresh.")

    def save(self, tag: str):
        gpath = os.path.join(self.cfg.ckpt_dir, f"G_{tag}.pth")
        dpath = os.path.join(self.cfg.ckpt_dir, f"D_{tag}.pth")
        torch.save({"model": self.net_g.state_dict(), "step": self.step}, gpath)
        torch.save({"model": self.net_d.state_dict(), "step": self.step}, dpath)
        print(f"[trainer] saved {gpath} and {dpath}.")

    def step_one(self, batch: dict) -> dict:
        device = self.device
        x = batch["x"].to(device, non_blocking=True)
        x_lengths = batch["x_lengths"].to(device, non_blocking=True)
        spec = batch["spec"].to(device, non_blocking=True)
        spec_lengths = batch["spec_lengths"].to(device, non_blocking=True)
        wav = batch["wav"].to(device, non_blocking=True)
        sid = batch["sid"].to(device, non_blocking=True)
        tone = batch["tone"].to(device, non_blocking=True)
        language = batch["language"].to(device, non_blocking=True)
        bert = batch["bert"].to(device, non_blocking=True)
        ja_bert = batch["ja_bert"].to(device, non_blocking=True)

        amp_enabled = self.scaler.is_enabled()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            (
                o, l_dur_sdp, l_dur_dp, _attn,
                ids_slice, x_mask, y_mask,
                (z, z_p, m_p_exp, logs_p_exp, m_q, logs_q),
            ) = self.net_g(
                x, x_lengths, spec, spec_lengths, sid, tone, language, bert, ja_bert,
            )

            mel_y_full = spec_to_mel_torch(
                spec, FILTER_LENGTH, N_MELS, SAMPLING_RATE, F_MIN, F_MAX,
            )
            mel_y_slice = commons_extract(mel_y_full, ids_slice, self.cfg.segment_size)
            mel_o = mel_spectrogram_torch(
                o.squeeze(1), FILTER_LENGTH, N_MELS, SAMPLING_RATE,
                HOP_LENGTH, FILTER_LENGTH, F_MIN, F_MAX,
            )

            wav_slice = _slice_wav_for_segment(wav, ids_slice, self.cfg.segment_size)

        # ---- Discriminator step ----
        self.opt_d.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            y_d_rs, y_d_gs, _, _ = self.net_d(wav_slice, o.detach())
            loss_d, _, _ = discriminator_loss(y_d_rs, y_d_gs)
        self.scaler.scale(loss_d).backward()
        self.scaler.unscale_(self.opt_d)
        commons.clip_grad_value_(self.net_d.parameters(), GRAD_CLIP)
        self.scaler.step(self.opt_d)

        # ---- Generator step ----
        self.opt_g.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            y_d_rs2, y_d_gs2, fmap_rs, fmap_gs = self.net_d(wav_slice, o)
            loss_fm = feature_matching_loss(fmap_rs, fmap_gs)
            loss_gen, _ = generator_loss(y_d_gs2)
            loss_mel = mel_loss(mel_o, mel_y_slice) * C_MEL
            loss_kl = kl_loss(z_p, logs_q, m_p_exp, logs_p_exp, y_mask) * C_KL
            loss_dur = (l_dur_sdp + l_dur_dp) * C_DUR
            loss_g_total = loss_gen + loss_fm + loss_mel + loss_kl + loss_dur
        self.scaler.scale(loss_g_total).backward()
        self.scaler.unscale_(self.opt_g)
        commons.clip_grad_value_(self.net_g.parameters(), GRAD_CLIP)
        self.scaler.step(self.opt_g)
        self.scaler.update()

        # MAS noise annealing (matches the model's noise_scale_delta)
        if self.net_g.use_noise_scaled_mas:
            self.net_g.current_mas_noise_scale = max(
                0.0, self.net_g.current_mas_noise_scale - self.net_g.noise_scale_delta
            )

        return {
            "loss_d": loss_d.detach().item(),
            "loss_g": loss_g_total.detach().item(),
            "loss_mel": loss_mel.detach().item(),
            "loss_kl": loss_kl.detach().item(),
            "loss_dur": loss_dur.detach().item(),
            "loss_fm": loss_fm.detach().item(),
            "loss_gen": loss_gen.detach().item(),
        }

    def run(self):
        print(f"[trainer] device={self.device} batch_size={self.cfg.batch_size} "
              f"steps={self.cfg.total_steps}")
        t0 = time.time()
        last_log = t0

        while self.step < self.cfg.total_steps:
            for batch in self.loader:
                if self.step >= self.cfg.total_steps:
                    break
                stats = self.step_one(batch)
                self.step += 1

                if self.step % self.cfg.log_interval == 0:
                    now = time.time()
                    sps = self.cfg.log_interval / max(1e-6, now - last_log)
                    last_log = now
                    print(
                        f"step {self.step}/{self.cfg.total_steps} "
                        f"lr={self.opt_g.param_groups[0]['lr']:.2e} "
                        f"d={stats['loss_d']:.3f} g={stats['loss_g']:.3f} "
                        f"mel={stats['loss_mel']:.3f} kl={stats['loss_kl']:.3f} "
                        f"dur={stats['loss_dur']:.3f} fm={stats['loss_fm']:.3f} "
                        f"adv={stats['loss_gen']:.3f} sps={sps:.2f}"
                    )
                if self.step % self.cfg.save_interval == 0:
                    self.save(str(self.step))

            self.scheduler_g.step()
            self.scheduler_d.step()

        self.save("final")
        print(f"[trainer] done in {time.time() - t0:.1f}s.")
