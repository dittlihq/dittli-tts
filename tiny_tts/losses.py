"""Training losses for VITS-style TTS with HiFi-GAN discriminator."""
import torch
from torch.nn import functional as F

from tiny_tts.nn import commons


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """KL[q(z|y) || p(z|x)] with the prior expanded to spec frames.

    Shapes: z_p, m_p, logs_p, m_q (implicit), logs_q ~ [B, C, T_y]
            z_mask ~ [B, 1, T_y]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    return kl / torch.sum(z_mask)


def feature_matching_loss(fmap_r, fmap_g):
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss = loss + torch.mean(torch.abs(rl - gl))
    return loss * 2.0


def discriminator_loss(disc_real_outputs, disc_gen_outputs):
    """Least-squares discriminator loss. Returns (total, per_disc_real, per_disc_gen)."""
    loss = 0.0
    r_losses, g_losses = [], []
    for dr, dg in zip(disc_real_outputs, disc_gen_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss = loss + r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_gen_outputs):
    """LSGAN generator loss against the gen-side discriminator outputs."""
    loss = 0.0
    g_losses = []
    for dg in disc_gen_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        g_losses.append(l)
        loss = loss + l
    return loss, g_losses


def mel_loss(o_mel, y_mel):
    """L1 loss on mel-spectrograms."""
    return F.l1_loss(o_mel, y_mel)
