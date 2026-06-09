"""HiFi-GAN MultiPeriod + MultiScale discriminator.

Used during training only — the generator (VoiceSynthesizer) is the only
component shipped at inference time.
"""

import torch
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

LRELU_SLOPE = 0.1


def _get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                norm(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(_get_padding(kernel_size), 0))),
                norm(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(_get_padding(kernel_size), 0))),
                norm(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(_get_padding(kernel_size), 0))),
                norm(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(_get_padding(kernel_size), 0))),
                norm(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(_get_padding(kernel_size), 0))),
            ]
        )
        self.conv_post = norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        # pad to multiple of period
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), mode="reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                norm(Conv1d(1, 16, 15, 1, padding=7)),
                norm(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """HiFi-GAN MPD + a single MSD branch (combined as in the original paper)."""

    def __init__(self, periods=(2, 3, 5, 7, 11), use_spectral_norm=False):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
            + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods]
        )

    def forward(self, y, y_hat):
        """Returns (logits_real_list, logits_gen_list, fmap_real_list, fmap_gen_list)."""
        y_d_rs, y_d_gs = [], []
        fmap_rs, fmap_gs = [], []
        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
