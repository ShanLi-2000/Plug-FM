"""Frozen-backbone task readouts and CSI loss."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualCSIRefiner(nn.Module):
    """Small supervised reconstruction head, separately fitted for each student."""

    def __init__(self, embedding_dim: int, channel_rms: float, width: int = 16, depth: int = 3):
        super().__init__()
        self.register_buffer("channel_rms", torch.tensor(float(channel_rms)))
        self.stem = nn.Conv2d(5, width, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                          nn.Conv2d(width, width, 3, padding=1))
            for _ in range(depth)
        ])
        self.condition = nn.Linear(embedding_dim, 2 * width)
        self.final = nn.Conv2d(width, 2, 3, padding=1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, coarse: torch.Tensor, embedding: torch.Tensor, pilot_mask: torch.Tensor,
                student_channel: torch.Tensor) -> torch.Tensor:
        if coarse.ndim != 3 or not torch.is_complex(coarse):
            raise ValueError("coarse must be complex [B,M,F]")
        scale = self.channel_rms.to(coarse.real.dtype)
        tone = pilot_mask.any(dim=0).to(coarse.real.dtype)
        x = torch.stack((coarse.real / scale, coarse.imag / scale,
                         torch.tanh(student_channel.real / scale),
                         torch.tanh(student_channel.imag / scale),
                         tone[None, None].expand(len(coarse), coarse.shape[1], -1)), dim=1)
        x = F.gelu(self.stem(x))
        gain, shift = self.condition(embedding).chunk(2, dim=-1)
        x = x * (1 + 0.1 * torch.tanh(gain[:, :, None, None])) + 0.1 * shift[:, :, None, None]
        for block in self.blocks:
            x = F.gelu(x + block(x))
        correction = self.final(x) * scale
        return coarse + torch.complex(correction[:, 0], correction[:, 1])


def nmse_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (estimate - target).abs().square().sum() / target.abs().square().sum().clamp_min(1e-8)


class LinearTaskHead(nn.Module):
    def __init__(self, mean, std, classes):
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clamp_min(1e-3).clone())
        self.linear = nn.Linear(len(mean), classes)
    def forward(self, feature):
        return self.linear((feature-self.mean)/self.std)
