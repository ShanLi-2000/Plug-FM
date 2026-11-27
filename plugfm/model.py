"""Plug-FM backbone and train-only LMMSE calibration."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .observation import UplinkCombPilotConfig, uplink_comb_ls, uplink_pilot_grid


class UplinkFrequencyLMMSE:
    """Train-only frequency covariance, independently applied to all BS antennas."""

    def __init__(self, pilots: torch.Tensor, mask: torch.Tensor,
                 mean_channel: torch.Tensor, gain: torch.Tensor):
        self.pilots, self.mask = pilots, mask
        self.mean_channel, self.gain = mean_channel, gain

    @classmethod
    @torch.no_grad()
    def fit(cls, train_channels: torch.Tensor, pilots: torch.Tensor, mask: torch.Tensor,
            noise_variance: float, *, shrinkage: float = 0.05,
            batch_size: int = 256) -> "UplinkFrequencyLMMSE":
        if train_channels.ndim != 3 or not torch.is_complex(train_channels):
            raise ValueError("Expected complex TRAIN channels [B,M,F]")
        if not 0 <= shrinkage < 1 or noise_variance <= 0 or batch_size < 1:
            raise ValueError("Invalid LMMSE regularization, noise, or batch size")
        channels = train_channels.cpu()
        n, m, f = channels.shape
        if pilots.shape != mask.shape or mask.shape[1] != f:
            raise ValueError("Pilot grid and training channel sizes disagree")
        freqs = torch.where(mask.any(dim=0))[0]
        mean = torch.zeros((m, f), dtype=torch.complex128)
        for start in range(0, n, batch_size):
            mean += channels[start:start + batch_size].to(torch.complex128).sum(0)
        mean /= n
        covariance = torch.zeros((f, f), dtype=torch.complex128)
        for start in range(0, n, batch_size):
            centered = channels[start:start + batch_size].to(torch.complex128) - mean
            covariance += torch.einsum("bmf,bmg->fg", centered.conj(), centered)
        covariance /= n * m
        covariance = covariance.conj()  # E[(H_f-mu_f) (H_g-mu_g)^*].
        power = covariance.diag().real.mean()
        covariance = (1 - shrinkage) * covariance + shrinkage * power * torch.eye(f, dtype=torch.complex128)
        pilot_energy = pilots.abs().square().sum(dim=0).index_select(0, freqs).to(torch.float64)
        pilot_noise = noise_variance / pilot_energy
        cpp = covariance.index_select(0, freqs).index_select(1, freqs)
        cpp = cpp + torch.diag(pilot_noise.to(torch.complex128))
        chp = covariance.index_select(1, freqs)
        gain = torch.linalg.solve(cpp.T, chp.T).T
        return cls(pilots.cpu().to(torch.complex64), mask.cpu(),
                   mean.to(torch.complex64), gain.to(torch.complex64))

    def to(self, device: torch.device | str) -> "UplinkFrequencyLMMSE":
        return type(self)(self.pilots.to(device), self.mask.to(device),
                          self.mean_channel.to(device), self.gain.to(device))

    def estimate(self, observation: dict) -> torch.Tensor:
        if not torch.equal(observation["mask"], self.mask) or not torch.allclose(observation["pilots"], self.pilots):
            raise ValueError("Pilot grid differs from train-only LMMSE calibration")
        ls, frequencies = uplink_comb_ls(observation)
        centered = ls - self.mean_channel.index_select(-1, frequencies)[None]
        return self.mean_channel[None] + torch.einsum("bmi,fi->bmf", centered, self.gain)


class GridAveragePool(nn.Module):
    """Equivalent adaptive grid average using deterministic nonoverlap kernels."""
    def forward(self, x):
        height, width = x.shape[-2:]
        if height % 2 or width % 2:
            return F.adaptive_avg_pool2d(x, (2, 2))
        return F.avg_pool2d(x, kernel_size=(height // 2, width // 2),
                            stride=(height // 2, width // 2))


@dataclass(frozen=True)
class CompactConfig:
    width: int = 32
    depth: int = 2
    embedding_dim: int = 64
    data_mode: str = "moment"
    channel_rms: float = 1.0
    noise_variance: float = 0.1
    embedding_hidden: int = 0
    representation_mode: str = "aligned"
    global_attention: bool = False
    attention_layers: int = 1
    frequency_bins: int = 2
    global_readout: str = "mean"


class ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                  nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return F.gelu(x + self.body(x))


class CompactUplinkStudent(nn.Module):
    def __init__(self, config: CompactConfig, estimator: UplinkFrequencyLMMSE):
        super().__init__()
        if config.data_mode not in ("soft", "moment", "beam_power", "angular_moment"):
            raise ValueError("Invalid data statistic")
        if config.representation_mode not in ("aligned", "pooled"):
            raise ValueError("Invalid representation mode")
        if config.frequency_bins<1 or 32%config.frequency_bins:
            raise ValueError("frequency_bins must divide the 32-carrier grid")
        if config.global_readout not in ('mean','cls','stats') or (config.global_readout!='mean' and not config.global_attention):
            raise ValueError('CLS/statistical readout requires global attention')
        if min(config.width, config.depth, config.embedding_dim) < 1:
            raise ValueError("Invalid model dimensions")
        if config.channel_rms <= 0 or config.noise_variance <= 0:
            raise ValueError("Invalid calibration")
        self.config = config
        self.representation_dim = config.embedding_dim if config.representation_mode == "aligned" else 2*config.width*(2 if config.global_readout=='stats' else 1)
        for name, value in (("pilots", estimator.pilots), ("pilot_mask", estimator.mask),
                            ("mean_channel", estimator.mean_channel), ("lmmse_gain", estimator.gain)):
            self.register_buffer(name, value.clone())
        self.register_buffer("channel_rms", torch.tensor(config.channel_rms))
        self.register_buffer("noise_variance", torch.tensor(config.noise_variance))
        self.pilot_stem = nn.Sequential(nn.Conv2d(2, config.width, 3, padding=1), nn.GELU())
        self.data_stem = nn.Sequential(nn.Conv2d(1 if config.data_mode=="beam_power" else 3,
                                                 config.width, 3, padding=1), nn.GELU())
        channels = 2 * config.width
        self.shared = nn.Sequential(*[ConvBlock(channels) for _ in range(config.depth)])
        if config.global_attention:
            if config.global_readout=='cls':
                self.global_token=nn.Parameter(torch.randn(1,1,channels)*.02)
            self.token_position = nn.Parameter(torch.randn(32*config.frequency_bins,channels)*.02)
            self.attention = nn.TransformerEncoderLayer(channels,4,dim_feedforward=2*channels,
                dropout=0.,activation="gelu",batch_first=True)
            self.extra_attention = nn.ModuleList([nn.TransformerEncoderLayer(channels,4,
                dim_feedforward=2*channels,dropout=0.,activation="gelu",batch_first=True)
                for _ in range(config.attention_layers-1)])
            self.embedding_head = nn.Linear(channels*(2 if config.global_readout=='stats' else 1),config.embedding_dim)
        elif config.embedding_hidden:
            self.embedding_head = nn.Sequential(GridAveragePool(), nn.Flatten(),
                nn.Linear(4 * channels, config.embedding_hidden), nn.GELU(),
                nn.Linear(config.embedding_hidden, config.embedding_dim))
        else:
            self.embedding_head = nn.Sequential(GridAveragePool(), nn.Flatten(),
                                                nn.Linear(4 * channels, config.embedding_dim))
        self.channel_head = nn.Conv2d(channels, 2, 1)
        nn.init.zeros_(self.channel_head.weight)
        nn.init.zeros_(self.channel_head.bias)

    def public_observation(self, y):
        return {"pilot_y": y * self.pilot_mask[None, None], "pilots": self.pilots,
                "mask": self.pilot_mask,
                "pilot_subcarriers": torch.where(self.pilot_mask.any(0))[0]}

    def pilot_channel(self, y):
        obs = self.public_observation(y)
        ls, indices = uplink_comb_ls(obs)
        delta = ls - self.mean_channel.index_select(-1, indices)[None]
        return self.mean_channel[None] + torch.einsum("bmi,fi->bmf", delta, self.lmmse_gain)

    def data_statistics(self, y, pilot):
        mask = ~self.pilot_mask
        yd = y * mask[None, None]
        count = mask.sum(0).to(y.real.dtype).clamp_min(1)
        if self.config.data_mode=="beam_power":
            angular=torch.fft.fftshift(torch.fft.fft(yd,dim=1,norm="ortho"),dim=1)
            return None, angular.abs().square().sum(2)/count[None,None]
        anchor_power = pilot.abs().square().sum(1).clamp_min(1e-8)
        matched = (pilot.conj()[:, :, None] * yd).sum(1)
        if self.config.data_mode == "soft":
            z = matched / anchor_power[:, None]
            q = torch.tensor([-1-1j, -1+1j, 1-1j, 1+1j], device=y.device,
                             dtype=y.dtype) / (2 ** .5)
            temperature = (self.noise_variance / anchor_power).clamp_min(1e-6)
            probabilities = torch.softmax(-(z[..., None] - q).abs().square() /
                                          temperature[:, None, :, None], -1)
            symbols = (probabilities * q).sum(-1) * mask[None]
            channel = (yd * symbols.conj()[:, None]).sum(2) / count[None, None]
        else:
            # (R_YY - sigma^2 I) h_p / regularized ||h_p||^2. Common unknown
            # symbols disappear in the second moment; no constellation decoder.
            denominator = anchor_power + y.shape[1] * self.noise_variance / count[None]
            channel = ((yd * matched.conj()[:, None]).sum(2) / count[None, None]
                       - self.noise_variance * pilot) / denominator[:, None].clamp_min(1e-8)
        if self.config.data_mode=="angular_moment":
            angular=torch.fft.fftshift(torch.fft.fft(yd,dim=1,norm="ortho"),dim=1)
            power=angular.abs().square().sum(2)/count[None,None]
        else:
            power = yd.abs().square().sum(2) / count[None, None]
        return channel, power

    def forward(self, received, *, return_alignment=True):
        y = received["y"] if isinstance(received, dict) else received
        shape = (self.mean_channel.shape[0], *self.pilot_mask.shape)
        if not torch.is_complex(y) or y.ndim != 4 or tuple(y.shape[1:]) != shape:
            raise ValueError(f"Expected complex received [B,{shape[0]},{shape[1]},{shape[2]}]")
        y = y.to(dtype=self.pilots.dtype)
        h0 = self.pilot_channel(y)
        hd, pd = self.data_statistics(y, h0)
        scale = self.channel_rms
        if self.config.data_mode in ("beam_power","angular_moment"):
            pilot_domain=torch.fft.fftshift(torch.fft.fft(h0,dim=1,norm="ortho"),dim=1)
            power_feature=torch.log1p((pd-self.noise_variance).clamp_min(0)/scale.square())
            if self.config.data_mode=="angular_moment":
                hd=torch.fft.fftshift(torch.fft.fft(hd,dim=1,norm="ortho"),dim=1)
                difference=(hd-pilot_domain)/scale
                data_input=torch.stack((difference.real,difference.imag,power_feature),1)
            else:
                data_input=power_feature[:,None]
        else:
            pilot_domain=h0
            difference = (hd - h0) / scale
            data_input = torch.stack((difference.real, difference.imag,
                                      torch.log1p(pd / scale.square())), 1)
        pilot_input = torch.stack((pilot_domain.real / scale, pilot_domain.imag / scale), 1)
        feature = self.shared(torch.cat((self.pilot_stem(pilot_input),
                                         self.data_stem(data_input)), 1))
        delta = self.channel_head(feature)
        internal = pilot_domain + scale * torch.complex(delta[:, 0], delta[:, 1])
        if self.config.data_mode in ("beam_power","angular_moment"):
            internal=torch.fft.ifft(torch.fft.ifftshift(internal,dim=1),dim=1,norm="ortho")
        if self.config.global_attention:
            # Keep every antenna; summarize configurable contiguous frequency bins.
            if feature.shape[-2:] != (32,32):raise ValueError("This token layout expects 32x32")
            pooling_width=32//self.config.frequency_bins
            tokens = F.avg_pool2d(feature,kernel_size=(1,pooling_width),stride=(1,pooling_width)).flatten(2).transpose(1,2)
            tokens=tokens+self.token_position[None]
            if self.config.global_readout=='cls':
                tokens=torch.cat((self.global_token.expand(len(tokens),-1,-1),tokens),1)
            tokens=self.attention(tokens)
            for layer in self.extra_attention:tokens=layer(tokens)
            global_feature = tokens[:,0] if self.config.global_readout=='cls' else tokens.mean(1)
            if self.config.global_readout=='stats':
                global_feature=torch.cat((global_feature,(tokens.var(1,unbiased=False)+1e-6).sqrt()),-1)
            alignment_input = global_feature
        else:
            global_feature = feature.mean((2,3))
            alignment_input = feature
        aligned = (self.embedding_head(alignment_input) if self.embedding_head is not None and
                   (return_alignment or self.config.representation_mode == "aligned") else None)
        embedding = aligned if self.config.representation_mode == "aligned" else global_feature
        output = {"embedding": embedding, "spatial_features": feature, "refined_channel": internal}
        if aligned is not None: output["aligned_embedding"] = aligned
        return output


class NoiseCalibration:
    def __init__(self, state, device):
        self.state = state
        self.device = torch.device(device)
        self.mean = state['mean'].to(device)
        self.covariance = state['covariance'].to(device)
        self.pilots = state['pilots'].to(device)
        self.mask = state['mask'].to(device)
        self.power = state['reference_power']
        self.indices = torch.where(self.mask.any(0))[0]
        self.energy = self.pilots.abs().square().sum(0)[self.indices].double()
        self.gains = {}

    @classmethod
    def fit(cls, channels, protocol, reference_power, device='cpu'):
        h = channels.cpu().to(torch.complex128)
        mean = h.mean(0)
        z = h - mean
        covariance = torch.einsum('bmf,bmg->fg', z, z.conj()) / (h.shape[0] * h.shape[1])
        covariance = .95 * covariance + .05 * covariance.diag().real.mean() * torch.eye(h.shape[-1])
        pilots, mask = uplink_pilot_grid(UplinkCombPilotConfig(**protocol))
        return cls({'mean': mean.to(torch.complex64), 'covariance': covariance,
                    'pilots': pilots, 'mask': mask, 'reference_power': float(reference_power),
                    'train_users': len(h), 'shrinkage': .05}, device)

    def variance(self, snr):
        return self.power * 10. ** (-float(snr) / 10.)

    def gain(self, snr):
        key = float(snr)
        if key not in self.gains:
            covariance = self.covariance
            cross = covariance[:, self.indices]
            pilot_cov = covariance[self.indices][:, self.indices]
            pilot_cov = pilot_cov + torch.diag((self.variance(snr) / self.energy).to(covariance.dtype))
            self.gains[key] = torch.linalg.solve(pilot_cov.T, cross.T).T.to(torch.complex64)
        return self.gains[key]

    def estimator(self, snr):
        return UplinkFrequencyLMMSE(self.pilots, self.mask, self.mean, self.gain(snr))

    def estimate(self, observation, snr):
        ls, indices = uplink_comb_ls(observation)
        return self.mean[None] + torch.einsum('bmi,fi->bmf', ls-self.mean[:, indices][None], self.gain(snr))


class MixedStudent(CompactUplinkStudent):
    """Original body; noise changes analytical preprocessing, never neural weights.

    Pilot-only is a controlled input ablation with the SAME parameter capacity:
    its data-stem input is constant zero during both training and inference.
    """
    def __init__(self, config, calibration, pilot_only=False):
        super().__init__(config, calibration.estimator(0))
        if config.data_mode != 'moment':
            raise ValueError('This locked experiment uses the original moment data branch')
        self.calibration = calibration
        self.pilot_only = pilot_only

    def data_statistics(self, y, pilot):
        if self.pilot_only:
            return pilot, torch.zeros_like(pilot.real)
        return super().data_statistics(y, pilot)

    def forward(self, received, snr, *, return_alignment=True):
        with torch.no_grad():
            self.lmmse_gain.copy_(self.calibration.gain(snr))
            self.noise_variance.fill_(self.calibration.variance(snr))
        return super().forward(received, return_alignment=return_alignment)


class ContinuousCalibration(NoiseCalibration):
    """Bound analytical coefficient cache; continuous SNR must not grow it forever."""
    def gain(self, snr):
        key = float(snr)
        if key not in self.gains and len(self.gains) >= 32:
            self.gains.pop(next(iter(self.gains)))
        return super().gain(snr)
