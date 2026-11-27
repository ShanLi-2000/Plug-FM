"""Complex uplink observations and comb-pilot least squares."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

ChannelMapping = Literal["native_uplink", "tdd_reciprocal_downlink"]


def generate_data_symbols(
    batch_size: int,
    pilot_mask: torch.Tensor,
    *,
    modulation: str = "qpsk",
    dtype: torch.dtype = torch.complex64,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate unit-average-power data ``S[B,T,F]``, zero on pilot elements.

    This helper is for dataset/simulator construction.  Its output is ground
    truth and must not be passed to the student as an observation.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if pilot_mask.ndim != 2 or pilot_mask.dtype != torch.bool:
        raise ValueError("pilot_mask must be a bool [T,F] tensor")
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError("Data symbol dtype must be complex64 or complex128")
    if modulation not in ("gaussian", "qpsk", "16qam"):
        raise ValueError("modulation must be 'gaussian', 'qpsk', or '16qam'")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pilot_mask.device).manual_seed(int(seed))
    shape = (batch_size, *pilot_mask.shape)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    if modulation == "gaussian":
        real = torch.randn(shape, device=pilot_mask.device, dtype=real_dtype, generator=generator)
        imag = torch.randn(shape, device=pilot_mask.device, dtype=real_dtype, generator=generator)
        symbols = torch.complex(real, imag) / math.sqrt(2)
    else:
        order = 2 if modulation == "qpsk" else 4
        indices = torch.randint(order, (2, *shape), device=pilot_mask.device, generator=generator)
        levels = (2 * indices + 1 - order).to(real_dtype)
        normalization = math.sqrt(2) if modulation == "qpsk" else math.sqrt(10)
        symbols = torch.complex(levels[0], levels[1]) / normalization
    return symbols.masked_fill(pilot_mask.unsqueeze(0), 0).to(dtype)


@dataclass(frozen=True)
class UplinkCombPilotConfig:
    num_rx_antennas: int = 32
    num_subcarriers: int = 32
    num_ofdm_symbols: int = 16
    pilot_symbols: tuple[int, ...] = tuple(range(8))
    pilot_subcarriers: tuple[int, ...] = tuple(range(0, 32, 4))
    pilot_energy: float = 1.0

    def validate(self) -> None:
        if min(self.num_rx_antennas, self.num_subcarriers, self.num_ofdm_symbols) < 1:
            raise ValueError("Antenna, subcarrier and symbol counts must be positive")
        for name, indices, size in (
            ("pilot_symbols", self.pilot_symbols, self.num_ofdm_symbols),
            ("pilot_subcarriers", self.pilot_subcarriers, self.num_subcarriers),
        ):
            if not indices or len(indices) != len(set(indices)) or any(i < 0 or i >= size for i in indices):
                raise ValueError(f"{name} must be nonempty, unique and in range")
        if not math.isfinite(self.pilot_energy) or self.pilot_energy <= 0:
            raise ValueError("pilot_energy must be finite and positive")

    @property
    def pilot_resource_fraction(self) -> float:
        return len(self.pilot_symbols) * len(self.pilot_subcarriers) / (
            self.num_ofdm_symbols * self.num_subcarriers
        )


def _uplink_channels(
    channels: torch.Tensor, config: UplinkCombPilotConfig, mapping: ChannelMapping
) -> torch.Tensor:
    if mapping not in ("native_uplink", "tdd_reciprocal_downlink"):
        raise ValueError("Declare channel_mapping as native_uplink or tdd_reciprocal_downlink")
    h = torch.as_tensor(channels)
    if h.ndim == 4:
        if mapping != "tdd_reciprocal_downlink" or h.shape[1] != 1:
            raise ValueError("Official [B,1,M,F] data require tdd_reciprocal_downlink")
        h = h[:, 0]
    if h.ndim != 3 or tuple(h.shape[1:]) != (config.num_rx_antennas, config.num_subcarriers):
        raise ValueError(
            f"Expected complex [B,{config.num_rx_antennas},{config.num_subcarriers}]"
            " or official [B,1,M,F] channels"
        )
    if not torch.is_complex(h) or h.dtype not in (torch.complex64, torch.complex128):
        raise TypeError("channels must be complex64 or complex128")
    if len(h) == 0 or not torch.isfinite(h).all():
        raise ValueError("channels must be nonempty and finite")
    return h


def uplink_pilot_grid(
    config: UplinkCombPilotConfig = UplinkCombPilotConfig(),
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Known scalar UE pilots P[T,F] and mask[T,F]."""
    config.validate()
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError("Pilot dtype must be complex64 or complex128")
    pilots = torch.zeros((config.num_ofdm_symbols, config.num_subcarriers), device=device, dtype=dtype)
    mask = torch.zeros_like(pilots, dtype=torch.bool)
    times = torch.tensor(config.pilot_symbols, device=device)
    freqs = torch.tensor(config.pilot_subcarriers, device=device)
    mask[times[:, None], freqs[None, :]] = True
    pilots[mask] = math.sqrt(config.pilot_energy)
    return pilots, mask


@torch.no_grad()
def estimate_uplink_pilot_reference_power(
    train_channels: torch.Tensor,
    *,
    channel_mapping: ChannelMapping,
    config: UplinkCombPilotConfig = UplinkCombPilotConfig(),
    batch_size: int = 4096,
) -> float:
    """Train-only average noiseless pilot power per BS receive antenna/RE."""
    if batch_size < 1 or len(train_channels) == 0:
        raise ValueError("Need positive batch_size and nonempty training channels")
    total, count = 0.0, 0
    for start in range(0, len(train_channels), batch_size):
        h = _uplink_channels(train_channels[start : start + batch_size], config, channel_mapping)
        pilot_h = h[..., list(config.pilot_subcarriers)]
        total += pilot_h.abs().square().sum(dtype=torch.float64).item()
        count += pilot_h.numel()
    reference = config.pilot_energy * total / count
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("Training pilot power must be finite and positive")
    return reference


def simulate_uplink_received_data(
    channels: torch.Tensor,
    *,
    channel_mapping: ChannelMapping,
    config: UplinkCombPilotConfig = UplinkCombPilotConfig(),
    snr_db: float | None = None,
    reference_power: float | None = None,
    noise_variance: float | None = None,
    data_modulation: str = "qpsk",
    data_symbols: torch.Tensor | None = None,
    seed: int | None = None,
) -> dict[str, torch.Tensor | float | str]:
    """Generate complex Y[B,M,T,F] without exposing H or unknown UE data."""
    config.validate()
    h = _uplink_channels(channels, config, channel_mapping)
    if (snr_db is None) == (noise_variance is None):
        raise ValueError("Specify exactly one of snr_db or noise_variance")
    if snr_db is not None:
        if reference_power is None or not math.isfinite(reference_power) or reference_power <= 0:
            raise ValueError("snr_db requires positive TRAIN-split reference_power")
        if not math.isfinite(snr_db):
            raise ValueError("snr_db must be finite")
        noise_variance = reference_power * 10 ** (-snr_db / 10)
    if noise_variance is None or not math.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("noise_variance must be finite and nonnegative")

    pilots, mask = uplink_pilot_grid(config, device=h.device, dtype=h.dtype)
    if data_symbols is None:
        symbols = generate_data_symbols(len(h), mask, modulation=data_modulation, dtype=h.dtype, seed=seed)
    else:
        symbols = torch.as_tensor(data_symbols, device=h.device)
        if symbols.shape != (len(h), config.num_ofdm_symbols, config.num_subcarriers):
            raise ValueError("data_symbols must have shape [B,T,F]")
        if not torch.is_complex(symbols) or not torch.isfinite(symbols).all():
            raise ValueError("data_symbols must be finite complex values")
        if torch.count_nonzero(symbols[:, mask]):
            raise ValueError("data_symbols must be zero on pilot elements")
        symbols = symbols.to(dtype=h.dtype)
    x = pilots.unsqueeze(0) + symbols
    y = h[:, :, None, :] * x[:, None, :, :]
    if noise_variance:
        generator = None if seed is None else torch.Generator(device=h.device).manual_seed(int(seed) + 1)
        real = torch.randn(y.shape, device=h.device, dtype=h.real.dtype, generator=generator)
        imag = torch.randn(y.shape, device=h.device, dtype=h.real.dtype, generator=generator)
        y = y + torch.complex(real, imag) * math.sqrt(noise_variance / 2)
    return {
        "y": y,
        "pilot_y": y.masked_fill((~mask)[None, None, :, :], 0),
        "pilots": pilots,
        "mask": mask,
        "data_mask": ~mask,
        "pilot_symbols": torch.tensor(config.pilot_symbols, device=h.device),
        "pilot_subcarriers": torch.tensor(config.pilot_subcarriers, device=h.device),
        "pilot_resource_fraction": config.pilot_resource_fraction,
        "data_modulation": data_modulation if data_symbols is None else "provided",
        "protocol": "single_user_uplink_bs_rx_array",
        "channel_mapping": channel_mapping,
    }


def uplink_comb_ls(
    observation: dict[str, torch.Tensor | float | str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average known pilots to get LS H[B,M,Fp] on observed comb tones."""
    pilot_y = observation["pilot_y"]
    pilots = observation["pilots"]
    mask = observation["mask"]
    freqs = observation["pilot_subcarriers"]
    if not isinstance(pilot_y, torch.Tensor) or pilot_y.ndim != 4:
        raise ValueError("pilot_y must be [B,M,T,F]")
    if not isinstance(pilots, torch.Tensor) or not isinstance(mask, torch.Tensor) or not isinstance(freqs, torch.Tensor):
        raise TypeError("observation is missing known pilot tensors")
    weighted = pilot_y * pilots.conj()[None, None, :, :]
    numerator = weighted.sum(dim=2).index_select(-1, freqs)
    denominator = pilots.abs().square().sum(dim=0).index_select(0, freqs)
    if torch.any(denominator <= 0):
        raise ValueError("Each pilot subcarrier needs nonzero pilot energy")
    return numerator / denominator[None, None, :], freqs


def observation(h: torch.Tensor, reference_power: float,
                config: UplinkCombPilotConfig, snr_db: float, seed: int) -> dict:
    return simulate_uplink_received_data(
        h, channel_mapping="tdd_reciprocal_downlink", config=config,
        snr_db=snr_db, reference_power=reference_power,
        data_modulation="qpsk", seed=seed,
    )


def obs(h, context, snr, seed):
    return observation(h, context['power'], UplinkCombPilotConfig(**context['protocol']), float(snr), int(seed))
