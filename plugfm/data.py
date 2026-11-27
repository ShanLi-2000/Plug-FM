"""Rebuild the in-scene Centro split from the downloaded NPZ."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .model import CompactConfig, ContinuousCalibration, NoiseCalibration
from .observation import UplinkCombPilotConfig, estimate_uplink_pilot_reference_power
from .utils import full_transform, nested_ids, read, sha, transform, user_split


ROOT = Path(__file__).resolve().parents[1]
SCENE = read(ROOT / 'data/scene.json')
DISTILLATION_SAMPLES = 2048
STUDENT_METHODS = ('mse_csi', 'pilot_only_csi', 'feature_mse_only')
ADAPTED_METHODS = ('perfect_teacher', 'lora_teacher', 'full_transfer')
SNRS = tuple(range(-20, 21, 5))
LAYOUT = UplinkCombPilotConfig(pilot_symbols=tuple(range(8)), pilot_subcarriers=(0, 8, 16, 24))


def read_scene(path, smoke=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Cannot find {path}. Download the Centro NPZ as described in data/README.md.')
    if sha(path) != SCENE['sha256']:
        raise ValueError('The NPZ does not match the benchmark checksum in data/scene.json.')
    with np.load(path, allow_pickle=False) as source:
        h, los, beam = source['channels'], source['los'].reshape(-1), source['beam_labels'][:, 0]
        if h.ndim == 4 and h.shape[1] == 1:
            h = h[:, 0]
        if h.ndim != 3 or h.shape[1:] != (32, 32) or not np.iscomplexobj(h):
            raise ValueError('Expected complex H[N, 32, 32] after squeezing the UE antenna axis.')
        ids = user_split(los, beam, source['rx_pos'], SCENE['split_seed'], SCENE['split_counts'])
        power = estimate_uplink_pilot_reference_power(torch.from_numpy(h[ids['train']]).to(torch.complex64),
                                                    channel_mapping='tdd_reciprocal_downlink')
        partitions = {}
        for name, index in ids.items():
            if smoke:
                index = index[:64 if name == 'train' else 32]
            partitions[name] = {'h': torch.from_numpy(h[index]).to(torch.complex64),
                                'los': torch.from_numpy(los[index]).long(),
                                'beam': torch.from_numpy(beam[index]).long(), 'ids': index}
    # The original experiment fixes the reference at its training-only value
    # before changing the pilot grid, so the noise scale is shared by all cases.
    return partitions, {'power': power, 'protocol': asdict(LAYOUT)}


def prepare_training(teacher, parts, observation, options):
    context = dict(observation, teacher_sha256=teacher.checkpoint_sha256)
    for split in ('train', 'val'):
        context.update({f'{split}_{key}': parts[split][key] for key in ('h', 'los', 'beam')})
        context[f'raw_{split}_t'] = teacher.features(parts[split]['h'], options.batch_size)
    ids = nested_ids(len(context['train_h']), options.count, options.seed)
    distill_h = context['train_h'][ids]
    norm = full_transform(context['raw_train_t'][ids])
    context.update(distill_h=distill_h, normalization=norm,
                   distill_t=transform(context['raw_train_t'][ids], norm))
    # Fit teacher whitening and the LMMSE mean/covariance on the distillation subset.
    fitted = NoiseCalibration.fit(distill_h, context['protocol'], context['power'])
    calibration = ContinuousCalibration(fitted.state, options.device)
    config = CompactConfig(width=8 if options.smoke else 32, depth=1, embedding_dim=teacher.feature_dim,
                           data_mode='moment', channel_rms=float(distill_h.abs().square().mean().sqrt()),
                           noise_variance=calibration.variance(0), representation_mode='pooled', global_attention=True)
    return context, calibration, config
