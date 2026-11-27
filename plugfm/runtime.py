"""Run a saved case or use a deployed student with complex Y[B, M, T, F]."""
from pathlib import Path

import numpy as np
import torch

from .data import ADAPTED_METHODS, SNRS, STUDENT_METHODS
from .heads import LinearTaskHead, ResidualCSIRefiner
from .joint import prediction
from .lora import NativeEncoder
from .model import CompactConfig, ContinuousCalibration
from .observation import obs
from .teachers import Teacher, decode, head_for
from .training import student_for
from .utils import metrics, seed_everything, transform


def load(path):
    return torch.load(Path(path), map_location='cpu', weights_only=True)


def load_linear(state, device):
    head = LinearTaskHead(state['mean'], state['std'], state['linear.weight'].shape[0]).to(device)
    head.load_state_dict(state, strict=True)
    return head.eval().requires_grad_(False)


def case_path(root, wfm, seed, method, samples):
    return Path(root) / wfm / f'seed{seed}' / f'{method}_n{samples}.pt'


class Predictor:
    def __init__(self, bundle, teacher_path, source, device):
        self.bundle = bundle
        self.method = bundle['method']
        self.config = CompactConfig(**bundle['config'])
        self.cal = ContinuousCalibration(bundle['calibration'], device)
        self.context = {'protocol': bundle['protocol'], 'power': self.cal.power}
        self.heads = {}
        if self.method in STUDENT_METHODS:
            self.student = student_for(self.method, self.config, self.cal, device)
            self.student.load_state_dict(bundle['student'], strict=True)
            self.student.eval().requires_grad_(False)
            self.heads['csi'] = ResidualCSIRefiner(self.student.representation_dim, self.config.channel_rms, 24).to(device)
        elif self.method == 'coarse_teacher':
            self.teacher = self.get_teacher(teacher_path, source, device)
            self.heads['csi'] = head_for(self.teacher)
        elif self.method in ADAPTED_METHODS:
            self.adapters = {}
            for task, state in bundle['adapted'].items():
                teacher = self.get_teacher(teacher_path, source, device)
                net = NativeEncoder(teacher)
                seed_everything(bundle['seed'] + 4123)
                net.configure(self.method, rank=4, alpha=4.)
                net.load_learned(state['encoder'])
                if self.method == 'lora_teacher':
                    net.merge_lora()
                head = head_for(teacher) if task == 'csi' else load_linear(state['head'], device)
                head.load_state_dict(state['head'], strict=True)
                self.adapters[task] = teacher, net.eval().requires_grad_(False), head.eval().requires_grad_(False)
            return
        else:
            raise ValueError(f'This release does not support running checkpoints for {self.method}.')
        self.heads['csi'].load_state_dict(bundle['heads']['csi'], strict=True)
        self.heads['csi'].eval().requires_grad_(False)
        for task in ('los', 'beam'):
            self.heads[task] = load_linear(bundle['heads'][task], device)

    def get_teacher(self, teacher_path, source, device):
        teacher = Teacher(self.bundle['wfm'], teacher_path, source, device)
        if teacher.checkpoint_sha256 != self.bundle['teacher_sha256']:
            raise ValueError('The teacher checkpoint does not match this saved case.')
        return teacher

    @torch.no_grad()
    def __call__(self, received, snr, clean_h=None):
        if self.method in STUDENT_METHODS:
            out = self.student(received, snr, return_alignment=False)
            h, features = out['refined_channel'], out['embedding']
            csi = self.heads['csi'](h, features, self.cal.mask, h)
        elif self.method in ADAPTED_METHODS:
            h = clean_h if self.method == 'perfect_teacher' else self.cal.estimate(received, snr)
            ctx = {'normalization': self.bundle['zca']}
            return {task: prediction(teacher, net, head, task, h, ctx)
                    for task, (teacher, net, head) in self.adapters.items()}
        else:
            h = self.cal.estimate(received, snr)
            tokens = self.teacher.tokens(h)
            csi = decode(self.teacher, self.heads['csi'], tokens)
            features = transform(tokens.mean(1), self.bundle['zca'])
        return {'csi': csi, 'los': self.heads['los'](features), 'beam': self.heads['beam'](features)}


def evaluate_case(predictor, test, device, batch_size=128, repeats=3, snrs=SNRS):
    seed = predictor.bundle['seed']
    htest, los, beam = (test[name] for name in ('h', 'los', 'beam'))
    energy = htest.abs().square().sum((1, 2))
    perfect = predictor.method == 'perfect_teacher'
    rows = []
    for snr in (0,) if perfect else snrs:
        observed = []
        for repeat in range(1 if perfect else repeats):
            errors, los_logits, beam_logits = [], [], []
            for offset in range(0, len(htest), batch_size):
                h = htest[offset:offset + batch_size].to(device)
                # Use the same data/noise seed in every case at this SNR.
                received = None if perfect else obs(h, predictor.context, snr, seed + 80_000_000 + repeat * 100_000_000 + offset)
                out = predictor(received, snr, clean_h=h if perfect else None)
                if not perfect:
                    errors.append((out['csi'] - h).abs().square().sum((1, 2)).cpu())
                los_logits.append(out['los'].cpu())
                beam_logits.append(out['beam'].cpu())
            error = energy if perfect else torch.cat(errors)
            observed.append(metrics(error.numpy(), energy.numpy(), torch.cat(los_logits).numpy(),
                                    torch.cat(beam_logits).numpy(), los.numpy(), beam.numpy()))
        keys = ('los_auc', 'beam_top1') if perfect else ('csi_nmse_db', 'los_auc', 'beam_top1')
        mean = {key: float(np.mean([row[key] for row in observed])) for key in keys}
        rows.append(dict(wfm=predictor.bundle['wfm'], method=predictor.method, seed=seed, snr_db=snr,
                         csi_nmse_db=None, **{key: mean[key] for key in keys if key != 'csi_nmse_db'}))
        if not perfect:
            rows[-1]['csi_nmse_db'] = mean['csi_nmse_db']
    if perfect:
        return [dict(rows[0], snr_db=snr) for snr in snrs]
    return rows
