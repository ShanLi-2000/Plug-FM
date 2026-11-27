"""Reproducible sampling, metrics, and checkpoint I/O."""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, temp)
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def lock(path, value):
    if Path(path).exists() and read(path) != value:
        raise ValueError(f'Protocol changed: {path}; use another output directory')
    write(path, value)


def nested_ids(total, count, seed):
    if not 0 < count <= total:
        raise ValueError('Invalid number of distinct training users')
    permutation = np.random.default_rng(20261029 + seed * 101).permutation(total)
    return torch.from_numpy(np.sort(permutation[:count])).long()


def reference_order(count, size, steps, observation_seed, epoch):
    """Cycle independently shuffled users; match legacy randperm at full N."""
    if count < 1 or min(size, steps) < 1:
        raise ValueError('Need positive count, batch size and update count')
    generator = torch.Generator().manual_seed(observation_seed + epoch * 10501)
    blocks, length = [], 0
    while length < size * steps:
        blocks.append(torch.randperm(count, generator=generator))
        length += count
    return torch.cat(blocks)[:size * steps].reshape(steps, size)


def acquire_run_lock(folder):
    """An OS-held lock is released even when a training process is interrupted."""
    folder.mkdir(parents=True, exist_ok=True)
    stream = (folder / '.runner.lock').open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError('Another process already holds this experiment directory') from None
    return stream


def state_cpu(module: torch.nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def full_transform(raw, ridge=1e-5):
    """Invertible regularized ZCA: keep all D coordinates, no rank truncation."""
    raw = raw.double()
    mean = raw.mean(0)
    x = raw - mean
    eigen, directions = torch.linalg.eigh(x.T @ x / len(x))
    floor = max(float(eigen[-1]) * ridge, 1e-10)
    matrix = (directions * (eigen.clamp_min(0) + floor).rsqrt()[None]) @ directions.T
    return {"mean": mean, "matrix": matrix, "ridge": floor,
            "raw_dimension": raw.shape[1], "eigenvalues": eigen.float()}


def transform(raw, norm):
    # Double precision avoids cancellation around the LayerNorm null direction;
    # outputs keep the model feature dtype and dimension.
    return ((raw.double() - norm["mean"].to(raw.device)) @ norm["matrix"].to(raw.device)).to(raw.dtype)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def neural_digest(model):
    import hashlib
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        digest.update(name.encode())
        digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cap_stratified(ids, strata, maximum, rng):
    """Proportional capped subsample with exact total; no model scores used."""
    if not maximum or len(ids) <= maximum:
        return np.sort(ids)
    groups = [ids[strata[ids] == category] for category in np.unique(strata[ids])]
    wanted = np.array([len(group) for group in groups], dtype=float) * maximum / len(ids)
    counts = np.floor(wanted).astype(int)
    order = np.argsort(-(wanted - counts), kind='stable')
    counts[order[:maximum - counts.sum()]] += 1
    return np.sort(np.concatenate([rng.choice(group, count, replace=False)
                                  for group, count in zip(groups, counts) if count]))


def user_split(los, beam, positions, seed, limits):
    los, beam, positions = np.asarray(los).reshape(-1), np.asarray(beam).reshape(-1), np.asarray(positions)
    if len(los) != len(beam) or len(positions) != len(los):
        raise ValueError('Inconsistent user metadata')
    if len(np.unique(positions, axis=0)) != len(positions):
        raise ValueError('Repeated user positions require a grouped split; random row splitting is forbidden')
    if not np.isfinite(positions).all() or not np.isin(los, [0, 1]).all() or not np.isin(beam, np.arange(16)).all():
        raise ValueError('Invalid positions/labels')
    rng = np.random.default_rng(seed)
    strata = los.astype(int) * 16 + beam.astype(int)
    parts = {name: [] for name in ('train', 'val', 'test')}
    for category in np.unique(strata):
        ids = rng.permutation(np.flatnonzero(strata == category))
        n = len(ids)
        nv = max(1, round(.15 * n)) if n >= 3 else 0
        nt = min(max(1, round(.15 * n)), n - nv - 1) if n >= 3 else 0
        parts['val'].extend(ids[:nv])
        parts['test'].extend(ids[nv:nv + nt])
        parts['train'].extend(ids[nv + nt:])
    result = {name: cap_stratified(np.asarray(ids, dtype=np.int64), strata, limits[name], rng)
              for name, ids in parts.items()}
    all_ids = np.concatenate(list(result.values()))
    if len(np.unique(all_ids)) != len(all_ids):
        raise AssertionError('User leakage across partitions')
    for name, ids in result.items():
        if len(ids) < 32 or len(np.unique(los[ids])) != 2:
            raise ValueError(f'{name} is too small or lacks a LOS class')
    return result


def metrics(errors, energy, los_logits, beam_logits, los, beam):
    """NMSE is total error energy / total signal energy; AUC handles ties."""
    errors, energy = np.asarray(errors, dtype=float), np.asarray(energy, dtype=float)
    score = np.asarray(los_logits)[:, 1] - np.asarray(los_logits)[:, 0]
    labels = np.asarray(los).astype(int)
    positive, negative = score[labels == 1], np.sort(score[labels == 0])
    auc = None
    if len(positive) and len(negative):
        lower = np.searchsorted(negative, positive, side='left')
        upper = np.searchsorted(negative, positive, side='right')
        auc = float((lower + .5 * (upper - lower)).sum() / (len(positive) * len(negative)))
    pred = np.asarray(los_logits).argmax(-1)
    ba = float(np.mean([(pred[labels == c] == c).mean() for c in (0, 1)]))
    ratio = float(errors.sum() / energy.sum())
    return {'csi_nmse': ratio, 'csi_nmse_db': float(10 * np.log10(max(ratio, 1e-40))),
            'los_auc': auc, 'los_balanced_accuracy': ba, 'los_accuracy': float((pred == labels).mean()),
            'beam_top1': float((np.asarray(beam_logits).argmax(-1) == np.asarray(beam)).mean()),
            'median_user_nmse_db': float(np.median(10 * np.log10(np.maximum(errors / energy, 1e-40))))}


def schedule(steps, seed, epoch):
    """One continuous SNR per minibatch, randomized stratification each epoch.

    Every step is marginally Uniform[-20,20]. Stratification avoids accidentally
    starving a region in a short epoch; no discretization to the test grid.
    """
    generator = torch.Generator().manual_seed(seed + 8127 + epoch * 7919)
    values = -20. + 40. * (torch.arange(steps, dtype=torch.float64) +
                          torch.rand(steps, dtype=torch.float64, generator=generator)) / steps
    return values[torch.randperm(steps, generator=generator)].tolist()
