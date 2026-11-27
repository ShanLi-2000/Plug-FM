"""Task-specific LoRA and full encoder/head fine-tuning used by the main comparison."""
import gc
import math

import numpy as np
import torch
from torch.nn import functional as F

from .heads import LinearTaskHead
from .lora import NativeEncoder
from .observation import obs
from .teachers import head_for, decode
from .training import event
from .utils import parameter_count, reference_order, save, schedule, seed_everything, state_cpu, transform, write

SNRS = tuple(range(-20, 21, 5))


def make_head(a, teacher, task, norm):
    if task == 'csi':
        seed_everything(a.seed + 7321)
        return head_for(teacher)
    seed_everything(a.seed + 8321)
    # Match the original los-then-beam initialization ordering.
    heads = {k: LinearTaskHead(norm['mean'], norm['std'], n).to(a.device) for k, n in (('los', 2), ('beam', 16))}
    return heads[task]


def prediction(teacher, net, head, task, h, ctx):
    z = net(h)
    if task == 'csi':
        return decode(teacher, head, z)
    return head(transform(z.mean(1), ctx['normalization']))


def received_input(a, ctx, cal, h, snr, noise_seed, method):
    if method == 'perfect_teacher':
        return h
    return cal.estimate(obs(h, ctx, snr, noise_seed), snr)


@torch.no_grad()
def validation(a, teacher, net, head, task, method, ctx, cal, clean_features):
    net.eval()
    head.eval()
    if method == 'perfect_teacher' and task != 'csi':
        return float(F.cross_entropy(head(clean_features['val'].to(a.device)), ctx['val_' + task].to(a.device))), []
    values = []
    for snr in (0,) if method == 'perfect_teacher' else SNRS:
        total, energy, count = 0., 0., 0
        for offset in range(0, len(ctx['val_h']), a.batch_size):
            h = ctx['val_h'][offset:offset + a.batch_size].to(a.device)
            noise_seed = a.seed + (60000000 if task == 'csi' else 70000000) + offset
            x = received_input(a, ctx, cal, h, snr, noise_seed, method)
            out = prediction(teacher, net, head, task, x, ctx)
            if task == 'csi':
                total += float((out - h).abs().square().sum())
                energy += float(h.abs().square().sum())
            else:
                target = ctx['val_' + task][offset:offset + len(h)].to(a.device)
                total += float(F.cross_entropy(out, target, reduction='sum'))
                count += len(h)
        values.append(10 * math.log10(max(total / energy, 1e-40)) if task == 'csi' else total / count)
    return float(np.mean(values)), values


def train_task(a, teacher, ctx, cal, method, task, dest, coarse_norm):
    kind = teacher.kind
    if (dest / 'training.json').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    event(a, stage='fit', wfm=kind, method=method, task=task)
    net = NativeEncoder(teacher).to(a.device)
    original_digest = net.frozen_digest()
    seed_everything(a.seed + 4123)
    net.configure(method, a.rank, a.alpha)
    frozen_before = net.frozen_digest()
    clean_features = None
    norm = None
    if task != 'csi':
        if method == 'perfect_teacher':
            clean_features = {key: transform(ctx['raw_' + key + '_t'], ctx['normalization'])
                              for key in ('train', 'val')}
            x = clean_features['train']
            norm = {'mean': x.mean(0), 'std': x.std(0, unbiased=False)}
        else:
            norm = coarse_norm
    head = make_head(a, teacher, task, norm)
    encoder_parameters = net.training_parameters()
    encoder_trainable_count = sum(p.numel() for p in encoder_parameters)
    head_lr = 3e-4 if task == 'csi' else 1e-3
    encoder_lr = a.lora_lr if method == 'lora_teacher' else a.full_lr
    groups = [{'params': list(head.parameters()), 'lr': head_lr}]
    if encoder_parameters:
        groups.append({'params': encoder_parameters, 'lr': encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    epochs = a.csi_epochs if task == 'csi' else a.class_epochs
    start, best, history = 0, math.inf, []
    resume = dest / 'resume.pt'
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        net.model.load_state_dict(saved['encoder'], strict=False)
        head.load_state_dict(saved['head'])
        optimizer.load_state_dict(saved['optimizer'])
        start, best, history = saved['epoch'], saved['best'], saved['history']
        del saved
    for epoch in range(start, epochs):
        seed_everything(a.seed + (7321 if task == 'csi' else 8321) + epoch)
        adapt = method != 'perfect_teacher' and epoch >= a.warmup_epochs
        net.model.train(adapt)
        head.train()
        order_seed = a.seed + (29123 if task == 'csi' else 83123)
        order = reference_order(len(ctx['train_h']), a.batch_size, a.steps, order_seed, epoch)
        snrs = schedule(a.steps, a.seed + (1000 if task == 'csi' else 2000), epoch)
        loss_sum = 0.
        for step, ids in enumerate(order):
            optimizer.zero_grad(set_to_none=True)
            if clean_features is not None:
                x = clean_features['train'][ids].to(a.device)
                target = ctx['train_' + task][ids].to(a.device)
                loss = F.cross_entropy(head(x), target)
                loss.backward()
                loss_sum += float(loss.detach())
            else:
                h = ctx['train_h'][ids].to(a.device)
                noise_seed = a.seed + (29000 if task == 'csi' else 72000000) + epoch * 100003 + step
                with torch.no_grad():
                    x = received_input(a, ctx, cal, h, snrs[step], noise_seed, method)
                denominator = h.abs().square().sum().clamp_min(1e-8)
                if adapt:
                    out = prediction(teacher, net, head, task, x, ctx)
                else:
                    # The head sees frozen encoder features during warm-up.
                    with torch.no_grad():
                        z = net(x)
                        features = transform(z.mean(1), ctx['normalization']) if task != 'csi' else None
                    out = decode(teacher, head, z) if task == 'csi' else head(features)
                if task == 'csi':
                    loss = (out - h).abs().square().sum() / denominator
                else:
                    target = ctx['train_' + task][ids].to(a.device)
                    loss = F.cross_entropy(out, target, reduction='sum') / len(h)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite teacher adaptation loss')
                loss.backward()
                loss_sum += float(loss.detach())
            if adapt:
                torch.nn.utils.clip_grad_norm_(encoder_parameters, 1.)
            if task == 'csi':
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
        score, per_snr = validation(a, teacher, net, head, task, method, ctx, cal, clean_features)
        history.append({'epoch': epoch + 1, 'training_loss': loss_sum / a.steps, 'validation': score,
                        'per_snr_validation': per_snr, 'encoder_adapting': adapt, 'sampled_snrs_db': snrs})
        if score < best:
            best = score
            save(dest / 'best.pt', {'encoder': net.learned_state(), 'head': state_cpu(head),
                                    'epoch': epoch + 1, 'validation': score, 'method': method,
                                    'teacher_sha256': teacher.checkpoint_sha256})
        # Only the active task retains a large optimizer checkpoint.
        save(resume, {'encoder': net.encoder_state() if method == 'full_transfer' else net.learned_state(),
                      'head': state_cpu(head), 'optimizer': optimizer.state_dict(),
                      'epoch': epoch + 1, 'best': best, 'history': history})
        event(a, stage='task_training', wfm=kind, method=method, task=task,
              epoch=epoch + 1, epochs=epochs, validation=score)
    if method in ('perfect_teacher', 'lora_teacher'):
        assert net.frozen_digest() == frozen_before
    best_state = torch.load(dest / 'best.pt', map_location='cpu', weights_only=True)
    net.load_learned(best_state['encoder'])
    head.load_state_dict(best_state['head'])
    net.eval()
    head.eval()
    merger_verified = False
    if method == 'lora_teacher':
        with torch.no_grad():
            h = ctx['val_h'][:2].to(a.device)
            x = cal.estimate(obs(h, ctx, 0, a.seed + 992), 0)
            expected = net(x)
            net.merge_lora()
            torch.testing.assert_close(net(x), expected, rtol=3e-5, atol=3e-5)
        merger_verified = True
    assert net.encoder_count() == teacher.encoder_parameters
    write(dest / 'training.json', {
        'kind': kind, 'method': method, 'task': task, 'history': history,
        'epochs': epochs, 'warmup_epochs': 0 if method == 'perfect_teacher' else a.warmup_epochs,
        'head_train_users': len(ctx['train_h']), 'validation_users': len(ctx['val_h']),
        'deployment_encoder_parameters': teacher.encoder_parameters,
        'encoder_trainable_parameters': encoder_trainable_count, 'head_parameters': parameter_count(head),
        'head_lr': head_lr, 'encoder_lr': 0 if method == 'perfect_teacher' else encoder_lr,
        'logical_batch': a.batch_size, 'updates_per_epoch': a.steps,
        'global_NMSE_denominator_preserved': True, 'rank': a.rank, 'alpha': a.alpha,
        'lora_targets': net.lora_targets, 'lora_merge_verified': merger_verified,
        'base_frozen_verified': method in ('perfect_teacher', 'lora_teacher'),
        'best_epoch': best_state['epoch'], 'task_specific_encoder': method != 'perfect_teacher',
        'original_encoder_digest': original_digest,
        'input': 'ground-truth CSI' if method == 'perfect_teacher' else 'pilot-derived LMMSE CSI',
        'selection': 'clean validation' if method == 'perfect_teacher' else 'nine-SNR validation macro task loss',
        'no_full_token_bank': True, 'teacher_pretraining_checkpoint_unchanged': True})
    # Generated temporary state only; best weights and all histories survive.
    if resume.exists():
        resume.unlink()
    del teacher, ctx, cal, net, head, optimizer, best_state, clean_features
    gc.collect()
    torch.cuda.empty_cache()
