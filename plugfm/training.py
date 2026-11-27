"""Student distillation and three downstream task readouts."""
import json
import math
import time
from dataclasses import asdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .heads import LinearTaskHead, ResidualCSIRefiner, nmse_loss
from .model import MixedStudent
from .observation import obs
from .teachers import decode, head_for
from .utils import parameter_count, reference_order, save, schedule, seed_everything, state_cpu, transform, write

TEST_SNRS = tuple(range(-20, 21, 5))


def student_for(method, config, calibration, device):
    return MixedStudent(config, calibration, pilot_only=method == 'pilot_only_csi').to(device)


class TeacherFeatureView(nn.Module):
    """Class heads see the mean of unmasked tokens, transformed by the training ZCA."""
    def __init__(self, teacher, calibration, normalization):
        super().__init__()
        self.teacher, self.calibration = teacher, calibration
        self.normalization = normalization

    @torch.no_grad()
    def forward(self, received, snr, return_alignment=False):
        channel = self.calibration.estimate(received, snr)
        tokens = self.teacher.tokens(channel)
        return {'embedding': transform(tokens.mean(1), self.normalization)}


@torch.no_grad()
def encode(method, model, teacher, calibration, context, received, snr):
    return model(received, snr, return_alignment=False)


def event(options, **values):
    values['time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    print(json.dumps(values, ensure_ascii=False), flush=True)
    write(options.output_dir / 'progress.json', values)


@torch.no_grad()
def representation_validation(model, context, options, seed):
    model.eval()
    values = []
    for snr in TEST_SNRS:
        error = energy = 0.
        for offset in range(0, len(context['val_h']), options.batch_size):
            h = context['val_h'][offset:offset + options.batch_size].to(options.device)
            received = obs(h, context, snr, seed + 50_000_000 + offset)
            pred = model(received, snr)['refined_channel']
            error += float((pred - h).abs().square().sum()); energy += float(h.abs().square().sum())
        values.append(10 * math.log10(max(error / energy, 1e-40)))
    return values


def train_representation(method, context, calibration, config, options, seed, folder):
    seed_everything(seed + 21)
    model = student_for(method, config, calibration, options.device)
    if method == 'feature_mse_only':
        # The zero-initialized internal CSI correction is unused by feature MSE.
        model.channel_head.requires_grad_(False)
    if (folder / 'representation.json').exists():
        model.load_state_dict(torch.load(folder / 'model.pt', map_location=options.device, weights_only=True)['model'])
        return model.eval().requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    start, history = 0, []
    resume = folder / 'representation_resume.pt'
    if resume.exists():
        old = torch.load(resume, map_location=options.device, weights_only=True)
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        start, history = old['epoch'], old['history']
    for epoch in range(start, options.epochs):
        seed_everything(seed + 21 + epoch); model.train()
        lr = 3e-4 if epoch < options.warm_epochs else 1e-4
        for group in optimizer.param_groups:
            group['lr'] = lr
        order = reference_order(options.count, options.batch_size, options.steps_per_epoch, seed + 29000, epoch)
        snrs = schedule(options.steps_per_epoch, seed, epoch)
        sums = np.zeros(3)
        for step, ids in enumerate(order):
            h = context['distill_h'][ids].to(options.device)
            received = obs(h, context, snrs[step], seed + 29_000 + epoch * 100_003 + step)
            out = model(received, snrs[step])
            feature = F.mse_loss(out['aligned_embedding'], context['distill_t'][ids].to(options.device))
            csi = nmse_loss(out['refined_channel'], h)
            loss = feature if method == 'feature_mse_only' else feature + options.csi_weight * csi
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite representation objective')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            sums += [float(loss.detach()), float(feature.detach()), float(csi.detach())]
        val = (representation_validation(model, context, options, seed)
               if (epoch + 1) % options.validate_every == 0 or epoch + 1 == options.epochs else None)
        row = {'epoch': epoch + 1, 'loss': (sums / len(order)).tolist(), 'lr': lr,
               'sampled_snrs_db': snrs, 'validation_db': val}
        history.append(row)
        candidate = {'model': state_cpu(model), 'config': asdict(config), 'epoch': epoch + 1,
                     'method': method, 'teacher_sha256': context['teacher_sha256']}
        save(folder / 'model.pt', candidate)
        save(resume, {'model': candidate['model'], 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
                      'history': history})
        event(options, stage='representation', seed=seed, method=method, epoch=epoch + 1,
              epochs=options.epochs, loss=row['loss'][0])
    model.load_state_dict(torch.load(folder / 'model.pt', map_location=options.device, weights_only=True)['model'])
    write(folder / 'representation.json', {'history': history, 'unique_channels': options.count,
          'training_snr_distribution': 'continuous Uniform[-20,20] dB', 'parameters': parameter_count(model),
          'selection': 'fixed final update'})
    del optimizer
    return model.eval().requires_grad_(False)


def csi_prediction(teacher, head, mode, out, received, coarse):
    channel = out['refined_channel']
    return head(channel, out['embedding'], received['mask'], channel)


def new_csi_heads(method, model, teacher, config, options):
    return {'student': ResidualCSIRefiner(model.representation_dim, config.channel_rms, 24).to(options.device)}


def train_csi(method, model, teacher, calibration, context, config, options, seed, folder):
    seed_everything(seed + 7321)
    heads = new_csi_heads(method, model, teacher, config, options)
    if (folder / 'csi_training.json').exists():
        for mode, head in heads.items():
            head.load_state_dict(torch.load(folder / f'csi_{mode}.pt', map_location=options.device, weights_only=True)['head'])
        return {key: head.eval().requires_grad_(False) for key, head in heads.items()}
    optimizers = {key: torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4) for key, head in heads.items()}
    start, best, history = 0, {key: math.inf for key in heads}, []
    resume = folder / 'csi_resume.pt'
    if resume.exists():
        old = torch.load(resume, map_location=options.device, weights_only=True)
        for key in heads:
            heads[key].load_state_dict(old['heads'][key]); optimizers[key].load_state_dict(old['optimizers'][key])
        start, best, history = old['epoch'], old['best'], old['history']
    for epoch in range(start, options.head_epochs):
        seed_everything(seed + 7321 + epoch)
        for head in heads.values():
            head.train()
        order = reference_order(len(context['train_h']), options.batch_size, options.head_steps, seed + 29123, epoch)
        snrs = schedule(options.head_steps, seed + 1000, epoch)
        for step, ids in enumerate(order):
            h = context['train_h'][ids].to(options.device)
            received = obs(h, context, snrs[step], seed + 29_000 + epoch * 100_003 + step)
            with torch.no_grad():
                out = encode(method, model, teacher, calibration, context, received, snrs[step])
                coarse = calibration.estimate(received, snrs[step]) if method.endswith('teacher') else None
            for key, head in heads.items():
                pred = csi_prediction(teacher, head, key, out, received, coarse)
                loss = nmse_loss(pred, h)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite CSI head objective')
                optimizers[key].zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); optimizers[key].step()
        for head in heads.values():
            head.eval()
        values = {key: [] for key in heads}
        with torch.no_grad():
            for snr in TEST_SNRS:
                error = {key: 0. for key in heads}; energy = 0.
                for offset in range(0, len(context['val_h']), options.batch_size):
                    h = context['val_h'][offset:offset + options.batch_size].to(options.device)
                    received = obs(h, context, snr, seed + 60_000_000 + offset)
                    out = encode(method, model, teacher, calibration, context, received, snr)
                    coarse = calibration.estimate(received, snr) if method.endswith('teacher') else None
                    for key, head in heads.items():
                        pred = csi_prediction(teacher, head, key, out, received, coarse)
                        error[key] += float((pred - h).abs().square().sum())
                    energy += float(h.abs().square().sum())
                for key in heads:
                    values[key].append(10 * math.log10(max(error[key] / energy, 1e-40)))
        scores = {key: float(np.mean(v)) for key, v in values.items()}
        history.append({'epoch': epoch + 1, 'validation_db': values, 'macro_db': scores, 'sampled_snrs_db': snrs})
        for key, head in heads.items():
            if scores[key] < best[key]:
                best[key] = scores[key]
                save(folder / f'csi_{key}.pt', {'head': state_cpu(head), 'epoch': epoch + 1, 'validation_db': best[key]})
        save(resume, {'heads': {key: state_cpu(h) for key, h in heads.items()},
                     'optimizers': {key: o.state_dict() for key, o in optimizers.items()},
                     'epoch': epoch + 1, 'best': best, 'history': history})
        event(options, stage='csi_head', method=method, seed=seed, epoch=epoch + 1,
              epochs=options.head_epochs, validation_db=scores)
    for key, head in heads.items():
        head.load_state_dict(torch.load(folder / f'csi_{key}.pt', map_location=options.device, weights_only=True)['head'])
    write(folder / 'csi_training.json', {'history': history, 'unique_train_users': len(context['train_h']),
          'parameters': {key: parameter_count(head) for key, head in heads.items()},
          'training_snr_distribution': 'continuous Uniform[-20,20] dB', 'encoder_frozen': True,
          'validation_snrs': TEST_SNRS, 'selection': 'nine-point validation macro CSI dB'})
    del optimizers
    return {key: head.eval().requires_grad_(False) for key, head in heads.items()}


@torch.no_grad()
def feature_normalization(method, model, teacher, calibration, context, options, seed):
    values = []
    n = len(context['train_h'])
    steps = math.ceil(n / options.batch_size)
    snrs = schedule(steps, seed + 3000, 0)
    for step, offset in enumerate(range(0, n, options.batch_size)):
        h = context['train_h'][offset:offset + options.batch_size].to(options.device)
        received = obs(h, context, snrs[step], seed + 71_000_000 + offset)
        values.append(encode(method, model, teacher, calibration, context, received, snrs[step])['embedding'].cpu())
    x = torch.cat(values)
    return x.mean(0), x.std(0, unbiased=False)


def train_class_heads(method, model, teacher, calibration, context, options, seed, folder):
    norm_path = folder / 'class_normalization.pt'
    if norm_path.exists():
        norm = torch.load(norm_path, map_location='cpu', weights_only=True)
    else:
        mean, std = feature_normalization(method, model, teacher, calibration, context, options, seed)
        norm = {'mean': mean, 'std': std}; save(norm_path, norm)
    seed_everything(seed + 8321)
    heads = {task: LinearTaskHead(norm['mean'], norm['std'], 2 if task == 'los' else 16).to(options.device)
             for task in ('los', 'beam')}
    if (folder / 'class_training.json').exists():
        for task, head in heads.items():
            head.load_state_dict(torch.load(folder / f'{task}.pt', map_location=options.device, weights_only=True)['head'])
        return {key: head.eval().requires_grad_(False) for key, head in heads.items()}
    optimizers = {task: torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4) for task, head in heads.items()}
    start, best, history = 0, {key: math.inf for key in heads}, []
    resume = folder / 'class_resume.pt'
    if resume.exists():
        old = torch.load(resume, map_location=options.device, weights_only=True)
        for key in heads:
            heads[key].load_state_dict(old['heads'][key]); optimizers[key].load_state_dict(old['optimizers'][key])
        start, best, history = old['epoch'], old['best'], old['history']
    # Only validation features may be cached: training observations and continuous
    # SNRs are freshly generated every step, not cached at nine discrete SNRs.
    val_x = []
    with torch.no_grad():
        for snr in TEST_SNRS:
            parts = []
            for offset in range(0, len(context['val_h']), options.batch_size):
                h = context['val_h'][offset:offset + options.batch_size].to(options.device)
                received = obs(h, context, snr, seed + 70_000_000 + offset)
                parts.append(encode(method, model, teacher, calibration, context, received, snr)['embedding'].cpu())
            val_x.append(torch.cat(parts).to(options.device))
    for epoch in range(start, options.class_epochs):
        seed_everything(seed + 8321 + epoch)
        order = reference_order(len(context['train_h']), options.batch_size, options.head_steps, seed + 83123, epoch)
        snrs = schedule(options.head_steps, seed + 2000, epoch)
        for step, ids in enumerate(order):
            h = context['train_h'][ids].to(options.device)
            with torch.no_grad():
                received = obs(h, context, snrs[step], seed + 72_000_000 + epoch * 100_003 + step)
                x = encode(method, model, teacher, calibration, context, received, snrs[step])['embedding']
            for task, head in heads.items():
                loss = F.cross_entropy(head(x), context['train_' + task][ids].to(options.device))
                optimizers[task].zero_grad(set_to_none=True); loss.backward(); optimizers[task].step()
        with torch.no_grad():
            scores = {task: float(np.mean([float(F.cross_entropy(head(x), context['val_' + task].to(options.device)))
                                           for x in val_x])) for task, head in heads.items()}
        history.append({'epoch': epoch + 1, 'validation_ce': scores, 'sampled_snrs_db': snrs})
        for task, head in heads.items():
            if scores[task] < best[task]:
                best[task] = scores[task]
                save(folder / f'{task}.pt', {'head': state_cpu(head), 'epoch': epoch + 1, 'validation_ce': scores[task]})
        save(resume, {'heads': {key: state_cpu(h) for key, h in heads.items()},
                     'optimizers': {key: o.state_dict() for key, o in optimizers.items()},
                     'epoch': epoch + 1, 'best': best, 'history': history})
        if (epoch + 1) % 10 == 0 or epoch + 1 == options.class_epochs:
            event(options, stage='class_heads', seed=seed, method=method, epoch=epoch + 1,
                  epochs=options.class_epochs, validation_ce=scores)
    for task, head in heads.items():
        head.load_state_dict(torch.load(folder / f'{task}.pt', map_location=options.device, weights_only=True)['head'])
    write(folder / 'class_training.json', {'history': history, 'unique_train_users': len(context['train_h']),
          'parameters': {task: parameter_count(head) for task, head in heads.items()},
          'training_snr_distribution': 'continuous Uniform[-20,20] dB',
          'fresh_training_observations_every_step': True, 'encoder_frozen': True})
    del optimizers, val_x
    return {key: head.eval().requires_grad_(False) for key, head in heads.items()}


def fit_teacher_head(a, layout, t, cal, ctx, seed, folder):
    seed_everything(seed+7321);head=head_for(t)
    done=folder/'csi_training.json'
    if done.exists():return
    optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-4)
    start=0;best=math.inf;history=[];resume=folder/'csi_resume.pt'
    if resume.exists():
        s=torch.load(resume,map_location=a.device,weights_only=True)
        head.load_state_dict(s['head']);optimizer.load_state_dict(s['optimizer'])
        start,best,history=s['epoch'],s['best'],s['history']
    def prediction(h,snr,noise_seed):
        with torch.no_grad():
            received=obs(h,ctx,snr,noise_seed);coarse=cal.estimate(received,snr)
            tokens=t.tokens(coarse)
        return decode(t,head,tokens)
    for epoch in range(start,a.head_epochs):
        seed_everything(seed+7321+epoch);head.train()
        order=reference_order(len(ctx['train_h']),a.batch_size,a.head_steps,seed+29123,epoch)
        snrs=schedule(a.head_steps,seed+1000,epoch)
        for step,ids in enumerate(order):
            h=ctx['train_h'][ids].to(a.device)
            pred=prediction(h,snrs[step],seed+29000+epoch*100003+step)
            loss=nmse_loss(pred,h)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite teacher head loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
        head.eval();values=[]
        with torch.no_grad():
            for snr in TEST_SNRS:
                error=energy=0.
                for offset in range(0,len(ctx['val_h']),a.batch_size):
                    h=ctx['val_h'][offset:offset+a.batch_size].to(a.device)
                    pred=prediction(h,snr,seed+60000000+offset)
                    error+=float((pred-h).abs().square().sum());energy+=float(h.abs().square().sum())
                values.append(10*math.log10(error/energy))
        score=float(np.mean(values));history.append({'epoch':epoch+1,'per_snr_validation_db':values,'macro_db':score})
        if score<best:
            best=score;save(folder/'csi_direct.pt',{'head':state_cpu(head),'epoch':epoch+1,'validation_db':score})
        save(resume,{'head':state_cpu(head),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'best':best,'history':history})
        event(a,stage='teacher_csi_head',layout=layout,wfm=t.kind,seed=seed,method='coarse_teacher',epoch=epoch+1,validation_db=score)
    assert all(p.grad is None for p in t.model.parameters())
    write(done,{'history':history,'head_parameters':parameter_count(head),'teacher_frozen':True,
        'readout':'native/paper-adapted direct, no CSI residual skip',
        'train_users':len(ctx['train_h']),'training_snrs':'continuous Uniform[-20,20] dB'})
    del head,optimizer
