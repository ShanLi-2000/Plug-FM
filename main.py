"""Plug-FM command-line and PyCharm entry point."""
import argparse
import csv
import os
from pathlib import Path

import settings

ROOT = Path(__file__).resolve().parent
METHODS = ('mse_csi', 'coarse_teacher', 'perfect_teacher', 'lora_teacher', 'full_transfer',
           'pilot_only_csi', 'feature_mse_only')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('demo', 'train', 'evaluate', 'plot', 'check'), nargs='?', default=settings.ACTION)
    p.add_argument('--wfm', choices=('contrawimae', 'lwm'), default=settings.WFM)
    p.add_argument('--method', choices=METHODS, default=settings.METHOD)
    p.add_argument('--seed', type=int, default=settings.SEED)
    p.add_argument('--device', default=settings.DEVICE)
    p.add_argument('--checkpoint', type=Path, default=settings.CHECKPOINT)
    p.add_argument('--teacher-checkpoint', type=Path, help='Override the bundled cleaned teacher checkpoint.')
    p.add_argument('--scene-file', type=Path, help='Path to the Centro NPZ listed in data/scene.json.')
    p.add_argument('--contra-source', type=Path, default=ROOT / 'external/ContraWiMAE')
    p.add_argument('--lwm-source', type=Path, default=ROOT / 'external/LWM')
    p.add_argument('--output-dir', type=Path, help='Generated training, evaluation or figure directory.')
    p.add_argument('--plot-group', choices=('joint',), default=settings.PLOT_GROUP)
    p.add_argument('--no-std', action='store_true', help='Hide seed standard-deviation bands.')
    p.add_argument('--smoke', action='store_true', default=settings.SMOKE)
    return p.parse_args()


def checkpoint_inputs(args, wfm):
    source = args.contra_source if wfm == 'contrawimae' else args.lwm_source
    return args.teacher_checkpoint or ROOT / 'checkpoints/teachers' / f'{wfm}.pt', source


def main():
    args = parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    from plugfm.utils import read, sha, write

    if args.action == 'check':
        manifest = read(ROOT / 'checkpoints/MANIFEST.json')
        for entry in manifest['files']:
            path = ROOT / entry['file']
            if not path.is_file() or sha(path) != entry['sha256']:
                raise ValueError(f'Checkpoint missing or checksum mismatch: {path}')
        print(f"Verified {len(manifest['files'])} checkpoint files.")
        return
    if args.action == 'plot':
        from plugfm.plots import plot
        plot(ROOT / 'results' / args.plot_group / 'summary.json', args.plot_group,
             args.output_dir or ROOT / 'outputs/figures', show_std=not args.no_std)
        return

    import torch
    from plugfm.data import DISTILLATION_SAMPLES, SCENE, STUDENT_METHODS, read_scene
    from plugfm.observation import obs
    from plugfm.runtime import Predictor, case_path, evaluate_case, load

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this Python environment. Use --device cpu or a CUDA-enabled PyTorch.')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    args.scene_file = args.scene_file or ROOT / 'data' / SCENE['path']

    if args.action == 'train':
        from plugfm.experiment import train_case

        args.count = 32 if args.smoke else DISTILLATION_SAMPLES
        args.csi_weight = 1.
        args.epochs, args.warm_epochs, args.head_epochs, args.class_epochs = (2, 1, 2, 2) if args.smoke else (60, 40, 40, 50)
        args.csi_epochs = args.head_epochs
        args.steps_per_epoch = args.head_steps = args.steps = 2 if args.smoke else 64
        args.batch_size, args.validate_every = (16, 1) if args.smoke else (128, 10)
        args.rank, args.alpha, args.lora_lr, args.full_lr = 4, 4., 1e-4, 1e-5
        args.warmup_epochs = 1 if args.smoke else 10
        run_folder = case_path(ROOT / 'outputs' / ('smoke' if args.smoke else 'runs'), args.wfm, args.seed, args.method, args.count)
        args.output_dir = args.output_dir or run_folder.with_suffix('')
        print(f'Saved {train_case(args, *checkpoint_inputs(args, args.wfm))}')
        return

    path = args.checkpoint or case_path(ROOT / 'checkpoints/cases', args.wfm, args.seed, args.method, DISTILLATION_SAMPLES)
    if not path.is_file():
        raise FileNotFoundError(f'No checkpoint at {path}. See checkpoints/README.md for available cases.')
    bundle = load(path)
    if args.action == 'demo' and bundle['method'] not in STUDENT_METHODS:
        raise ValueError('Demo expects a student checkpoint. Use evaluate for teacher baselines.')
    predictor = Predictor(bundle, *checkpoint_inputs(args, bundle['wfm']), args.device)
    if args.action == 'demo':
        g = torch.Generator(device=args.device).manual_seed(3)
        h = torch.randn(2, 32, 32, generator=g, device=args.device, dtype=torch.complex64) * predictor.config.channel_rms
        received = obs(h, predictor.context, 0, 11)
        out = predictor(received, 0)
        print('Synthetic Y:', tuple(received['y'].shape), 'on', args.device)
        print('Output shapes:', {key: tuple(value.shape) for key, value in out.items()})
        return

    smoke = bundle.get('smoke', False)
    parts, _ = read_scene(args.scene_file, smoke)
    rows = evaluate_case(predictor, parts['test'], args.device, batch_size=16 if smoke else 128, repeats=1 if smoke else 3)
    out = args.output_dir or ROOT / 'outputs/evaluation' / bundle['wfm'] / f"seed{bundle['seed']}" / path.stem
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'metrics.json', {'smoke': smoke, 'rows': rows})
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Evaluated {len(parts["test"]["h"])} users. Saved {out / "metrics.csv"}.')


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from None
