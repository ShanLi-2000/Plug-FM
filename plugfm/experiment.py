"""Train one matched case and export the same format as the bundled weights."""
from dataclasses import asdict
from pathlib import Path

from . import training
from .data import ADAPTED_METHODS, STUDENT_METHODS, prepare_training, read_scene
from .joint import train_task
from .runtime import load
from .teachers import Teacher
from .utils import acquire_run_lock, lock, save, sha, state_cpu


def train_case(options, teacher_path, source):
    parts, observation = read_scene(options.scene_file, options.smoke)
    teacher = Teacher(options.wfm, teacher_path, source, options.device)
    ctx, cal, config = prepare_training(teacher, parts, observation, options)
    folder = options.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    with acquire_run_lock(folder):
        lock(folder / 'protocol.json', {
            'wfm': options.wfm, 'method': options.method, 'seed': options.seed, 'samples': options.count,
            'smoke': options.smoke, 'device': options.device, 'teacher_sha256': sha(teacher_path),
            'layout': 'time8_freq4', 'config': asdict(config),
        })
        case = dict(wfm=options.wfm, method=options.method, seed=options.seed, samples=options.count,
                    layout='time8_freq4', config=asdict(config), calibration=cal.state,
                    protocol=ctx['protocol'], teacher_sha256=teacher.checkpoint_sha256, smoke=options.smoke)
        method = options.method
        if method in STUDENT_METHODS:
            student = training.train_representation(method, ctx, cal, config, options, options.seed, folder)
            csi = training.train_csi(method, student, None, cal, ctx, config, options, options.seed, folder)['student']
            heads = training.train_class_heads(method, student, None, cal, ctx, options, options.seed, folder)
            case.update(student=state_cpu(student), heads={key: state_cpu(head) for key, head in dict(heads, csi=csi).items()})
        elif method in ADAPTED_METHODS:
            case.update(zca=ctx['normalization'], adapted={})
            coarse_view = training.TeacherFeatureView(teacher, cal, ctx['normalization'])
            coarse_norm = None
            if method != 'perfect_teacher':
                mean, std = training.feature_normalization('classification_view', coarse_view, None, cal, ctx,
                                                           options, options.seed)
                coarse_norm = {'mean': mean, 'std': std}
            tasks = ('los', 'beam') if method == 'perfect_teacher' else ('csi', 'los', 'beam')
            for task in tasks:
                # Each task starts from the original teacher, with independent weights.
                task_teacher = Teacher(options.wfm, teacher_path, source, options.device)
                dest = folder / task
                train_task(options, task_teacher, ctx, cal, method, task, dest, coarse_norm)
                weights = load(dest / 'best.pt')
                case['adapted'][task] = dict(encoder=weights['encoder'], head=weights['head'])
        elif method == 'coarse_teacher':
            training.fit_teacher_head(options, 'time8_freq4', teacher, cal, ctx, options.seed, folder)
            view = training.TeacherFeatureView(teacher, cal, ctx['normalization'])
            heads = training.train_class_heads('classification_view', view, None, cal, ctx, options, options.seed, folder)
            case.update(zca=ctx['normalization'], heads={key: state_cpu(head) for key, head in heads.items()})
            case['heads']['csi'] = load(folder / 'csi_direct.pt')['head']
        else:
            raise ValueError(f'Unsupported training method: {method}')
        path = Path(folder) / 'case.pt'
        save(path, case)
    return path
