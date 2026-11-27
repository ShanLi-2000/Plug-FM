"""Make the released SNR figures from the saved metric tables."""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .utils import read


TASKS = (('csi_nmse_db', 'CSI NMSE (dB)', 1), ('los_auc', 'LoS AUROC', 1), ('beam_top1', '16-beam top-1 (%)', 100))
NAMES = {'contrawimae': 'ContraWiMAE', 'lwm': 'LWM'}
CURVES = {
    'coarse_teacher': ('Coarse Teacher', '#0072B2', '--', 's'),
    'enhanced_teacher': ('Enhanced CSI Teacher', '#009E73', '-.', 'D'),
    'perfect_teacher': ('Perfect Teacher', '#333333', '--', None),
    'lora_teacher': ('Task LoRA Teacher', '#8554A0', ':', 'v'),
    'full_transfer': ('Task Full FT Teacher', '#A66C00', '-.', '^'),
    'mse_csi': ('Plug-FM', '#D55E00', '-', 'o'),
}
GROUPS = {
    'joint': tuple(CURVES)[:6],
}


def plot(summary, group, destination, show_std=True):
    report = read(summary)
    points = {(r['wfm'], r['method'], r['snr_db']): r for r in report['rows']}
    wfms = [name for name in NAMES if any(key[0] == name for key in points)]
    snrs = sorted({key[2] for key in points})
    rows = len(wfms)
    height = 2.38 + (rows - 1) * 1.76
    methods = GROUPS[group]
    font = ['Times New Roman', 'DejaVu Serif']
    with plt.rc_context({'font.family': font, 'font.size': 8, 'axes.titlesize': 9,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(rows, 3, figsize=(7.16, height), squeeze=False, sharex=True, sharey='col')
        handles = {}
        for idx, wfm in enumerate(wfms):
            for ax, (metric, ylabel, scale) in zip(axes[idx], TASKS):
                for method in methods:
                    values = [points[wfm, method, s] for s in snrs]
                    if values[0][metric] is None:
                        continue
                    label, color, linestyle, marker = CURVES[method]
                    y = scale * np.array([r[metric] for r in values])
                    std = [r.get(metric + '_std') for r in values]
                    if show_std and all(s is not None for s in std):
                        spread = scale * np.array(std)
                        low, high = y - spread, y + spread
                        if metric != 'csi_nmse_db':
                            low, high = np.clip(low, 0, scale), np.clip(high, 0, scale)
                        ax.fill_between(snrs, low, high, color=color, alpha=.12, linewidth=0)
                    (handles[method],) = ax.plot(snrs, y, label=label, color=color, linestyle=linestyle,
                                                marker=marker, linewidth=1.1, markersize=3, markerfacecolor='white')
                ax.set(ylabel=ylabel, xticks=snrs, xlim=(snrs[0] - .8, snrs[-1] + .8))
                ax.spines[['top', 'right']].set_visible(False)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.grid(axis='y', color='#D9D9D9', linewidth=.45)
            if rows > 1:
                axes[idx, 0].set_title(NAMES[wfm], loc='left')
        columns = 3
        legend = [handles[name] for col in range(columns) for name in methods[col::columns]]
        two_rows = len(legend) > columns
        fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .04 / height),
                   ncols=columns, frameon=False, fontsize=8.2, handlelength=2.6)
        fig.supxlabel('Test SNR (dB)', y=(.55 if two_rows else .36) / height, fontsize=8)
        fig.subplots_adjust(left=.073, right=.986, bottom=(.91 if two_rows else .72) / height,
                            top=1 - (.24 if rows > 1 else .10) / height, wspace=.37, hspace=.29)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for ext in ('pdf', 'png'):
            file = destination / f'plugfm_{group}.{ext}'
            fig.savefig(file, dpi=300)
            print(f'Saved {file}')
        plt.close(fig)
