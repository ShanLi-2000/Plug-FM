# Released weights

All files are weights-only PyTorch dictionaries with the associated configuration and normalization. `MANIFEST.json` has file sizes, roles, and SHA-256 checksums. Run `python main.py check` to verify the bundle.

- `teachers/{contrawimae,lwm}.pt` are the project-trained checkpoints used as the source teachers, with architecture settings and pretraining normalization. Obtain the corresponding author source separately.
- `cases/<wfm>/seed<seed>/<method>_n2048.pt` contains the selected weights for the runnable methods listed in the release README, for seeds 2026–2028. Student bundles include the backbone, train-time projection, pilot/LMMSE calibration, and three downstream heads. The projection is bypassed at inference. Teacher cases include the task heads, normalization, and any task-specific encoder updates. Perfect Teacher contains only the two classification heads.
- Enhanced CSI Teacher is available as recorded comparison data in `results/joint/`.

Optimizer state, training feature caches, raw CSI, test labels and predictions are omitted. Fresh runs write resumable training state to `outputs/`, leaving the released weights available for evaluation.
