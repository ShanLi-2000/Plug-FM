# Plug-FM

Plug-FM distills a CSI-based wireless foundation model into a shared student that takes complex uplink observations. This release covers the **ContraWiMAE and LWM** experiments on Centro, including CSI estimation, LoS classification and 16-beam selection.

## Quick start

In the release folder, select a Python 3.10+ environment and run `python -m pip install -r requirements.txt`. For GPU training, use a CUDA-enabled build of [PyTorch](https://pytorch.org/get-started/locally/) for your machine.

```sh
python main.py
python main.py check
python main.py plot --plot-group joint
```

The first command loads a trained ContraWiMAE student and its three heads, then runs a synthetic complex `[2, 32, 16, 32]` observation through them. The demo and plotting commands can run immediately after installing dependencies. Figures are saved in `outputs/figures/`.

**PyCharm 点击运行**。选择 **Open** 打开整个文件夹，选择已安装依赖的 Python interpreter，然后右键 `main.py` → **Run 'main'**。默认运行 demo。要绘制主实验图，在 `settings.py` 中把 `ACTION` 改为 `'plot'`。下载数据和对应外部模型后，可以将 `ACTION` 改为 `'train'` 或 `'evaluate'`，设置 `WFM`、`METHOD`、`SEED` 和 `DEVICE`，再点击运行。命令行参数可覆盖这些默认值。

## Data and upstream code

The training and evaluation file is a [ContraWiMAE DeepMIMO](https://huggingface.co/datasets/BerkIGuler/ContraWiMAEDataset) scene. Run `python data/download_data.py`, or use the Hugging Face CLI command in [data/README.md](data/README.md). It lands at `data/sub6/task/train/city_29_centro_3p5_bs000.npz`.

| Source | Repository | Local placement |
| --- | --- | --- |
| ContraWiMAE | [BerkIGuler/WirelessContrastiveMaskedLearning](https://github.com/BerkIGuler/WirelessContrastiveMaskedLearning) | `external/ContraWiMAE/` |
| LWM | [wi-lab/lwm](https://huggingface.co/wi-lab/lwm) | `external/LWM/lwm_model.py` |
| Attention-Aided CE | [EricGJB/Attention_channel_estimation](https://github.com/EricGJB/Attention_channel_estimation) | `external/Attention_channel_estimation/` |

See the clone and download commands in [external/README.md](external/README.md). **Enhanced CSI Teacher** is included as recorded main-experiment metrics and a plot curve. Those results were generated with our angle–frequency adaptation of the CNN/SE design using an LMMSE CSI input. For source reference, see the authors' optional Attention-Aided CE repository. The paper's WirelessGPT example was a local reproduction and is outside this release's code, weights, and result tables.

## Re-evaluate and train

```sh
# Re-evaluate a released student on the downloaded test users.
python main.py evaluate --wfm contrawimae --method mse_csi --seed 2026 --device cuda

# Verify the training path quickly after obtaining the LWM source.
python main.py train --wfm lwm --method mse_csi --smoke --device cuda

# Train Plug-FM with the paper's 2,048-channel distillation setting.
python main.py train --wfm contrawimae --method mse_csi --seed 2026 --device cuda
```

Training prints the path to `case.pt`. Pass that path to `python main.py evaluate --checkpoint <path-to-case.pt> --device cuda`, or set `CHECKPOINT` in `settings.py`. Re-run the same training command to resume a run. For shared folders or repeated configurations, use `--output-dir` to select a distinct output location. Benchmark results use GPU-generated observations. CPU mode is available for checking the workflow and may produce different noisy observations.

| `--method` | Experiment |
| --- | --- |
| `mse_csi` | Plug-FM, global feature MSE plus internal CSI NMSE with weight 1.0 |
| `pilot_only_csi` | Same student architecture with the data-stem input held at zero |
| `feature_mse_only` | Full observation with feature MSE alone |
| `coarse_teacher` | Frozen teacher with pilot-derived LMMSE input |
| `perfect_teacher` | Frozen teacher with true CSI for LoS and beam selection |
| `lora_teacher` / `full_transfer` | Per-task head warm-up for 10 epochs followed by joint encoder/head fine-tuning |

Main protocol uses a 6.25% comb pilot grid (`time8_freq4`), continuous training SNR in −20 to 20 dB, and nine test SNRs at 5 dB spacing. Student distillation uses 2,048 training channels and 60 epochs with 64 updates per epoch. The CSI head uses 40 epochs and classification heads use 50 epochs, each with 8,192 train users.

## Included experiments

- [Checkpoint manifest](checkpoints/MANIFEST.json) and [weight descriptions](checkpoints/README.md) cover the trained teacher variants, Plug-FM, and its ablations.
- `results/joint/` holds the main experiment, including the recorded **Enhanced CSI Teacher** comparison. It contains `by_seed.csv`, `by_snr.csv`, `macro.csv`, `paired_differences.csv`, and `summary.json`.
- `results/figures/` contains PDF and PNG exports of the main experiment figure.
- SNR figures show the three training seeds' mean and sample standard deviation. Per-seed SNR scores average three matched data/noise observations. The `macro.csv` tables equally weight the nine SNRs.

Code and results in this folder are the Plug-FM release candidate. Upstream source and data use their respective repository terms.
