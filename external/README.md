# External source locations

Run these commands from the Plug-FM release root. The paths are relative to that root.

| Source | Author repository | Expected path |
| --- | --- | --- |
| ContraWiMAE | [WirelessContrastiveMaskedLearning](https://github.com/BerkIGuler/WirelessContrastiveMaskedLearning) | `external/ContraWiMAE/contrawimae/models/contrawimae.py` |
| LWM | [wi-lab/lwm](https://huggingface.co/wi-lab/lwm) | `external/LWM/lwm_model.py` |
| Attention-Aided CE | [Attention_channel_estimation](https://github.com/EricGJB/Attention_channel_estimation) | `external/Attention_channel_estimation/CE_naive_attention.py` (optional reference) |

```sh
git clone https://github.com/BerkIGuler/WirelessContrastiveMaskedLearning.git external/ContraWiMAE
hf download wi-lab/lwm lwm_model.py --revision 3096624de46ebdd13c313559f787a84b888a905e --local-dir external/LWM
git clone https://github.com/EricGJB/Attention_channel_estimation.git external/Attention_channel_estimation
```

Use the first command for ContraWiMAE training or teacher evaluation and the second for LWM. The demo, result figures, and evaluation of the distilled student load the packaged student directly. Already downloaded checkouts can be selected with `--contra-source` and `--lwm-source`. Point `--lwm-source` at the directory containing `lwm_model.py`.

The LWM file used by the experiments has SHA-256 `b8cbfcade602e154f97898b5709e661c708f900f9168ca71fea4cbbaed218e36`. ContraWiMAE uses the public model package and strict checkpoint loading. Keep the original source version if an upstream API changes.

Gao et al. provide public author code for [An Attention-Aided Deep Learning Framework for Massive MIMO Channel Estimation](https://arxiv.org/abs/2108.09430). The public repository did not show an explicit license file at verification. The clone command is for optional reading. The Enhanced CSI Teacher entries in `results/joint/` record the project's 2D angle–frequency CNN/SE adaptation with LMMSE input. They are available for plotting.
