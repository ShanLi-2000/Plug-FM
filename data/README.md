# DeepMIMO channel data

The dataset is hosted in the [ContraWiMAE DeepMIMO dataset repository](https://huggingface.co/datasets/BerkIGuler/ContraWiMAEDataset). `scene.json` identifies the exact file and checksum used for site distillation, task adaptation, and testing. From the root of this release, download the Centro file (about 161 MiB).

```sh
hf download BerkIGuler/ContraWiMAEDataset sub6/task/train/city_29_centro_3p5_bs000.npz --repo-type dataset --revision 0373102eafb5f908cab031b13faac93b8c9ca740 --local-dir data
```

Or run `python data/download_data.py`. Clicking Run on `data/download_data.py` in PyCharm has the same effect. The downloader checks SHA-256. A previously downloaded copy can be referenced with `--scene-file`.

The experiment uses 8,192 train users, 2,048 validation users, and 2,048 test users drawn from disjoint subsets of the same site. The split is recreated from LoS, 16-beam index, and user position with split seed `20261028`. Train-only LMMSE and target feature statistics are fitted on the selected distillation subset. No downloaded NPZ or user-level prediction files ship in the release.

`pretraining_subset.json` lists the 17 upstream files used to train the bundled source teachers. The teacher weights are already provided. To obtain that separate pretraining subset (about 3.6 GiB), run `python data/download_data.py --pretraining`. Cite the dataset authors as described on the dataset card.
