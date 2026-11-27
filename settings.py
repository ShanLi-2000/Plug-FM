"""Defaults used when running main.py with the PyCharm Run button."""

# demo: built-in weights and a synthetic input. plot: paper results.
# train/evaluate: first follow README.md to download the scene and teacher code.
ACTION = 'demo'
WFM = 'contrawimae'  # 'contrawimae' or 'lwm'
METHOD = 'mse_csi'  # Plug-FM
DEVICE = 'auto'  # 'auto', 'cpu', 'cuda', or e.g. 'cuda:0'
SEED = 2026
PLOT_GROUP = 'joint'
SMOKE = False  # True runs two epochs on a small subset for setup checks.
CHECKPOINT = None  # Optional path to a previously trained case.pt.
