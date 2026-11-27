"""Load local training weights using separately obtained author source code."""
import copy
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .lwm_adapter import LWMTeacher
from .utils import parameter_count


class Teacher:
    def __init__(self, kind, checkpoint, source, device):
        self.kind = kind
        self.device = torch.device(device)
        source = Path(source).resolve()
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['kind'] != kind:
            raise ValueError(f'Expected {kind} weights in {checkpoint}.')
        config = dict(saved['config'])
        self.statistics = saved['normalization']
        self.checkpoint_sha256 = saved['original_sha256']
        if kind == 'contrawimae':
            entry = source / 'contrawimae/models/contrawimae.py'
            if not entry.is_file():
                raise FileNotFoundError(f'Place the official ContraWiMAE repository under {source}. See README.md.')
            sys.path.insert(0, str(source))
            from contrawimae.models.contrawimae import ContraWiMAE
            config['patch_size'] = tuple(config['patch_size'])
            self.feature_dim = config['encoder_dim']
            self.model = ContraWiMAE(device=self.device, **config)
            self.encoder_parameters = parameter_count(self.model.encoder)
        elif kind == 'lwm':
            self.model = LWMTeacher(config, source)
            self.feature_dim = self.model.feature_dim
            self.head_names = frozenset(self.model.head_parameter_names())
            self.encoder_parameters = sum(p.numel() for n, p in self.model.named_parameters() if n not in self.head_names)
        else:
            raise ValueError(f'Unknown teacher {kind}.')
        self.model.load_state_dict(saved['model'], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)

    def normalized(self, h):
        s = self.statistics
        h = h.to(self.device, dtype=torch.complex64)
        return torch.complex((h.real - s['real_mean']) / s['real_std'],
                             (h.imag - s['imag_mean']) / s['imag_std'])

    @torch.no_grad()
    def tokens(self, h):
        h = self.normalized(h)
        if self.kind == 'contrawimae':
            return self.model.encode(h, apply_mask=False)
        return self.model.encode_tokens(h).float()

    @torch.no_grad()
    def features(self, h, batch_size):
        return torch.cat([self.tokens(part).mean(1).cpu() for part in h.split(batch_size)])


class LWMReconstructionHead(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        b = teacher.model.backbone
        self.linear = copy.deepcopy(b.linear)
        self.norm = copy.deepcopy(b.norm)
        self.decoder = copy.deepcopy(b.decoder)
        self.decoder_bias = nn.Parameter(b.decoder_bias.detach().clone())

    def forward(self, tokens):
        values = (self.decoder(self.norm(F.relu(self.linear(tokens)))) + self.decoder_bias).float().flatten(1)
        return torch.complex(values[:, :1024].reshape(-1, 32, 32), values[:, 1024:].reshape(-1, 32, 32))


def head_for(teacher):
    if teacher.kind == 'contrawimae':
        head = copy.deepcopy(teacher.model.decoder)
    else:
        head = LWMReconstructionHead(teacher)
    return head.to(teacher.device).requires_grad_(True)


def decode(teacher, head, tokens):
    if teacher.kind == 'contrawimae':
        from contrawimae.models.modules.patching import InversePatcher
        ids = torch.arange(tokens.shape[1], device=tokens.device)[None].expand(len(tokens), -1)
        patches = head(tokens, ids, tokens.shape[1])
        h = InversePatcher((32, 32), teacher.model.patch_size)(patches)
    else:
        h = head(tokens)
    s = teacher.statistics
    return torch.complex(h.real * s['real_std'] + s['real_mean'], h.imag * s['imag_std'] + s['imag_mean'])
