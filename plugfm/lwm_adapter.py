"""NPZ input adapter. The LWM backbone is loaded from the separate author repository."""
import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def official_module(source):
    path = Path(source) / 'lwm_model.py'
    if not path.is_file():
        raise FileNotFoundError(f'Place the upstream lwm_model.py at {path}. See README.md.')
    spec = importlib.util.spec_from_file_location('plugfm_external_lwm', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LWMTeacher(nn.Module):
    def __init__(self, config, source):
        super().__init__()
        self.config = dict(config)
        self.length = int(config['patch_length'])
        if self.length < 1 or 1024 % self.length:
            raise ValueError('patch-length must divide 32*32 without crossing real/imag boundaries')
        if not 0 < config['mask_ratio'] < 1 or config['encoder_layers'] < 1:
            raise ValueError('Invalid mask ratio/layer count')
        if config['pooling'] not in ('mean', 'cls'):
            raise ValueError('Expected mean/cls pooling')
        self.real_tokens = 1024 // self.length
        if self.real_tokens < 2:
            raise ValueError('Need at least two patches per real/imag component')
        self.feature_dim = 64
        # The official implementation deliberately uses 12 heads of width 5,
        # not a torch MultiheadAttention requiring 64 to be divisible by 12.
        self.backbone = official_module(source).lwm(element_length=self.length, d_model=64,
                                             max_len=1 + 2 * self.real_tokens,
                                             n_layers=config['encoder_layers'])

    def patchify(self, h):
        if h.ndim != 3 or tuple(h.shape[1:]) != (32, 32) or not torch.is_complex(h):
            raise ValueError('LWM expects complex [B,32,32] channels')
        values = torch.cat((h.real.flatten(1), h.imag.flatten(1)), dim=1)
        return values.reshape(len(h), 2 * self.real_tokens, self.length)

    def make_tokens(self, h, *, mask, generator=None):
        patches = self.patchify(h)
        cls = patches.new_full((len(h), 1, self.length), .2)
        tokens = torch.cat((cls, patches), dim=1)
        if not mask:
            return tokens, None, None
        number = min(self.real_tokens - 1, max(1, int(self.real_tokens * self.config['mask_ratio'])))
        positions = torch.rand(len(h), self.real_tokens, device=h.device, generator=generator).argsort(1)[:, :number]
        positions = torch.cat((positions, positions + self.real_tokens), dim=1) + 1
        rows = torch.arange(len(h), device=h.device)[:, None]
        target = tokens[rows, positions].clone()
        decision = torch.rand(positions.shape, device=h.device, generator=generator)
        random_values = torch.rand(target.shape, device=h.device, generator=generator)
        # Official make_sample: 10% U(0,1), 80% constant 0.1, 10% unchanged.
        replacement = torch.where((decision < .1)[..., None], random_values,
                                  torch.where((decision < .9)[..., None], torch.full_like(target, .1), target))
        tokens[rows, positions] = replacement
        return tokens, positions, target

    def pretraining_loss(self, h, generator=None):
        tokens, positions, target = self.make_tokens(h, mask=True, generator=generator)
        prediction, _ = self.backbone(tokens, positions)
        return F.mse_loss(prediction.float(), target.float()), {'masked_tokens_per_user': positions.shape[1]}

    def _encode_all(self, h):
        tokens, _, _ = self.make_tokens(h, mask=False)
        output = self.backbone.embedding(tokens)
        for layer in self.backbone.layers:
            output, _ = layer(output)
        return output

    def encode_tokens(self, h):
        return self._encode_all(h)[:, 1:]

    def encode(self, h):
        output = self._encode_all(h)
        return output[:, 0] if self.config['pooling'] == 'cls' else output[:, 1:].mean(1)

    def reconstruct(self, h):
        output = self._encode_all(h)[:, 1:]
        value = self.backbone.decoder(self.backbone.norm(F.relu(self.backbone.linear(output)))) + self.backbone.decoder_bias
        value = value.float().flatten(1)
        return torch.complex(value[:, :1024].reshape(-1, 32, 32), value[:, 1024:].reshape(-1, 32, 32))

    def head_parameter_names(self):
        return [n for n, _ in self.named_parameters() if n.startswith(('backbone.linear.', 'backbone.norm.', 'backbone.decoder.', 'backbone.decoder_bias'))]

    def head_state_names(self):
        return [n for n in self.state_dict() if n.startswith(('backbone.linear.', 'backbone.norm.', 'backbone.decoder.', 'backbone.decoder_bias'))]
