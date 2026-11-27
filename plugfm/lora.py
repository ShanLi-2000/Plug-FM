"""Native encoder training and mergeable FFN LoRA for the two supported WFMs."""
import hashlib
import math

import torch
from torch import nn
from torch.nn.utils import parametrize


class LowRankWeight(nn.Module):
    def __init__(self, weight, rank=4, alpha=4.):
        super().__init__()
        self.a = nn.Parameter(weight.new_empty(rank, weight.shape[1]))
        self.b = nn.Parameter(weight.new_zeros(weight.shape[0], rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, weight):
        return weight + self.scale * (self.b @ self.a)


class NativeEncoder(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.kind = teacher.kind
        self.model = teacher.model
        self.statistics = dict(teacher.statistics)
        self.feature_dim = teacher.feature_dim
        self.base_count = teacher.encoder_parameters
        if self.kind == 'contrawimae':
            self.head_states = {n for n in self.model.state_dict() if not n.startswith('encoder.')}
            self.base_names = {n for n, _ in self.model.named_parameters() if n.startswith('encoder.')}
        else:
            self.head_states = set(self.model.head_state_names())
            self.base_names = {n for n, _ in self.model.named_parameters() if n not in teacher.head_names}
        assert sum(p.numel() for n, p in self.model.named_parameters() if n in self.base_names) == self.base_count
        self.model.eval().requires_grad_(False)
        self.mode = 'perfect_teacher'
        self.lora_targets = []

    def encoder_count(self):
        if self.kind == 'contrawimae':
            return sum(p.numel() for n, p in self.model.named_parameters() if n.startswith('encoder.'))
        return sum(p.numel() for n, p in self.model.named_parameters() if n not in self.head_states)

    def normalized(self, h):
        s = self.statistics
        return torch.complex((h.real - s['real_mean']) / s['real_std'],
                             (h.imag - s['imag_mean']) / s['imag_std'])

    def forward(self, h):
        x = self.normalized(h)
        if self.kind == 'contrawimae':
            # Public ContraWiMAE.encode is no-grad: call its native modules.
            return self.model.encoder(self.model.patcher(x), apply_mask=False).float()
        return self.model.encode_tokens(x).float()

    def configure(self, mode, rank=4, alpha=4.):
        self.mode = mode
        self.model.requires_grad_(False)
        if mode == 'full_transfer':
            for name, p in self.model.named_parameters():
                p.requires_grad_(name in self.base_names)
        elif mode == 'lora_teacher':
            targets = []
            for name, layer in self.model.named_modules():
                if not isinstance(layer, nn.Linear) or name + '.weight' not in self.base_names:
                    continue
                if (self.kind == 'contrawimae' and name.startswith('encoder.') and name.endswith(('linear1', 'linear2'))
                    or self.kind == 'lwm' and '.pos_ffn.' in name and name.endswith(('fc1', 'fc2'))
                    ):
                    targets.append((name, layer))
            assert len(targets) == 24
            for name, layer in targets:
                parametrize.register_parametrization(layer, 'weight', LowRankWeight(layer.weight, rank, alpha))
                self.lora_targets.append(name)
        elif mode != 'perfect_teacher':
            raise ValueError(mode)
        self.model.eval()

    def training_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def encoder_state(self):
        return {n: v.detach().cpu().clone() for n, v in self.model.state_dict().items() if n not in self.head_states}

    def learned_state(self):
        keys = self.learned_keys()
        return {n: v.detach().cpu().clone() for n, v in self.model.state_dict().items() if n in keys}

    def learned_keys(self):
        if self.mode == 'lora_teacher':
            return {n for n in self.model.state_dict() if '.parametrizations.' in n and n.endswith(('.a', '.b'))}
        return {n for n in self.model.state_dict() if n not in self.head_states} if self.mode == 'full_transfer' else set()

    def load_learned(self, state):
        expected = self.learned_keys()
        assert set(state) == expected
        result = self.model.load_state_dict(state, strict=False)
        assert not result.unexpected_keys

    def frozen_digest(self):
        digest = hashlib.sha256()
        for name, p in self.model.named_parameters():
            if '.parametrizations.' in name and name.endswith(('.a', '.b')):
                continue
            digest.update(name.encode())
            digest.update(p.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def merge_lora(self):
        assert self.mode == 'lora_teacher'
        for name in self.lora_targets:
            layer = self.model.get_submodule(name)
            parametrize.remove_parametrizations(layer, 'weight', leave_parametrized=True)
        self.mode = 'merged_lora'
        self.model.eval().requires_grad_(False)
        assert self.encoder_count() == self.base_count
