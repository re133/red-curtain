"""CPU FP8 expert storage with one bounded, shared GPU LRU cache."""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from red_curtain.fish_omni.moe import dequantize_fp8_blocks


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


class ExpertCache:
    def __init__(self, budget_bytes, device="cuda:0", check=lambda: None):
        if budget_bytes < 0:
            raise ValueError("negative cache budget")
        self.budget = budget_bytes
        self.device = torch.device(device)
        self.check = check
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.transfer_bytes = 0

    def get(self, layer, expert, storage, dtype):
        self.check()
        key = (layer, expert, dtype)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        required = sum(storage.get_buffer(n)[expert].numel() for n in
                       ("w13_weight", "w2_weight")) * torch.empty((), dtype=dtype).element_size()
        while self.entries and self.bytes + required > self.budget:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= sum(map(tensor_bytes, removed))
        result = []
        for prefix in ("w13", "w2"):
            raw = storage.get_buffer(prefix + "_weight")[expert]
            scales = storage.get_buffer(prefix + "_scale_inv")[expert]
            self.transfer_bytes += tensor_bytes(raw) + tensor_bytes(scales)
            result.append(dequantize_fp8_blocks(
                raw.to(self.device), scales.to(self.device), dtype))
        pair = tuple(result)
        if required <= self.budget:
            self.entries[key] = pair
            self.bytes += required
        return pair

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def resize(self,budget_bytes):
        if budget_bytes<0:raise ValueError('negative cache budget')
        self.budget=budget_bytes
        while self.entries and self.bytes>self.budget:
            _,removed=self.entries.popitem(last=False)
            self.bytes-=sum(map(tensor_bytes,removed))

    def stats(self):
        return dict(cache_bytes=self.bytes, cache_budget=self.budget, hits=self.hits,
                    misses=self.misses, host_to_device_bytes=self.transfer_bytes)


class OffloadedExperts(nn.Module):
    def __init__(self, storage, layer, cache):
        super().__init__()
        self.storage, self.layer, self.cache = storage, layer, cache

    def forward(self, x, topk_ids, topk_weights):
        # Copy only routing IDs to CPU. Weights remain FP8 in CPU RAM.
        ids = topk_ids.cpu()
        if x.shape[0] == 1:
            # Decode one token using two batched GEMMs instead of 16 separate GEMMs.
            weights=[self.cache.get(self.layer,int(expert),self.storage,x.dtype) for expert in ids[0]]
            w13=torch.stack([pair[0] for pair in weights])
            w2=torch.stack([pair[1] for pair in weights])
            rows=x.expand(len(weights),-1).unsqueeze(-1)
            gate,up=torch.bmm(w13,rows).squeeze(-1).chunk(2,dim=-1)
            result=torch.bmm(w2,(F.silu(gate)*up).unsqueeze(-1)).squeeze(-1)
            return (result.float()*topk_weights[0,:,None].float()).sum(0,keepdim=True).to(x.dtype)
        out = torch.zeros_like(x, dtype=torch.float32)
        for expert in ids.unique(sorted=True).tolist():
            locations = (ids == expert).nonzero()
            rows, slots = locations[:, 0].to(x.device), locations[:, 1].to(x.device)
            w13, w2 = self.cache.get(self.layer, expert, self.storage, x.dtype)
            gate, up = F.linear(x[rows], w13).chunk(2, dim=-1)
            contribution = F.linear(F.silu(gate) * up, w2).float()
            contribution *= topk_weights[rows, slots, None].float()
            out.index_add_(0, rows, contribution)
        return out.to(x.dtype)
