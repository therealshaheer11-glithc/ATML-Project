"""Minimal GPT-2 for fast batched inference with a preallocated KV cache.

Why not Hugging Face generate(): we need (1) dropout switched on only for the
six latent steps, at every site where HF GPT-2 applies it (embedding, attention
probabilities, attention output, MLP output), (2) dropout masks drawn from an
explicit torch.Generator so any shard can be replayed bit-for-bit, (3) hidden
states fed back as inputs, and (4) one prefill per question shared by all N
samples. Weights load from any HF GPT-2 state dict; numerics match HF in eval
mode (tests/test_gpt2_equivalence.py checks this).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Drop:
    """Dropout setting for one forward call. p=0 or None means off."""
    p: float
    gen: torch.Generator


def _dropout(x: torch.Tensor, drop: Drop | None) -> torch.Tensor:
    if drop is None or drop.p <= 0:
        return x
    keep = torch.rand(x.shape, generator=drop.gen, device=x.device, dtype=torch.float32) >= drop.p
    return x * keep.to(x.dtype) * (1.0 / (1.0 - drop.p))


class KVCache:
    """Left-padded cache. All rows share one write pointer; `valid` marks real keys."""

    def __init__(self, n_layer, batch, n_head, max_len, head_dim, device, dtype):
        shape = (n_layer, batch, n_head, max_len, head_dim)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.valid = torch.zeros(batch, max_len, device=device, dtype=torch.bool)
        self.pos = torch.zeros(batch, device=device, dtype=torch.long)  # next position id
        self.length = 0
        self.max_len = max_len

    @property
    def batch(self):
        return self.valid.size(0)

    def repeat_rows(self, n: int, max_len: int | None = None) -> "KVCache":
        """New cache with every row repeated n times (row i -> rows i*n..i*n+n-1)."""
        max_len = max_len or self.max_len
        L = self.length
        out = KVCache.__new__(KVCache)
        nl, b, h, _, d = self.k.shape
        out.k = torch.empty((nl, b * n, h, max_len, d), device=self.k.device, dtype=self.k.dtype)
        out.v = torch.empty_like(out.k)
        out.k[:, :, :, :L] = self.k[:, :, :, :L].repeat_interleave(n, dim=1)
        out.v[:, :, :, :L] = self.v[:, :, :, :L].repeat_interleave(n, dim=1)
        out.valid = torch.zeros(b * n, max_len, device=self.valid.device, dtype=torch.bool)
        out.valid[:, :L] = self.valid[:, :L].repeat_interleave(n, dim=0)
        out.pos = self.pos.repeat_interleave(n, dim=0)
        out.length = L
        out.max_len = max_len
        return out


class FastGPT2(nn.Module):
    def __init__(self, vocab_size, n_positions=1024, n_embd=768, n_layer=12, n_head=12, eps=1e-5):
        super().__init__()
        self.vocab_size, self.n_embd, self.n_layer, self.n_head = vocab_size, n_embd, n_layer, n_head
        self.head_dim = n_embd // n_head
        self.eps = eps
        self.wte = nn.Parameter(torch.empty(vocab_size, n_embd))
        self.wpe = nn.Parameter(torch.empty(n_positions, n_embd))
        D = n_embd
        self.ln1_w = nn.Parameter(torch.empty(n_layer, D)); self.ln1_b = nn.Parameter(torch.empty(n_layer, D))
        self.ln2_w = nn.Parameter(torch.empty(n_layer, D)); self.ln2_b = nn.Parameter(torch.empty(n_layer, D))
        # stored as [out, in] for F.linear
        self.attn_w = nn.Parameter(torch.empty(n_layer, 3 * D, D)); self.attn_b = nn.Parameter(torch.empty(n_layer, 3 * D))
        self.aproj_w = nn.Parameter(torch.empty(n_layer, D, D)); self.aproj_b = nn.Parameter(torch.empty(n_layer, D))
        self.fc_w = nn.Parameter(torch.empty(n_layer, 4 * D, D)); self.fc_b = nn.Parameter(torch.empty(n_layer, 4 * D))
        self.mproj_w = nn.Parameter(torch.empty(n_layer, D, 4 * D)); self.mproj_b = nn.Parameter(torch.empty(n_layer, D))
        self.lnf_w = nn.Parameter(torch.empty(D)); self.lnf_b = nn.Parameter(torch.empty(D))

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_hf_state_dict(cls, sd: dict, n_head: int = 12) -> "FastGPT2":
        sd = {k: v for k, v in sd.items()}
        for prefix in ("base_causallm.", "module.", "model."):
            if any(k.startswith(prefix) for k in sd):
                sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
        if not any(k.startswith("transformer.") for k in sd):
            sd = {"transformer." + k: v for k, v in sd.items()}
        t = "transformer."
        wte = sd[t + "wte.weight"]
        n_layer = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith(t + "h."))
        m = cls(wte.shape[0], sd[t + "wpe.weight"].shape[0], wte.shape[1], n_layer, n_head)

        def stack(name, transpose=False):
            ws = [sd[f"{t}h.{i}.{name}"] for i in range(n_layer)]
            if transpose:  # HF Conv1D stores [in, out]
                ws = [w.t() for w in ws]
            return torch.stack(ws).float()

        with torch.no_grad():
            m.wte.copy_(wte.float()); m.wpe.copy_(sd[t + "wpe.weight"].float())
            m.ln1_w.copy_(stack("ln_1.weight")); m.ln1_b.copy_(stack("ln_1.bias"))
            m.ln2_w.copy_(stack("ln_2.weight")); m.ln2_b.copy_(stack("ln_2.bias"))
            m.attn_w.copy_(stack("attn.c_attn.weight", True)); m.attn_b.copy_(stack("attn.c_attn.bias"))
            m.aproj_w.copy_(stack("attn.c_proj.weight", True)); m.aproj_b.copy_(stack("attn.c_proj.bias"))
            m.fc_w.copy_(stack("mlp.c_fc.weight", True)); m.fc_b.copy_(stack("mlp.c_fc.bias"))
            m.mproj_w.copy_(stack("mlp.c_proj.weight", True)); m.mproj_b.copy_(stack("mlp.c_proj.bias"))
            m.lnf_w.copy_(sd[t + "ln_f.weight"].float()); m.lnf_b.copy_(sd[t + "ln_f.bias"].float())
        return m

    # ------------------------------------------------------------------ api
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.wte)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.wte)

    def new_cache(self, batch, max_len, device=None, dtype=None) -> KVCache:
        device = device or self.wte.device
        dtype = dtype or self.wte.dtype
        return KVCache(self.n_layer, batch, self.n_head, max_len, self.head_dim, device, dtype)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, cache: KVCache, mask: torch.Tensor | None = None,
                drop: Drop | None = None) -> torch.Tensor:
        """x: [B, T, D] input embeddings (token embeddings or fed-back thoughts).
        mask: [B, T] (1 = real input, 0 = left padding). Returns final-layer
        hidden states after ln_f, [B, T, D] (what Coconut feeds back)."""
        B, T, D = x.shape
        H, hd = self.n_head, self.head_dim
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        mask = mask.bool()
        start = cache.length
        end = start + T
        if end > cache.max_len:
            raise RuntimeError(f"KV cache overflow: {end} > {cache.max_len}")
        pos = cache.pos[:, None] + mask.long().cumsum(1) - 1
        pos = pos.clamp_min(0)
        cache.pos += mask.long().sum(1)
        cache.valid[:, start:end] = mask
        cache.length = end

        keys_valid = cache.valid[:, :end]  # [B, L]
        qi = torch.arange(start, end, device=x.device)
        ki = torch.arange(end, device=x.device)
        attn_mask = keys_valid[:, None, :] & (ki[None, None, :] <= qi[None, :, None])  # [B,T,L]
        # padded query rows would attend to nothing -> NaN; let them see themselves
        attn_mask = attn_mask | (ki[None, None, :] == qi[None, :, None])
        attn_mask = attn_mask[:, None]  # [B,1,T,L]

        h = x + F.embedding(pos, self.wpe)
        h = _dropout(h, drop)
        manual = drop is not None and drop.p > 0
        scale = 1.0 / math.sqrt(hd)
        for i in range(self.n_layer):
            a = F.layer_norm(h, (D,), self.ln1_w[i], self.ln1_b[i], self.eps)
            qkv = F.linear(a, self.attn_w[i], self.attn_b[i])
            q, k, v = qkv.split(D, dim=-1)
            q = q.view(B, T, H, hd).transpose(1, 2)
            cache.k[i, :, :, start:end] = k.view(B, T, H, hd).transpose(1, 2)
            cache.v[i, :, :, start:end] = v.view(B, T, H, hd).transpose(1, 2)
            K = cache.k[i, :, :, :end]
            V = cache.v[i, :, :, :end]
            if manual:
                s = torch.matmul(q, K.transpose(-1, -2)).float() * scale
                s = s.masked_fill(~attn_mask, torch.finfo(torch.float32).min)
                p = torch.softmax(s, dim=-1)
                p = _dropout(p, drop).to(V.dtype)
                o = torch.matmul(p, V)
            else:
                o = F.scaled_dot_product_attention(q, K, V, attn_mask=attn_mask, scale=scale)
            o = o.transpose(1, 2).reshape(B, T, D)
            o = F.linear(o, self.aproj_w[i], self.aproj_b[i])
            h = h + _dropout(o, drop)
            m = F.layer_norm(h, (D,), self.ln2_w[i], self.ln2_b[i], self.eps)
            m = F.gelu(F.linear(m, self.fc_w[i], self.fc_b[i]), approximate="tanh")
            m = F.linear(m, self.mproj_w[i], self.mproj_b[i])
            h = h + _dropout(m, drop)
        return F.layer_norm(h, (D,), self.lnf_w, self.lnf_b, self.eps)
