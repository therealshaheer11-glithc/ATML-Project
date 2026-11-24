"""Batched samplers.

coconut_rollout follows LatentITS (Wang et al.) src/sampling.py exactly:
  * prompt prefill with dropout OFF;
  * six latent forwards with dropout ON at every GPT-2 dropout site; their inputs are
    <|start-latent|>, h1, ..., h5 and their outputs are the thoughts h1..h6;
  * h6 is consumed with dropout OFF, then <|end-latent|> and greedy answer decoding with dropout OFF.
With a fixed prefix of f thoughts (MC annotation), h1..hf are inserted as-is and only steps f+1..6 are sampled.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .gpt2 import Drop, FastGPT2

N_THOUGHTS = 6


def _left_pad_embeds(model: FastGPT2, seqs: list[torch.Tensor], device):
    """seqs: list of [L_i, D] embeddings -> [B, L, D], mask [B, L] (left padded)."""
    L = max(s.size(0) for s in seqs)
    D = model.n_embd
    x = torch.zeros(len(seqs), L, D, device=device, dtype=model.wte.dtype)
    m = torch.zeros(len(seqs), L, device=device, dtype=torch.bool)
    for i, s in enumerate(seqs):
        x[i, L - s.size(0):] = s
        m[i, L - s.size(0):] = True
    return x, m


@torch.no_grad()
def greedy_answer(model: FastGPT2, cache, first_token: torch.Tensor, eos_id: int, max_new: int):
    """Greedy decode starting by feeding first_token [B]. Returns ids [B,T] (-1 after end),
    ntok [B] (incl. eos), conf [B] (mean max-prob over non-eos tokens, Wang et al.),
    logp [B] (sum log-prob of emitted tokens incl. eos)."""
    B = first_token.size(0)
    dev = first_token.device
    token = first_token
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    ids, probs, logps, active = [], [], [], []
    for _ in range(max_new):
        h = model(model.embed(token)[:, None], cache)
        lp = F.log_softmax(model.logits(h[:, -1]).float(), dim=-1)
        best_lp, nxt = lp.max(dim=-1)
        act = ~finished
        nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
        ids.append(torch.where(act, nxt, torch.full_like(nxt, -1)))
        probs.append(best_lp.exp())
        logps.append(best_lp)
        active.append(act)
        finished = finished | nxt.eq(eos_id)
        token = nxt
        if bool(finished.all()):
            break
    ids = torch.stack(ids, 1)
    act = torch.stack(active, 1)
    probs = torch.stack(probs, 1)
    logps = torch.stack(logps, 1)
    non_eos = act & ids.ne(eos_id)
    conf = (probs * non_eos).sum(1) / non_eos.sum(1).clamp_min(1)
    return {"ids": ids, "ntok": act.sum(1), "conf": conf, "logp": (logps * act).sum(1),
            "finished": finished}


@torch.no_grad()
def coconut_rollout(b, prompts: list[list[int]], n: int, p: float, gen: torch.Generator | None,
                    max_new: int = 16, prefix: torch.Tensor | None = None,
                    thought_hook=None):
    """prompts: Q token lists. Returns n rows per prompt (row q*n + j).
    prefix: optional [Q, f, D] fixed thoughts h1..hf (0 < f <= 6).
    thought_hook(step, h) -> h: optional intervention on each new thought (for later causal checks)."""
    m = b.model
    dev = m.wte.device
    f = 0 if prefix is None else prefix.size(1)
    start_emb = m.embed(torch.tensor([b.start_id], device=dev))  # [1, D]
    seqs = []
    for q, ids in enumerate(prompts):
        e = m.embed(torch.tensor(ids, device=dev))
        if f > 0:
            e = torch.cat([e, start_emb, prefix[q, :f - 1].to(e.dtype)], 0)
        seqs.append(e)
    x, mask = _left_pad_embeds(m, seqs, dev)
    max_len = x.size(1) + (N_THOUGHTS - f) + 1 + max_new + 1
    cache = m.new_cache(len(prompts), max_len)
    m(x, cache, mask)                                   # prefill, dropout off
    cache = cache.repeat_rows(n)
    B = len(prompts) * n
    if f == 0:
        current = start_emb[None].expand(B, 1, -1).contiguous()
        thoughts = []
    else:
        pre = prefix.to(m.wte.dtype).repeat_interleave(n, 0)
        current = pre[:, f - 1:f]
        thoughts = list(pre.unbind(1))
    drop = Drop(p, gen) if p > 0 else None
    for step in range(f, N_THOUGHTS):
        h = m(current, cache, drop=drop)               # dropout ON (latent steps only)
        if thought_hook is not None:
            h = thought_hook(step, h)
        thoughts.append(h[:, 0])
        current = h
    m(current, cache)                                   # consume h6, dropout off
    first = torch.full((B,), b.end_id, device=dev, dtype=torch.long)
    out = greedy_answer(m, cache, first, b.eos_id, max_new)
    out["thoughts"] = torch.stack(thoughts, 1)          # [B, 6, D]
    return out


@torch.no_grad()
def text_rollout(b, prompts: list[list[int]], n: int, temperature: float, gen: torch.Generator | None,
                 max_new: int = 64):
    """Text CoT / No-CoT sampling. temperature 0 = greedy. Returns ids, ntok, logp (sum of
    untempered log-probs of emitted tokens incl. eos), mean_logp."""
    m = b.model
    dev = m.wte.device
    seqs = [m.embed(torch.tensor(ids, device=dev)) for ids in prompts]
    x, mask = _left_pad_embeds(m, seqs, dev)
    cache = m.new_cache(len(prompts), x.size(1) + max_new + 1)
    h = m(x, cache, mask)
    last = h[:, -1]
    cache = cache.repeat_rows(n)
    last = last.repeat_interleave(n, 0)
    B = last.size(0)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    ids, logps, active = [], [], []
    for i in range(max_new):
        logits = m.logits(last).float()
        lp = F.log_softmax(logits, -1)
        if temperature > 0:
            probs = F.softmax(logits / temperature, -1)
            nxt = torch.multinomial(probs, 1, generator=gen)[:, 0]
        else:
            nxt = logits.argmax(-1)
        act = ~finished
        nxt = torch.where(finished, torch.full_like(nxt, b.eos_id), nxt)
        ids.append(torch.where(act, nxt, torch.full_like(nxt, -1)))
        logps.append(lp.gather(1, nxt[:, None])[:, 0])
        active.append(act)
        finished = finished | nxt.eq(b.eos_id)
        if bool(finished.all()) or i == max_new - 1:
            break
        last = m(m.embed(nxt)[:, None], cache)[:, -1]
    ids = torch.stack(ids, 1)
    act = torch.stack(active, 1)
    logp = (torch.stack(logps, 1) * act).sum(1)
    ntok = act.sum(1)
    return {"ids": ids, "ntok": ntok, "logp": logp, "mean_logp": logp / ntok.clamp_min(1),
            "finished": finished}


def decode(tok, ids: torch.Tensor, eos_id: int) -> list[str]:
    out = []
    for row in ids.tolist():
        row = [t for t in row if t >= 0 and t != eos_id]
        out.append(tok.decode(row, skip_special_tokens=True))
    return out
