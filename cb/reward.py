"""B5 / R12: Wang et al.'s PRM-HE and ORM, rebuilt on our sampler.

Annotation (= LatentITS src/annotate.py): per question, K=5 trajectories at the default p, deduplicated
by answer; for each kept trajectory and each prefix h1..ht (t=1..5) run M=10 continuations; SE_t = share
of correct continuations, HE_t = SE_t > 0. As in LatentITS, the fixed prefix is re-encoded with dropout
OFF, so for t=6 the continuation is deterministic but can differ from the original answer (whose context
was encoded with dropout ON); we decode it once instead of 10 identical copies.

Reward model (= LatentITS src/models.py): GPT-2 initialised from the Coconut checkpoint reads
[question, <start>, h1..h6, <end>, answer]; PRM heads read the hidden state at each thought position,
the ORM head reads the last answer token. Heads: Linear-ReLU-Linear (logits; sigmoid at use).
Deviations (see DECISIONS.md): PRM loss uses all six steps of a path in one forward with class-balanced
weights instead of subsampling single (path, step) items; training is time-capped; fp16 autocast.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from . import data as D
from .sampling import coconut_rollout, decode

K_TRAJ, M_CONT = 5, 10


# ---------------------------------------------------------------- annotation
def annotate_group(bundle, rows, p, seed, rows_per_batch=1024, max_new=16):
    dev = bundle.model.wte.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    prompts = [D.prompt_ids(bundle.tok, r["question"]) for r in rows]
    out = coconut_rollout(bundle, prompts, K_TRAJ, p, gen, max_new)
    texts = decode(bundle.tok, out["ids"], bundle.eos_id)
    answers = [D.parse_answer(t) for t in texts]
    keep = []
    for qi, r in enumerate(rows):
        seen = set()
        for j in range(K_TRAJ):
            k = qi * K_TRAJ + j
            if answers[k] in seen:
                continue
            seen.add(answers[k])
            keep.append(k)
    keep_t = torch.tensor(keep, device=dev)
    th = out["thoughts"][keep_t]                                  # [U, 6, D]
    U = len(keep)
    owner = [k // K_TRAJ for k in keep]
    gold = [rows[o]["gold"] for o in owner]
    correct = np.array([answers[k] == g for k, g in zip(keep, gold)])
    se = np.zeros((U, 6), dtype=np.float32)
    for t in range(1, 7):                                         # prefix length t = 1..6
        m_cont = M_CONT if t < 6 else 1                           # t=6: no sampled step left -> deterministic
        chunk = max(1, rows_per_batch // m_cont)
        for s in range(0, U, chunk):
            idx = list(range(s, min(U, s + chunk)))
            cont = coconut_rollout(bundle, [prompts[owner[u]] for u in idx], m_cont, p, gen, max_new,
                                   prefix=th[idx, :t])
            ca = [D.parse_answer(x) for x in decode(bundle.tok, cont["ids"], bundle.eos_id)]
            for a, u in enumerate(idx):
                se[u, t - 1] = np.mean([ca[a * m_cont + m] == gold[u] for m in range(m_cont)])
    ids = out["ids"][keep_t].cpu().numpy()
    df = pd.DataFrame({"qid": [rows[o]["qid"] for o in owner], "traj": [k % K_TRAJ for k in keep],
                       "answer": [answers[k] for k in keep], "correct": correct,
                       "se": [list(map(float, x)) for x in se],
                       "answer_ids": [r[r >= 0].astype(np.int32).tolist() for r in ids]})
    return df, th.to(torch.float16).cpu().numpy()


def run_annotation(bundle, rows, split, p, out_root, roots, worker, nworkers, deadline,
                   questions_per_group=48, rows_per_batch=1024):
    name = f"{split}__p{p:.2f}"
    d = Path(out_root) / "annotations" / name
    d.mkdir(parents=True, exist_ok=True)
    groups = [rows[i:i + questions_per_group] for i in range(0, len(rows), questions_per_group)]
    exists = lambda g: any((Path(r) / "annotations" / name / f"group_{g:06d}.parquet").exists() for r in roots)
    todo = [g for g in range(len(groups)) if g % nworkers == worker and not exists(g)]
    t0, done_q = time.time(), 0
    for k, g in enumerate(todo):
        if deadline is not None and deadline.expired():
            print(f"[annotate] {name}: deadline, {len(todo) - k} groups left (resumable)")
            return False
        df, th = annotate_group(bundle, groups[g], p, seed=1_000_003 * (g + 1), rows_per_batch=rows_per_batch)
        np.save(d / f"group_{g:06d}.thoughts.tmp.npy", th)
        (d / f"group_{g:06d}.thoughts.tmp.npy").rename(d / f"group_{g:06d}.thoughts.npy")
        df.to_parquet(d / f"group_{g:06d}.tmp.parquet", index=False)
        (d / f"group_{g:06d}.tmp.parquet").rename(d / f"group_{g:06d}.parquet")
        done_q += len(groups[g])
        if (k + 1) % 20 == 0:
            el = time.time() - t0
            print(f"[annotate] {name} w{worker}: {k + 1}/{len(todo)} groups, {done_q / el:.2f} q/s, "
                  f"ETA {(len(todo) - k - 1) * el / (k + 1) / 3600:.2f} h", flush=True)
    return True


def load_annotations(roots, name):
    files = {}
    for r in roots:
        for f in sorted((Path(r) / "annotations" / name).glob("group_*.parquet")):
            if ".tmp" not in f.name:
                files.setdefault(f.name, f)
    metas, arrs = [], []
    for k in sorted(files):
        f = files[k]
        metas.append(pd.read_parquet(f))
        arrs.append(np.load(str(f).replace(".parquet", ".thoughts.npy")))
    if not metas:
        raise FileNotFoundError(f"no annotations {name} in {roots}")
    return pd.concat(metas, ignore_index=True), np.concatenate(arrs, 0)


# ---------------------------------------------------------------- model
class RewardModel(nn.Module):
    def __init__(self, hf_sd: dict, kind: str, start_id: int, end_id: int, pad_id: int, n_head: int = 12):
        super().__init__()
        from transformers import GPT2Config, GPT2Model
        vocab, dim = hf_sd["transformer.wte.weight"].shape
        n_layer = 1 + max(int(k.split(".")[2]) for k in hf_sd if k.startswith("transformer.h."))
        cfg = GPT2Config(vocab_size=vocab, n_embd=dim, n_layer=n_layer, n_head=n_head,
                         n_positions=hf_sd["transformer.wpe.weight"].shape[0])
        self.gpt = GPT2Model(cfg)
        body = {k[len("transformer."):]: v for k, v in hf_sd.items() if k.startswith("transformer.")}
        missing, unexpected = self.gpt.load_state_dict(body, strict=False)
        missing = [m for m in missing if not m.endswith("attn.bias") and not m.endswith("masked_bias")]
        if missing:
            raise ValueError(f"reward backbone missing keys: {missing[:5]}")
        self.kind, self.start_id, self.end_id, self.pad_id = kind, start_id, end_id, pad_id

        def head():
            return nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.he = head()
        self.se = head() if kind == "prm" else None

    def forward(self, prompt_ids, prompt_mask, thoughts, answer_ids, answer_mask):
        B, Lp = prompt_ids.shape
        emb = self.gpt.wte
        se_ = torch.full((B, 1), self.start_id, device=prompt_ids.device)
        ee_ = torch.full((B, 1), self.end_id, device=prompt_ids.device)
        x = torch.cat([emb(prompt_ids), emb(se_), thoughts.to(emb.weight.dtype), emb(ee_), emb(answer_ids)], 1)
        mask = torch.cat([prompt_mask, torch.ones(B, 8, device=x.device, dtype=prompt_mask.dtype), answer_mask], 1)
        pos = (mask.cumsum(-1) - 1).clamp_min(0)
        h = self.gpt(inputs_embeds=x, attention_mask=mask, position_ids=pos).last_hidden_state
        begin = Lp + 1
        if self.kind == "prm":
            steps = h[:, begin:begin + 6]
            return {"he": self.he(steps).squeeze(-1), "se": self.se(steps).squeeze(-1)}
        last = begin + 6 + answer_mask.sum(-1)
        return {"he": self.he(h[torch.arange(B, device=h.device), last]).squeeze(-1)}


class Batcher:
    """Builds padded tensors from annotation/pool rows without a DataLoader (all in RAM)."""

    def __init__(self, tok, questions: dict, pad_id: int):
        self.cache = {q: D.prompt_ids(tok, t) for q, t in questions.items()}
        self.pad = pad_id

    def __call__(self, qids, thoughts, answer_ids, device):
        P = [self.cache[int(q)] for q in qids]
        A = [list(a) for a in answer_ids]
        Lp, La = max(map(len, P)), max(1, max(map(len, A)))
        B = len(P)
        pi = np.full((B, Lp), self.pad, np.int64); pm = np.zeros((B, Lp), np.int64)
        ai = np.full((B, La), self.pad, np.int64); am = np.zeros((B, La), np.int64)
        for i, (p, a) in enumerate(zip(P, A)):
            pi[i, Lp - len(p):] = p; pm[i, Lp - len(p):] = 1
            ai[i, :len(a)] = a; am[i, :len(a)] = 1
        t = lambda x: torch.from_numpy(x).to(device, non_blocking=True)
        return t(pi), t(pm), torch.from_numpy(np.asarray(thoughts, dtype=np.float32)).to(device), t(ai), t(am)


def _losses(model, out, he, se, w_pos, w_neg):
    if model.kind == "prm":
        w = torch.where(he > 0.5, w_pos, w_neg)
        l_he = (F.binary_cross_entropy_with_logits(out["he"].float(), he, reduction="none") * w).mean()
        l_se = F.mse_loss(torch.sigmoid(out["se"].float()), se)
        return l_he + l_se
    return F.binary_cross_entropy_with_logits(out["he"].float(), he)


def train_reward(kind, hf_sd, bundle, train, train_th, valid, valid_th, questions, out_dir, deadline,
                 device="cuda", batch=128, micro=32, lr=1e-4, wd=0.01, warmup=500, epochs=10,
                 eval_every=1000, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = RewardModel(hf_sd, kind, bundle.start_id, bundle.end_id, bundle.eos_id, bundle.model.n_head).to(device)
    batcher = Batcher(bundle.tok, questions, bundle.eos_id)
    se_tr = np.stack(train["se"].to_numpy()).astype(np.float32)
    cor_tr = train["correct"].to_numpy().astype(np.float32)
    if kind == "orm":   # Wang: balance correct / incorrect 1:1 by subsampling
        pos, neg = np.where(cor_tr > 0)[0], np.where(cor_tr == 0)[0]
        m = min(len(pos), len(neg))
        items = np.concatenate([rng.permutation(pos)[:m], rng.permutation(neg)[:m]])
        w_pos = w_neg = 1.0
    else:               # all 6 steps per path; class-balanced weights on HE
        items = np.arange(len(train))
        frac = (se_tr > 0).mean()
        w_pos, w_neg = 0.5 / max(frac, 1e-6), 0.5 / max(1 - frac, 1e-6)
    steps_per_epoch = math.ceil(len(items) / batch)
    accum = batch // micro
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    state = {"total": epochs * steps_per_epoch}
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 1 - s / max(state["total"], 1)))
    cuda = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    def run(rows_df, th, idx, train_mode):
        q = rows_df["qid"].to_numpy()[idx]
        tensors = batcher(q, th[idx], rows_df["answer_ids"].to_numpy()[idx], device)
        he = torch.from_numpy((np.stack(rows_df["se"].to_numpy()[idx]) > 0).astype(np.float32)).to(device)
        se = torch.from_numpy(np.stack(rows_df["se"].to_numpy()[idx]).astype(np.float32)).to(device)
        if kind == "orm":
            he = torch.from_numpy(rows_df["correct"].to_numpy()[idx].astype(np.float32)).to(device)
        with torch.autocast("cuda" if cuda else "cpu", dtype=torch.float16, enabled=cuda):
            out = model(*tensors)
        return _losses(model, out, he, se, w_pos, w_neg)

    def validate():
        model.eval(); tot, cnt = 0.0, 0
        vidx = np.arange(len(valid))
        with torch.no_grad():
            for s in range(0, len(vidx), 128):
                i = vidx[s:s + 128]
                tot += run(valid, valid_th, i, False).item() * len(i); cnt += len(i)
        model.train()
        return tot / max(cnt, 1)

    model.train()
    step, best, t0, log = 0, float("inf"), time.time(), []
    timed = False
    for ep in range(epochs):
        order = rng.permutation(items)
        for b in range(steps_per_epoch):
            if deadline.expired(margin_min=10):
                break
            idx = order[b * batch:(b + 1) * batch]
            opt.zero_grad(set_to_none=True)
            for m in range(0, len(idx), micro):
                loss = run(train, train_th, idx[m:m + micro], True) * len(idx[m:m + micro]) / len(idx)
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if step == 30 and not timed:   # fit the LR schedule to the time budget
                per = (time.time() - t0) / 30
                fit = int((deadline.remaining_h() * 3600 - 15 * 60) / (per * 1.08))
                state["total"] = max(warmup + 1, min(state["total"], step + fit))
                timed = True
                print(f"[{kind}] {per:.2f}s/step, {steps_per_epoch} steps/epoch, planned {state['total']} steps "
                      f"(= {state['total'] / steps_per_epoch:.2f} epochs)", flush=True)
            if step % eval_every == 0 or step == state["total"]:
                v = validate()
                log.append({"step": step, "val_loss": v, "train_loss": float(loss.detach()) * accum, "min": (time.time() - t0) / 60})
                print(f"[{kind}] step {step} val {v:.4f}", flush=True)
                if v < best:
                    best = v
                    torch.save(model.state_dict(), out_dir / "best.pt")
            if step >= state["total"]:
                break
        if step >= state["total"] or deadline.expired(margin_min=10):
            break
    v = validate()
    log.append({"step": step, "val_loss": v, "min": (time.time() - t0) / 60})
    if v < best:
        torch.save(model.state_dict(), out_dir / "best.pt")
    (out_dir / "train_log.json").write_text(json.dumps({
        "kind": kind, "steps": step, "steps_per_epoch": steps_per_epoch, "epochs_done": step / steps_per_epoch,
        "n_train_items": int(len(items)), "n_valid": int(len(valid)), "w_pos": w_pos, "w_neg": w_neg, "log": log}))
    (out_dir / "DONE").write_text("ok")


@torch.no_grad()
def score_pool(kind, hf_sd, bundle, ckpt_path, meta, thoughts, questions, device="cuda", bs=256):
    model = RewardModel(hf_sd, kind, bundle.start_id, bundle.end_id, bundle.eos_id, bundle.model.n_head).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    if str(device).startswith("cuda"):
        model.half()
    batcher = Batcher(bundle.tok, questions, bundle.eos_id)
    he, se = [], []
    q = meta["qid"].to_numpy(); a = meta["answer_ids"].to_numpy()
    for s in range(0, len(meta), bs):
        t = batcher(q[s:s + bs], thoughts[s:s + bs], a[s:s + bs], device)
        out = model(*t)
        he.append(torch.sigmoid(out["he"].float()).cpu().numpy())
        if "se" in out:
            se.append(torch.sigmoid(out["se"].float()).cpu().numpy())
    res = pd.DataFrame({"qid": q, "sid": meta["sid"].to_numpy()})
    he = np.concatenate(he)
    if kind == "prm":
        se = np.concatenate(se)
        for t in range(6):
            res[f"prm_he_{t + 1}"] = he[:, t]
            res[f"prm_se_{t + 1}"] = se[:, t]
        res["prm_he_last"], res["prm_he_min"] = he[:, -1], he.min(1)
        res["prm_he_mean"], res["prm_he_max"] = he.mean(1), he.max(1)
    else:
        res["orm"] = he
    return res
