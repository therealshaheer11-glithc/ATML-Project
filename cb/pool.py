"""Sample once, analyze many times: sharded, resumable, replayable pools.

A pool = one (checkpoint, split, sampler setting, seed) with n attempts per question.
Layout:  <root>/pools/<pool_name>/
            spec.json                      settings + batching plan (needed for replay)
            shard_00000.parquet            one row per attempt (metadata)
            shard_00000.thoughts.npy       [rows, 6, 768] float16 (if save_thoughts)
            DONE                           written when every shard exists
Replay: every shard draws its dropout masks from torch.Generator(seed=shard_seed), with a fixed
batch composition recorded in spec.json, so `replay_shard` regenerates it bit-for-bit (up to GPU
nondeterminism in matmuls) and can apply an intervention via thought_hook.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data as D
from .sampling import coconut_rollout, decode, text_rollout


@dataclass
class PoolSpec:
    ckpt: str                 # key in models.CHECKPOINTS
    split: str                # test | train | valid
    kind: str                 # latent | text
    setting: float            # dropout p (latent) or temperature (text); 0 = deterministic
    seed: int
    n: int                    # attempts per question
    max_new: int = 16
    rows_per_batch: int = 1024
    save_thoughts: bool = True
    subset: int | None = None  # train only: first `subset` questions of a fixed shuffle (seed 1234)
    tag: str = ""

    @property
    def name(self) -> str:
        s = f"{self.ckpt}__{self.split}"
        if self.subset:
            s += f"{self.subset}"
        s += f"__{'p' if self.kind == 'latent' else 'T'}{self.setting:.2f}__s{self.seed}__n{self.n}"
        return s + (f"__{self.tag}" if self.tag else "")


def shard_seed(spec: PoolSpec, shard: int) -> int:
    h = hashlib.sha256(f"{spec.name}|{shard}".encode()).hexdigest()
    return int(h[:15], 16)


def select_rows(spec: PoolSpec, data_dir) -> list[dict]:
    rows = D.load(data_dir, spec.split)
    if spec.subset:
        rows = D.train_subset(rows, spec.subset, seed=1234)
    return rows


def plan(spec: PoolSpec, rows: list[dict], tok) -> list[list[int]]:
    """Questions sorted by prompt length (less padding), chunked so each shard has ~rows_per_batch attempts."""
    lens = [(len(D.prompt_ids(tok, r["question"])), r["qid"]) for r in rows]
    order = [q for _, q in sorted(lens)]
    per = max(1, spec.rows_per_batch // spec.n)
    return [order[i:i + per] for i in range(0, len(order), per)]


def pool_dir(root, spec_or_name) -> Path:
    name = spec_or_name if isinstance(spec_or_name, str) else spec_or_name.name
    return Path(root) / "pools" / name


def find_pool(roots, name) -> Path | None:
    for r in roots:
        d = pool_dir(r, name)
        if (d / "spec.json").exists():
            return d
    return None


def is_done(roots, name) -> bool:
    d = find_pool(roots, name)
    return d is not None and (d / "DONE").exists()


def _shard_exists(roots, name, j) -> bool:
    return any((pool_dir(r, name) / f"shard_{j:05d}.parquet").exists() for r in roots)


def run_one_shard(spec: PoolSpec, bundle, by_qid: dict, qids: list[int], j: int, thought_hook=None):
    dev = bundle.model.wte.device
    prompts = [D.prompt_ids(bundle.tok, by_qid[q]["question"]) for q in qids]
    gen = None
    if spec.setting > 0:
        gen = torch.Generator(device=dev)
        gen.manual_seed(shard_seed(spec, j))
    if spec.kind == "latent":
        out = coconut_rollout(bundle, prompts, spec.n, spec.setting, gen, spec.max_new,
                              thought_hook=thought_hook)
    else:
        out = text_rollout(bundle, prompts, spec.n, spec.setting, gen, spec.max_new)
    return out


def _to_frame(spec, bundle, by_qid, qids, j, out) -> pd.DataFrame:
    texts = decode(bundle.tok, out["ids"], bundle.eos_id)
    answers = [D.parse_answer(t) for t in texts]
    B = len(texts)
    qcol = np.repeat(np.array(qids, dtype=np.int32), spec.n)
    gold = [by_qid[int(q)]["gold"] for q in qcol]
    ids = out["ids"].cpu().numpy()
    df = pd.DataFrame({
        "qid": qcol,
        "sid": np.tile(np.arange(spec.n, dtype=np.int16), len(qids)),
        "shard": np.full(B, j, dtype=np.int32),
        "row": np.arange(B, dtype=np.int32),
        "text": texts,
        "answer": answers,
        "gold": gold,
        "correct": [a == g for a, g in zip(answers, gold)],
        "correct_num": [D.numeric_equal(a, g) for a, g in zip(answers, gold)],
        "finished": out["finished"].cpu().numpy(),
        "ntok": out["ntok"].cpu().numpy().astype(np.int16),
        "logp": out["logp"].float().cpu().numpy(),
        "answer_ids": [r[r >= 0].astype(np.int32).tolist() for r in ids],
    })
    if "conf" in out:
        df["conf"] = out["conf"].float().cpu().numpy()
    if "mean_logp" in out:
        df["mean_logp"] = out["mean_logp"].float().cpu().numpy()
    return df


def run_pool(spec: PoolSpec, bundle, data_dir, out_root, roots, worker=0, nworkers=1, deadline=None,
             log_every=10) -> bool:
    """Generate every missing shard assigned to this worker. Returns True if the pool is complete."""
    rows = select_rows(spec, data_dir)
    by_qid = {r["qid"]: r for r in rows}
    shards = plan(spec, rows, bundle.tok)
    d = pool_dir(out_root, spec)
    d.mkdir(parents=True, exist_ok=True)
    meta = {"spec": asdict(spec), "name": spec.name, "n_shards": len(shards), "shards": shards}
    (d / "spec.json").write_text(json.dumps(meta))
    todo = [j for j in range(len(shards)) if j % nworkers == worker and not _shard_exists(roots, spec.name, j)]
    t0 = time.time()
    for k, j in enumerate(todo):
        if deadline is not None and deadline.expired():
            print(f"[pool] {spec.name}: deadline reached, {len(todo) - k} shards left (resumable)")
            return False
        out = run_one_shard(spec, bundle, by_qid, shards[j], j)
        df = _to_frame(spec, bundle, by_qid, shards[j], j, out)
        if spec.save_thoughts and spec.kind == "latent":
            arr = out["thoughts"].to(torch.float16).cpu().numpy()
            tmp = d / f"shard_{j:05d}.thoughts.tmp.npy"
            np.save(tmp, arr)
            tmp.rename(d / f"shard_{j:05d}.thoughts.npy")
        tmp = d / f"shard_{j:05d}.tmp.parquet"
        df.to_parquet(tmp, index=False)
        tmp.rename(d / f"shard_{j:05d}.parquet")   # parquet last = shard complete
        if (k + 1) % log_every == 0 or k + 1 == len(todo):
            el = time.time() - t0
            print(f"[pool] {spec.name} w{worker}: {k + 1}/{len(todo)} shards, "
                  f"{el / 60:.1f} min, acc so far {df['correct'].mean():.3f}", flush=True)
    complete = all(_shard_exists(roots, spec.name, j) for j in range(len(shards)))
    if complete:
        (d / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    return complete


# ---------------------------------------------------------------- readers
def shard_files(roots, name, suffix=".parquet"):
    files = {}
    for r in roots:
        d = pool_dir(r, name)
        for f in sorted(d.glob(f"shard_*{suffix}")):
            if ".tmp" in f.name or (suffix == ".parquet" and "thoughts" in f.name):
                continue
            files.setdefault(f.name, f)
    return [files[k] for k in sorted(files)]


def load_meta(roots, name, columns=None) -> pd.DataFrame:
    files = shard_files(roots, name)
    if not files:
        raise FileNotFoundError(f"pool {name} not found in {roots}")
    if columns is not None:
        columns = list(dict.fromkeys(["qid", "sid"] + list(columns)))
    df = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    return df.sort_values(["qid", "sid"]).reset_index(drop=True)


def load_thoughts(roots, name) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (meta sorted by qid,sid) and thoughts [rows, 6, D] float16 aligned with it."""
    files = shard_files(roots, name)
    metas, arrs = [], []
    for f in files:
        metas.append(pd.read_parquet(f, columns=["qid", "sid", "correct", "answer"]))
        arrs.append(np.load(str(f).replace(".parquet", ".thoughts.npy"), mmap_mode="r"))
    meta = pd.concat(metas, ignore_index=True)
    arr = np.concatenate(arrs, 0)
    order = np.lexsort((meta["sid"].values, meta["qid"].values))
    return meta.iloc[order].reset_index(drop=True), arr[order]


def replay_shard(roots, name, j, bundle, data_dir, thought_hook=None):
    """Regenerate shard j of a saved pool with the same RNG and batch composition."""
    d = find_pool(roots, name)
    meta = json.loads((d / "spec.json").read_text())
    spec = PoolSpec(**meta["spec"])
    rows = select_rows(spec, data_dir)
    by_qid = {r["qid"]: r for r in rows}
    return run_one_shard(spec, bundle, by_qid, meta["shards"][j], j, thought_hook=thought_hook)
