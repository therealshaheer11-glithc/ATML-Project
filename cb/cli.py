"""Entry point.  python -m cb.cli <command> [...]

Commands
  setup                 download data + checkpoints (into --cache)
  bench                 measure throughput on this GPU and project the hours of every run
  session1 / session2   the two Kaggle sessions (see README); resumable, time-capped, use all GPUs
  worker                internal: one GPU's share of a job list
  analyze               offline analysis of everything found -> runs/analysis/report.md
  geometry              G1-G4 geometry baselines (CPU, saved thoughts only) -> runs/geometry/report.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ENV_START = "CB_SESSION_START"


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def roots_from(args):
    prev = list(args.prev or [])
    if not prev:
        prev = sorted(set(glob.glob("/kaggle/input/*/runs") + glob.glob("/kaggle/input/*/*/runs")
                          + glob.glob("/kaggle/input/*/*/*/runs") + glob.glob("/kaggle/input/*/*/*/*/runs")))
    return [args.out] + [p for p in prev if Path(p).resolve() != Path(args.out).resolve()]


# ---------------------------------------------------------------- job lists
def pool_jobs(phase, default_p=None):
    from . import prereg as R
    from .pool import PoolSpec
    J = []
    if phase == "s1a":
        J.append(("pool", PoolSpec("coconut", "test", "latent", 0.0, 0, 1, rows_per_batch=512), "fp16"))
        J.append(("pool", PoolSpec("coconut", "test", "latent", 0.0, 0, 1, rows_per_batch=256, save_thoughts=False,
                                   tag="fp32"), "fp32"))
        J.append(("pool", PoolSpec("cot", "test", "text", 0.0, 0, 1, max_new=64, rows_per_batch=512,
                                   save_thoughts=False), "fp16"))
        for s in R.SEEDS:
            for p in R.P_GRID:
                J.append(("pool", PoolSpec("coconut", "test", "latent", p, s, R.N_POOL), "fp16"))
        for T in R.TEMPS:
            J.append(("pool", PoolSpec("cot", "test", "text", T, 0, R.N_POOL, max_new=64, save_thoughts=False,
                                       rows_per_batch=768), "fp16"))
        J.append(("pool", PoolSpec("coconut_you", "test", "latent", 0.0, 0, 1, rows_per_batch=512,
                                   save_thoughts=False), "fp16"))
        for p in (0.1, 0.2):
            J.append(("pool", PoolSpec("coconut_you", "test", "latent", p, 0, R.N_POOL, save_thoughts=False), "fp16"))
    elif phase == "s1b":
        J.append(("pool", PoolSpec("coconut", "train", "latent", default_p, 0, R.N_TRAIN_POOL,
                                   save_thoughts=False, subset=R.TRAIN_SUBSET), "fp16"))
        J.append(("annotate", "valid", None))
        J.append(("annotate", "train", None))
    elif phase == "s2a":
        J.append(("annotate", "valid", None))
        J.append(("annotate", "train", None))
    return J


def run_jobs(args, phase):
    import torch
    from . import data as D
    from . import models as M
    from . import prereg as R
    from .budget import Deadline
    from .pool import is_done, run_pool
    from .reward import run_annotation
    roots = roots_from(args)
    deadline = Deadline(args.hours)
    if args.phase_hours:
        deadline = deadline.child(args.phase_hours)
    default_p = _default_p(roots)
    bundles = {}

    def get(ckpt, dtype):
        key = (ckpt, dtype)
        if key not in bundles:
            bundles.clear()
            torch.cuda.empty_cache()
            bundles[key] = M.load(ckpt, args.cache, device=device(),
                                  dtype=torch.float32 if dtype == "fp32" else torch.float16)
        return bundles[key]

    for job in pool_jobs(phase, default_p):
        if deadline.expired(margin_min=5):
            print("[worker] session deadline reached; everything so far is saved and resumable")
            break
        if job[0] == "pool":
            spec, dtype = job[1], job[2]
            if is_done(roots, spec.name):
                continue
            print(f"[worker {args.worker}] pool {spec.name}", flush=True)
            run_pool(spec, get(spec.ckpt, dtype), args.data, args.out, roots, args.worker, args.nworkers, deadline)
        elif job[0] == "annotate":
            split = job[1]
            rows = D.load(args.data, split)
            if split == "train":
                rows = D.train_subset(rows, R.ANNOTATE_MAX_QUESTIONS, seed=1234)
            print(f"[worker {args.worker}] annotate {split} ({len(rows)} questions) at p={default_p}", flush=True)
            run_annotation(get("coconut", "fp16"), rows, split, default_p, args.out, roots,
                           args.worker, args.nworkers, deadline)


def _default_p(roots):
    for r in roots:
        f = Path(r) / "decisions" / "default_p.json"
        if f.exists():
            return json.loads(f.read_text())["default_p"]
    return None


# ---------------------------------------------------------------- orchestration
def spawn(args, extra, gpus):
    procs = []
    for i, g in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        cmd = [sys.executable, "-m", "cb.cli"] + extra + ["--worker", str(i), "--nworkers", str(len(gpus))] + common(args)
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        print(f"[session] worker exit codes {codes}")


def common(args):
    c = ["--out", args.out, "--data", args.data, "--cache", args.cache, "--hours", str(args.hours)]
    for p in args.prev or []:
        c += ["--prev", p]
    return c


def n_gpus():
    import torch
    return max(1, torch.cuda.device_count())


def prefetch(args):
    """Download data and all checkpoints once, before workers start (avoids concurrent downloads)."""
    from . import data as D
    from . import models as M
    D.download(args.data)
    for k in M.CHECKPOINTS:
        M.download(k, args.cache)


def session1(args):
    from . import analyze
    os.environ.setdefault(ENV_START, str(time.time()))
    prefetch(args)
    gpus = list(range(n_gpus()))
    print(f"[session1] {len(gpus)} GPU(s); budget {args.hours} h; earlier outputs: {roots_from(args)[1:]}", flush=True)
    if not (Path(args.out) / "bench.json").exists():
        subprocess.run([sys.executable, "-m", "cb.cli", "bench"] + common(args), check=False)
    spawn(args, ["worker", "--phase", "s1a"], gpus)
    res = analyze.run(roots_from(args), args.out, stage="choose_p")
    if res.get("default_p") is None:
        print("[session1] B2 incomplete (deadline?) -> rerun session1 next time; it resumes.")
        return
    print(f"[session1] default p = {res['default_p']}; Gate 1 = {res['anchors'].get('gate1')}", flush=True)
    spawn(args, ["worker", "--phase", "s1b"], gpus)
    analyze.run(roots_from(args), args.out)


def session2(args):
    from . import analyze
    from . import data as D
    from . import prereg as R
    os.environ.setdefault(ENV_START, str(time.time()))
    prefetch(args)
    gpus = list(range(n_gpus()))
    roots = roots_from(args)
    if _default_p(roots) is None:
        raise SystemExit("default p not decided yet: finish session1 first")
    # 1) finish annotation (capped)
    spawn(args, ["worker", "--phase", "s2a", "--phase-hours", str(R.ANNOTATE_SESSION2_MAX_H)], gpus)
    # 2) train PRM and ORM (parallel on 2 GPUs, else one after the other)
    if len(gpus) >= 2:
        procs = []
        for kind, g in (("prm", gpus[0]), ("orm", gpus[1])):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            procs.append(subprocess.Popen([sys.executable, "-m", "cb.cli", "train-rm", "--kind", kind,
                                           "--budget", str(R.RM_TRAIN_MAX_H)] + common(args), env=env))
        [p.wait() for p in procs]
    else:
        for kind, share in (("prm", 0.6), ("orm", 0.4)):
            subprocess.run([sys.executable, "-m", "cb.cli", "train-rm", "--kind", kind,
                            "--budget", str(R.RM_TRAIN_MAX_H * 2 * share)] + common(args), check=False)
    # 3) score the default-p pools
    procs = []
    for kind, g in (("prm", gpus[0]), ("orm", gpus[-1])):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        procs.append(subprocess.Popen([sys.executable, "-m", "cb.cli", "score-rm", "--kind", kind] + common(args),
                                      env=env))
        if len(gpus) == 1:
            procs[-1].wait()
    [p.wait() for p in procs]
    analyze.run(roots_from(args), args.out)


def train_rm(args):
    import torch
    from . import data as D
    from . import models as M
    from .budget import Deadline
    from .reward import load_annotations, train_reward
    roots = roots_from(args)
    p = _default_p(roots)
    out_dir = Path(args.out) / "reward" / args.kind
    if any((Path(r) / "reward" / args.kind / "DONE").exists() for r in roots):
        print(f"[train-rm] {args.kind} already trained"); return
    train, train_th = load_annotations(roots, f"train__p{p:.2f}")
    valid, valid_th = load_annotations(roots, f"valid__p{p:.2f}")
    need = set(train["qid"].astype(int))
    q = {r["qid"]: r["question"] for r in D.load(args.data, "train") if r["qid"] in need}
    qv = {r["qid"]: r["question"] for r in D.load(args.data, "valid")}
    # valid qids collide with train qids -> offset them
    valid = valid.assign(qid=valid["qid"] + 10_000_000)
    q.update({k + 10_000_000: v for k, v in qv.items()})
    print(f"[train-rm] {args.kind}: {len(train)} annotated paths from {train['qid'].nunique()} questions", flush=True)
    bundle = M.load("coconut", args.cache, device=device(), dtype=torch.float16)
    sd, _ = M.hf_state_dict("coconut", args.cache)
    deadline = Deadline(args.hours).child(args.budget)
    train_reward(args.kind, sd, bundle, train, train_th, valid, valid_th, q, out_dir, deadline,
                 device=device(), **(args.rm_kwargs or {}))


def score_rm(args):
    import torch
    from . import data as D
    from . import models as M
    from . import prereg as R
    from .analyze import b2_name
    from .pool import is_done, load_meta, load_thoughts
    from .reward import score_pool
    roots = roots_from(args)
    p = _default_p(roots)
    ck = None
    for r in roots:
        f = Path(r) / "reward" / args.kind / "best.pt"
        if f.exists():
            ck = f; break
    if ck is None:
        print(f"[score-rm] no {args.kind} checkpoint"); return
    bundle = M.load("coconut", args.cache, device=device(), dtype=torch.float16)
    sd, _ = M.hf_state_dict("coconut", args.cache)
    q = {r["qid"]: r["question"] for r in D.load(args.data, "test")}
    out = Path(args.out) / "scores"; out.mkdir(parents=True, exist_ok=True)
    for s in R.SEEDS:
        name = b2_name(p, s)
        f = out / f"{name}__{args.kind}.parquet"
        if f.exists() or not is_done(roots, name):
            continue
        meta, th = load_thoughts(roots, name)
        full = load_meta(roots, name, columns=["qid", "sid", "answer_ids"])
        meta = meta.merge(full, on=["qid", "sid"], how="left")
        res = score_pool(args.kind, sd, bundle, ck, meta, th, q, device=device())
        res.to_parquet(f, index=False)
        print(f"[score-rm] {args.kind} {name}", flush=True)


def bench(args):
    """Times one shard of each kind and projects the cost of every planned run on this GPU."""
    import torch
    from . import data as D
    from . import models as M
    from . import prereg as R
    from .pool import PoolSpec, plan, run_one_shard, select_rows
    from .reward import annotate_group
    D.download(args.data)
    res = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    b = M.load("coconut", args.cache)
    spec = PoolSpec("coconut", "test", "latent", 0.1, 99, R.N_POOL)
    rows = select_rows(spec, args.data)
    by = {r["qid"]: r for r in rows}
    shards = plan(spec, rows, b.tok)
    run_one_shard(spec, b, by, shards[0], 0)                      # warm-up
    torch.cuda.synchronize(); t = time.time()
    mid = len(shards) // 2
    for j in (mid, len(shards) - 1):
        run_one_shard(spec, b, by, shards[j], j)
    torch.cuda.synchronize()
    per_shard = (time.time() - t) / 2
    res["b2_pool_minutes"] = per_shard * len(shards) / 60
    tr = D.train_subset(D.load(args.data, "train"), 96, 1234)
    torch.cuda.synchronize(); t = time.time()
    annotate_group(b, tr[:48], 0.1, 1); annotate_group(b, tr[48:], 0.1, 2)
    torch.cuda.synchronize()
    qps = 96 / (time.time() - t)
    res["annotation_questions_per_s"] = qps
    res["annotation_hours_for_cap"] = R.ANNOTATE_MAX_QUESTIONS / qps / 3600
    del b; torch.cuda.empty_cache()
    c = M.load("cot", args.cache)
    cs = PoolSpec("cot", "test", "text", 1.0, 99, R.N_POOL, max_new=64, save_thoughts=False, rows_per_batch=768)
    cby = {r["qid"]: r for r in select_rows(cs, args.data)}
    csh = plan(cs, list(cby.values()), c.tok)
    torch.cuda.synchronize(); t = time.time()
    run_one_shard(cs, c, cby, csh[len(csh) // 2], 0)
    torch.cuda.synchronize()
    res["b3_pool_minutes"] = (time.time() - t) * len(csh) / 60
    n_b2 = len(R.P_GRID) * len(R.SEEDS)
    res["projection_hours_single_gpu"] = {
        "B2 (12 pools)": n_b2 * res["b2_pool_minutes"] / 60,
        "B3 (2 pools)": 2 * res["b3_pool_minutes"] / 60,
        "R11": res["b2_pool_minutes"] * (R.TRAIN_SUBSET * R.N_TRAIN_POOL) / (1319 * R.N_POOL) / 60,
        "R12 annotation (cap)": res["annotation_hours_for_cap"],
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "bench.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command")
    ap.add_argument("--out", default="/kaggle/working/runs" if Path("/kaggle").exists() else "runs")
    ap.add_argument("--prev", action="append", help="earlier sessions' runs/ dirs (default: /kaggle/input/*/runs)")
    ap.add_argument("--data", default="/tmp/cb_data")
    ap.add_argument("--cache", default="/tmp/cb_ckpt")
    ap.add_argument("--hours", type=float, default=11.4, help="whole-session wall-clock budget")
    ap.add_argument("--phase", default=None)
    ap.add_argument("--phase-hours", type=float, default=None)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--kind", default="prm")
    ap.add_argument("--budget", type=float, default=6.5)
    args = ap.parse_args()
    args.rm_kwargs = None
    if args.command in ("session1", "session2"):
        os.environ.setdefault(ENV_START, str(time.time()))
    cmd = args.command
    if cmd == "setup":
        from . import data as D
        from . import models as M
        D.download(args.data)
        for k in M.CHECKPOINTS:
            print(k, M.download(k, args.cache))
    elif cmd == "bench":
        bench(args)
    elif cmd == "worker":
        run_jobs(args, args.phase)
    elif cmd == "session1":
        session1(args)
    elif cmd == "session2":
        session2(args)
    elif cmd == "train-rm":
        train_rm(args)
    elif cmd == "score-rm":
        score_rm(args)
    elif cmd == "geometry":
        from . import geometry
        geometry.run(roots_from(args), args.out)
    elif cmd == "analyze":
        from . import analyze
        analyze.run(roots_from(args), args.out)
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
