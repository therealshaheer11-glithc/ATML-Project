"""End-to-end CPU smoke test of session1 + session2 logic with a tiny random model and tiny data.
python tests/smoke_pipeline.py   (no GPU, no internet)"""
import json
import random
import shutil
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_core as T  # noqa: E402

from cb import analyze, cli, models, prereg as R  # noqa: E402

root = Path("/tmp/cb_smoke"); shutil.rmtree(root, ignore_errors=True)
data = root / "data"; data.mkdir(parents=True)
random.seed(0)


def rows(n):
    return [{"question": f"q{i} " + "x" * random.randint(3, 12), "steps": ["<<1+1=2>>"],
             "answer": str(random.randint(0, 9))} for i in range(n)]


for split, n in (("test", 24), ("train", 60), ("valid", 10)):
    (data / f"gsm_{split}.json").write_text(json.dumps(rows(n)))

# tiny config
R.P_GRID, R.SEEDS, R.N_POOL, R.N_LIST, R.MAIN_N = [0.1, 0.2], [0, 1], 8, [1, 2, 4, 8], 4
R.TEMPS, R.TRAIN_SUBSET, R.N_TRAIN_POOL, R.ANNOTATE_MAX_QUESTIONS = [1.0], 10, 4, 12
hf = T.hf_model(std=0.08)
# a random model decodes the same tokens whatever the thoughts; make its readout sensitive so pools
# contain mixed questions and every analysis branch is exercised
from cb.gpt2 import FastGPT2  # noqa: E402
_Rm = torch.randn(64, T.V, generator=torch.Generator().manual_seed(3))
FastGPT2.logits = lambda self, h: torch.sin(h.float() * 300) @ _Rm.to(h.device)
models.load = lambda name, cache, device="cpu", dtype=None: T.bundle(hf)
models.hf_state_dict = lambda name, cache: (hf.state_dict(), None)
_jobs = cli.pool_jobs


def small_jobs(phase, p=None):
    J = _jobs(phase, p)
    for j in J:
        if j[0] == "pool":
            j[1].rows_per_batch = 32
    return J


cli.pool_jobs = small_jobs

args = types.SimpleNamespace(out=str(root / "runs"), prev=[str(root / "none")], data=str(data), cache="x",
                             hours=1.0, phase_hours=None, worker=0, nworkers=1, kind="prm", budget=0.2,
                             rm_kwargs={"batch": 8, "micro": 4, "warmup": 2, "epochs": 2, "eval_every": 5})
for w in (0, 1):                       # two "GPUs"
    args.worker, args.nworkers = w, 2
    cli.run_jobs(args, "s1a")
res = analyze.run(cli.roots_from(args), args.out, stage="choose_p")
print("default p:", res["default_p"], "gate1:", res["anchors"]["gate1"])
assert res["default_p"] in R.P_GRID
for w in (0, 1):
    args.worker = w
    cli.run_jobs(args, "s1b")
for kind in ("prm", "orm"):
    args.kind = kind
    cli.train_rm(args)
    cli.score_rm(args)
res = analyze.run(cli.roots_from(args), args.out)
sel = res["selection"]
print("primary:", sel["primary"])
print("N=4 selectors:", {k: round(v["acc"][0], 3) for k, v in sel["byN"]["4"].items() if isinstance(v, dict) and "acc" in v})
print("breakdown:", sel["breakdown_seed0"])
print("score quality:", {k: round(v["within_question"], 3) for k, v in sel["score_quality_seed0"].items()})
assert "best:prm_he_last" in sel["byN"]["4"] and "best:orm" in sel["byN"]["4"]
# replay check on a saved shard
from cb.pool import load_meta, replay_shard  # noqa: E402
name = analyze.b2_name(res["default_p"], 0)
out = replay_shard(cli.roots_from(args), name, 0, T.bundle(hf), str(data))
meta = load_meta(cli.roots_from(args), name)
m0 = meta[meta["shard"] == 0].sort_values("row")
assert out["ids"].shape[0] == len(m0)
print("report:\n", open(Path(args.out) / "analysis" / "report.md").read()[:1500])
from cb import geometry as G  # noqa: E402
G.RANKS = [8, 16]
G.PCA_FIT_PATHS = 50
g = G.run(cli.roots_from(args), args.out)
print(open(Path(args.out) / "geometry" / "report.md").read())
print("SMOKE OK")
