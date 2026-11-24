# Coconut baselines (Group 11)

Prerequisite runs B0–B6 for *Selecting Coconut Reasoning Paths with Answer-Sensitive Subspaces*.
Nothing from our own method is in here. **All analysis choices are fixed in [`DECISIONS.md`](DECISIONS.md)
and `cb/prereg.py`.** Don't change them after results exist without logging the change.

## What runs where

| Session | What it does | Kaggle |
|---|---|---|
| **1** | setup + benchmark → B1 anchors → B2 (4 p × 3 seeds × 64 attempts) → B3 text CoT → regime check on You et al.'s checkpoint → **picks the default p** → R11 train-set pool → R12 annotation (until time runs out) | GPU T4 ×2, Internet ON |
| **2** | finish R12 annotation (≤ 4 h) → train PRM (GPU 0) and ORM (GPU 1) in parallel (≤ 6.5 h each) → score the test pools → final report | GPU T4 ×2, Internet ON, **session-1 output attached** |

Everything is **resumable and time-capped**. If a session stops early, run the same session again with the
previous output attached and it continues from the last finished shard. Nothing is ever recomputed.

## Kaggle steps

**Session 1.** New notebook → Settings: *Accelerator = GPU T4 x2*, *Internet = on*. One cell:

```bash
!git clone https://github.com/<you>/<repo>.git /kaggle/working/cb && cd /kaggle/working/cb && \
 pip install -q -r requirements.txt && python -m cb.cli session1 --hours 11.4
```

Use **Save Version → Save & Run All (Commit)**, so it runs in the background for up to 12 h and the output
(`/kaggle/working/runs`) is saved.

**Session 2.** New notebook (same settings). *Add Input → Your Work →* the session-1 notebook. Then:

```bash
!git clone https://github.com/<you>/<repo>.git /kaggle/working/cb && cd /kaggle/working/cb && \
 pip install -q -r requirements.txt && python -m cb.cli session2 --hours 11.4
```

Earlier outputs are found automatically under `/kaggle/input/*/runs`, or you can pass `--prev <dir>`.
If you need a third session (for example, if annotation didn't finish), attach **both** earlier outputs and run `session2` again.

> The clone lives in `/kaggle/working/cb`, so it also ends up in the output. That's harmless (a few hundred KB).
> Model and data caches go to `/tmp` and are not saved.

## After session 1: check by hand
1. `runs/analysis/report.md`: Gate 1 (anchors) and Gate 2 (headroom).
2. `runs/analysis/B0_parse_check.csv`: hand-check the 50+50 parsed answers and fill in `hand_check_ok`.
3. `runs/bench.json`: measured throughput and projected hours for every run on this GPU.

If Gate 1 fails, **stop** and debug before session 2 (see `DECISIONS.md` § Gates).

## Outputs
```
runs/
  bench.json                        throughput + projected hours
  pools/<name>/                     one directory per (checkpoint, split, p or T, seed, n)
     spec.json  shard_*.parquet  shard_*.thoughts.npy  DONE
  decisions/default_p.json          chosen by the pre-registered rule; never overwritten
  annotations/{valid,train}__p*/    R12 Monte Carlo labels + thoughts
  reward/{prm,orm}/best.pt          reward models + train_log.json
  scores/<pool>__{prm,orm}.parquet  reward scores for every attempt
  analysis/results.json, report.md, B0_parse_check.csv
```
Each pool row: `qid, sid, shard, row, text, answer, gold, correct, correct_num, finished, ntok, logp, conf,
answer_ids`, plus the six thoughts (fp16) in the matching `.npy`.
`cb.pool.load_meta / load_thoughts / replay_shard` read the pools back. `replay_shard(..., thought_hook=f)`
regenerates a shard with the same dropout masks and lets you intervene on any thought (for the later causal check).

## Geometry baselines (G1–G4)
New notebook, Accelerator **None**, Internet on, attach both session outputs (*Add Input → Your Work → Notebooks*):
```bash
!git clone https://github.com/<you>/<repo>.git /kaggle/working/cb && cd /kaggle/working/cb && \
 pip install -q -r requirements.txt && python -m cb.cli geometry
```
Output: `runs/geometry/report.md`, `results.json`, and `scores__<pool>.parquet` (every score for every path).

## Local commands
```bash
python -m cb.cli setup        # download data + checkpoints
python -m cb.cli bench        # throughput on this GPU
python -m cb.cli analyze      # rerun the offline analysis on whatever exists
python -m pytest -q tests     # CPU: FastGPT2 == HF GPT-2, sampler == official loop, stats == brute force
python tests/smoke_pipeline.py  # CPU end-to-end run of both sessions on a tiny random model
```

## Code map
| file | role |
|---|---|
| `cb/gpt2.py` | GPT-2 with a preallocated KV cache and explicit-RNG dropout at the same four sites as HF |
| `cb/sampling.py` | Coconut latent sampler (identical schedule to LatentITS), text CoT sampler, greedy decoding |
| `cb/pool.py` | sharded, resumable, replayable pools |
| `cb/reward.py` | R12: MC annotation, PRM/ORM training and scoring |
| `cb/stats.py` | Pass@N, subset selectors, bootstrap, McNemar, Holm, AUROC, breakdowns |
| `cb/analyze.py`, `cb/report.py` | B1–B6 tables → `report.md` |
| `cb/prereg.py` | every pre-registered constant |
