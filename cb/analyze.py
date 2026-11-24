"""Offline analysis of saved pools -> runs/analysis/{results.json, report.md}.
All analysis choices are fixed in cb/prereg.py and DECISIONS.md before any pool exists."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import prereg as R
from . import stats as S
from .pool import PoolSpec, find_pool, is_done, load_meta, load_thoughts


def _name(**kw):
    return PoolSpec(**kw).name


def det_name():
    return _name(ckpt="coconut", split="test", kind="latent", setting=0.0, seed=0, n=1)


def b2_name(p, seed, ckpt="coconut"):
    return _name(ckpt=ckpt, split="test", kind="latent", setting=p, seed=seed, n=R.N_POOL)


def _mean_sd(x):
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "per_seed": x.tolist()}


# ---------------------------------------------------------------- B0 / B1
def anchors(roots):
    out = {}
    for key, name in [("coconut_det", det_name()),
                      ("coconut_det_fp32", _name(ckpt="coconut", split="test", kind="latent", setting=0.0,
                                                 seed=0, n=1, save_thoughts=False, tag="fp32")),
                      ("cot_greedy", _name(ckpt="cot", split="test", kind="text", setting=0.0, seed=0, n=1,
                                           max_new=64, save_thoughts=False)),
                      ("coconut_you_det", _name(ckpt="coconut_you", split="test", kind="latent", setting=0.0,
                                                seed=0, n=1, save_thoughts=False))]:
        if not is_done(roots, name):
            continue
        df = load_meta(roots, name)
        c = df["correct"].to_numpy()
        rng = np.random.default_rng(0)
        out[key] = {"acc": S.ci(c.astype(float), S.boot_idx(len(c), rng)),
                    "parse_empty": float((df["answer"] == "").mean()),
                    "unfinished": float((~df["finished"]).mean()),
                    "string_vs_numeric_disagree": int((df["correct"] != df["correct_num"]).sum())}
    if "coconut_det" in out and "coconut_det_fp32" in out:
        a = load_meta(roots, det_name())["correct"].to_numpy()
        b = load_meta(roots, _name(ckpt="coconut", split="test", kind="latent", setting=0.0, seed=0, n=1,
                                   save_thoughts=False, tag="fp32"))["correct"].to_numpy()
        out["fp16_vs_fp32"] = {"disagree_questions": int((a != b).sum()),
                               "acc_diff": float(a.mean() - b.mean())}
    gates = {}
    if "coconut_det" in out:
        gates["coconut_det"] = abs(out["coconut_det"]["acc"][0] * 100 - R.TARGET_COCONUT) <= R.GATE_TOL
    if "cot_greedy" in out:
        gates["cot_greedy"] = abs(out["cot_greedy"]["acc"][0] * 100 - R.TARGET_COT) <= R.GATE_TOL
    if "fp16_vs_fp32" in out:
        gates["fp16_ok"] = abs(out["fp16_vs_fp32"]["acc_diff"] * 100) <= R.FP16_TOL
    out["gate1"] = gates
    return out


def parse_check_sample(roots, out_dir, k=50):
    """B0: random attempts with raw text, parsed answer and gold, for hand-checking."""
    rows = []
    for name in [det_name(), b2_name(0.1, 0)]:
        if is_done(roots, name):
            df = load_meta(roots, name)
            rows.append(df.sample(min(k, len(df)), random_state=0)[["qid", "sid", "text", "answer", "gold",
                                                                   "correct", "correct_num", "finished"]]
                        .assign(pool=name))
    if rows:
        out = pd.concat(rows)
        out["hand_check_ok"] = ""
        out.to_csv(Path(out_dir) / "B0_parse_check.csv", index=False)


# ---------------------------------------------------------------- B2
def b2_table(roots, ckpt="coconut", with_diversity=True):
    res = {}
    for p in R.P_GRID:
        per_seed = []
        for s in R.SEEDS:
            name = b2_name(p, s, ckpt)
            if not is_done(roots, name):
                continue
            df = load_meta(roots, name, columns=["qid", "sid", "answer", "correct"])
            qids, codes, cor, cc, _ = S.to_arrays(df)
            row = {"seed": s, "acc_n1": float(cor.mean())}
            for N in R.N_LIST:
                row[f"pass@{N}"] = float(S.pass_at_n(cor, N).mean())
            bd, _ = S.breakdown(codes, cor, cc, None)
            row.update({"distinct_of_64": float(np.mean([len(x) for x in cc])),
                        "answer_entropy": S.answer_entropy(codes),
                        "mean_correct_of_64": bd["mean_correct_of_n"],
                        "top_wrong_beats_correct": bd["top_wrong_beats_correct_rate"]})
            if with_diversity and s == R.SEEDS[0] and find_pool(roots, name) and \
                    list(find_pool(roots, name).glob("*.thoughts.npy")):
                _, th = load_thoughts(roots, name)
                row["thought_diversity_by_step"] = S.thought_diversity(th, R.N_POOL)
            per_seed.append(row)
        if per_seed:
            keys = [k for k in per_seed[0] if k not in ("seed", "thought_diversity_by_step")]
            res[f"{p:.2f}"] = {k: _mean_sd([r[k] for r in per_seed]) for k in keys}
            res[f"{p:.2f}"]["seeds_done"] = [r["seed"] for r in per_seed]
            if "thought_diversity_by_step" in per_seed[0]:
                res[f"{p:.2f}"]["thought_diversity_by_step"] = per_seed[0]["thought_diversity_by_step"]
    return res


def choose_p(b2: dict) -> dict:
    """Pre-registered rule (prereg.P_RULE)."""
    cands = [(float(p), v[f"pass@{R.MAIN_N}"]["mean"], v["acc_n1"]["mean"]) for p, v in b2.items()
             if len(v["seeds_done"]) == len(R.SEEDS)]
    if len(cands) < len(R.P_GRID):
        return {"default_p": None, "reason": "not all p values have all seeds yet"}
    best = max(c[1] for c in cands)
    near = [c for c in cands if best - c[1] <= R.P_TIE_PASS32]
    chosen = max(near, key=lambda c: (c[2], -abs(c[0] - 0.1)))
    return {"default_p": chosen[0], f"pass@{R.MAIN_N}": chosen[1], "acc_n1": chosen[2],
            "rule": R.P_RULE, "candidates": cands}


# ---------------------------------------------------------------- B4 / B5 / B6
def _scores_for(roots, name):
    """Merge reward-model scores into the pool meta if present."""
    df = load_meta(roots, name)
    for r in roots:
        for kind in ("prm", "orm"):
            f = Path(r) / "scores" / f"{name}__{kind}.parquet"
            if f.exists() and not any(c.startswith(kind) for c in df.columns):
                df = df.merge(pd.read_parquet(f), on=["qid", "sid"], how="left")
    return df


def selection(roots, p, det_correct_by_qid=None):
    """B4 + B5 + B6 at the default p. Per-question accuracies are averaged over seeds before testing."""
    names = [b2_name(p, s) for s in R.SEEDS if is_done(roots, b2_name(p, s))]
    if not names:
        return {}
    rng = np.random.default_rng(R.ANALYSIS_SEED)
    per_seed = []
    for name in names:
        df = _scores_for(roots, name)
        score_cols = [c for c in R.SCORES if c in df.columns]
        qids, codes, cor, cc, sc = S.to_arrays(df, score_cols)
        best_of = [c for c in R.BEST_OF if c in sc]
        vote = [c for c in R.VOTE if c in sc]
        seed_res = {"qids": qids, "cor": cor, "codes": codes, "cc": cc, "scores": sc, "byN": {}}
        for N in R.N_LIST:
            acc, one, tie, distinct = S.selectors(codes, cor, cc, sc, N, rng, best_of=best_of, vote=vote)
            seed_res["byN"][N] = {"acc": acc, "one": one, "tie": tie, "distinct": distinct,
                                  "pass": S.pass_at_n(cor, N)}
        per_seed.append(seed_res)
    Q = len(per_seed[0]["qids"])
    bidx = S.boot_idx(Q, rng)
    out = {"pools": names, "byN": {}}
    for N in R.N_LIST:
        sel_names = list(per_seed[0]["byN"][N]["acc"])
        avg = {k: np.mean([sr["byN"][N]["acc"][k] for sr in per_seed], 0) for k in sel_names}
        pas = np.mean([sr["byN"][N]["pass"] for sr in per_seed], 0)
        row = {"pass": S.ci(pas, bidx)}
        for k in sel_names:
            row[k] = {"acc": S.ci(avg[k], bidx),
                      "per_seed": [float(sr["byN"][N]["acc"][k].mean()) for sr in per_seed]}
            if k != "majority":
                row[k]["vs_majority"] = S.paired(avg[k], avg["majority"], bidx)
                row[k]["efficiency"] = S.efficiency(avg[k], avg["majority"], pas, bidx)
                one_a = per_seed[0]["byN"][N]["one"][k]
                one_b = per_seed[0]["byN"][N]["one"]["majority"]
                row[k]["mcnemar_seed0_one_draw"] = S.mcnemar(one_a, one_b)
        row["majority_tie_rate"] = float(np.mean([sr["byN"][N]["tie"].mean() for sr in per_seed]))
        row["distinct_answers"] = float(np.mean([sr["byN"][N]["distinct"].mean() for sr in per_seed]))
        out["byN"][str(N)] = row
    # pre-registered test family at N = R.MAIN_N
    fam = {k: v["vs_majority"]["p_boot"] for k, v in out["byN"][str(R.MAIN_N)].items()
           if isinstance(v, dict) and "vs_majority" in v and k != R.PRIMARY}
    out["primary"] = {"comparison": f"{R.PRIMARY} vs majority at N={R.MAIN_N}",
                      "result": out["byN"][str(R.MAIN_N)].get(R.PRIMARY, {}).get("vs_majority")}
    out["exploratory_holm"] = S.holm(fam) if fam else {}
    # B6 breakdown on the full pool of 64, seed 0
    s0 = per_seed[0]
    det = None
    if det_correct_by_qid is not None:
        det = np.array([det_correct_by_qid[int(q)] for q in s0["qids"]])
    bd, masks = S.breakdown(s0["codes"], s0["cor"], s0["cc"], det)
    out["breakdown_seed0"] = bd
    if det is not None:
        oneN = s0["byN"][R.N_POOL]["one"]
        out["conversions_vs_deterministic_seed0_N64"] = {k: S.conversions(v, det) for k, v in oneN.items()}
    au = {}
    for k, v in s0["scores"].items():
        wq, nq = S.within_question_auroc(v, s0["cor"])
        au[k] = {"within_question": wq, "n_mixed": nq, "pooled": S.pooled_auroc(v, s0["cor"])}
        if k in ("conf", "orm", "prm_he_last"):
            au[k].update(S.classifier_report(v, s0["cor"]))
    out["score_quality_seed0"] = au
    return out


# ---------------------------------------------------------------- B3 / R11
def b3(roots, p):
    res = {}
    for T in R.TEMPS:
        name = _name(ckpt="cot", split="test", kind="text", setting=T, seed=0, n=R.N_POOL, max_new=64,
                     save_thoughts=False)
        if not is_done(roots, name):
            continue
        df = load_meta(roots, name)
        qids, codes, cor, cc, sc = S.to_arrays(df, ["mean_logp"])
        rng = np.random.default_rng(R.ANALYSIS_SEED)
        row = {"acc_n1": float(cor.mean()), "tokens_per_attempt": float(df["ntok"].mean())}
        for N in R.N_LIST:
            acc, _, _, _ = S.selectors(codes, cor, cc, sc, N, rng, best_of=["mean_logp"])
            row[f"N{N}"] = {"pass": float(S.pass_at_n(cor, N).mean()), "majority": float(acc["majority"].mean()),
                            "best_mean_logp": float(acc["best:mean_logp"].mean())}
        res[f"T{T}"] = row
    if p is not None and is_done(roots, b2_name(p, 0)):
        df = load_meta(roots, b2_name(p, 0), columns=["ntok"])
        res["coconut_forward_passes_per_attempt"] = float(6 + 1 + df["ntok"].mean())
    return res


def r11(roots, p):
    if p is None:
        return {}
    name = _name(ckpt="coconut", split="train", kind="latent", setting=p, seed=0, n=R.N_TRAIN_POOL,
                 save_thoughts=False, subset=R.TRAIN_SUBSET)
    if not is_done(roots, name):
        return {}
    tr = load_meta(roots, name)["correct"].mean()
    te = load_meta(roots, b2_name(p, 0), columns=["correct"])["correct"].mean()
    return {"train_attempt_acc": float(tr), "test_attempt_acc": float(te), "gap": float(tr - te)}


# ---------------------------------------------------------------- driver
def run(roots, out_root, stage="all"):
    out_dir = Path(out_root) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"anchors": anchors(roots)}
    parse_check_sample(roots, out_dir)
    res["b2"] = b2_table(roots)
    res["choose_p"] = choose_p(res["b2"])
    dec = Path(out_root) / "decisions"
    dec.mkdir(parents=True, exist_ok=True)
    fixed = None
    for r in roots:
        f = Path(r) / "decisions" / "default_p.json"
        if f.exists():
            fixed = json.loads(f.read_text())
            break
    if fixed is None and res["choose_p"]["default_p"] is not None:
        fixed = res["choose_p"]
        (dec / "default_p.json").write_text(json.dumps(fixed, indent=1))
    if fixed is not None and not (dec / "default_p.json").exists():
        (dec / "default_p.json").write_text(json.dumps(fixed, indent=1))
    p = fixed["default_p"] if fixed else None
    res["default_p"] = p
    if stage == "choose_p":
        (out_dir / "results_choose_p.json").write_text(json.dumps(res, indent=1, default=float))
        return res
    if p is not None:
        det = None
        if is_done(roots, det_name()):
            d = load_meta(roots, det_name())
            det = dict(zip(d["qid"].astype(int), d["correct"].astype(bool)))
        res["selection"] = selection(roots, p, det)
        row = res["selection"].get("byN", {}).get(str(R.MAIN_N))
        if row:
            gap = row["pass"][0] - row["majority"]["acc"][0]
            effs = {k: v["efficiency"] for k, v in row.items() if isinstance(v, dict) and "efficiency" in v
                    and k != "random"}
            res["gates"] = {"gate2_headroom_points": 100 * gap, "gate2_pass": gap >= R.GATE2_MIN_GAP,
                            "gate3_pass": not any(e["lo"] > 0.5 for e in effs.values()),
                            "gate3_efficiencies": effs}
    res["b3"] = b3(roots, p)
    res["r11"] = r11(roots, p)
    res["b2_you"] = b2_table(roots, ckpt="coconut_you", with_diversity=False)
    for r in roots:
        for f in sorted((Path(r) / "reward").glob("*/train_log.json")):
            res.setdefault("reward_training", {})[f.parent.name] = json.loads(f.read_text())
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=float))
    from .report import write_report
    write_report(res, out_dir / "report.md")
    print(f"[analyze] wrote {out_dir}/results.json and report.md")
    return res
