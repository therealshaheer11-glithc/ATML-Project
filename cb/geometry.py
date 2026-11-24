"""Geometry baselines G1–G4: existing trajectory scores in the full space and in PCA / random subspaces.

No training and no new model runs: everything is computed from thoughts already saved in session 1.

Scores (one value per path, from thoughts h1..h6):
  Proposal / Li et al. / Vilas et al.
    total_distance     sum_t ||h_t - h_{t-1}||                       (Vilas: Cumulative Change)
    velocity_entropy   entropy of p_t = v_t / sum v                   (Li: Latent Velocity Entropy)
    net_displacement   ||h6 - h1||                                     (Vilas: Net Change, up to a constant)
    alignment          mean_t cos(h_t - h_{t-1}, h6 - h1)             (Vilas: Aligned Change)
  Wang et al. (LatentITS geometry.py, Eq. 8-12)
    compactness        sqrt(mean_t ||h_t - mean(h)||^2)
    straightness       ||h6 - h1|| / sum_t ||h_t - h_{t-1}||
    local_smoothness   mean_t cos(h_t, h_{t+1})
    curvature          sum_t angle(step_t, step_{t+1})

Spaces: full (768 dims); PCA_r and random_r for r in RANKS. A subspace is applied per thought position as
h_t -> B_t B_t^T h_t, kept in 768 coordinates so steps between positions are comparable. This is the same
form the gradient basis B_t will take later. PCA bases are fit per position on train-question thoughts (no test
data, no labels); random bases are QR of Gaussian noise with a fixed seed.

Direction of "good" is fixed from the papers before looking at results (SIGN below).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import prereg as R
from . import stats as S

RANKS = [32, 64, 128, 256]
METRICS = ["total_distance", "velocity_entropy", "net_displacement", "alignment",
           "compactness", "straightness", "local_smoothness", "curvature"]
# +1: higher = more likely correct, -1: lower = more likely correct.
SIGN = {
    "total_distance": -1,    # Vilas: cumulative change larger for incorrect
    "velocity_entropy": -1,  # Li: low entropy (concentrated movement) = good
    "net_displacement": +1,  # Vilas: net change larger for correct
    "alignment": +1,         # Vilas: aligned change larger for correct
    # Wang's geometric loss pushes correct paths to HIGHER values of all four metrics
    "compactness": +1, "straightness": +1, "local_smoothness": +1, "curvature": +1,
}
SOURCE = {"total_distance": "Vilas / proposal (i)", "velocity_entropy": "Li / proposal (ii)",
          "net_displacement": "Vilas / proposal (iii)", "alignment": "Vilas / proposal (iii)",
          "compactness": "Wang", "straightness": "Wang", "local_smoothness": "Wang", "curvature": "Wang"}
REFERENCES = ["conf", "prm_he_last", "orm"]
PCA_FIT_PATHS = 40_000
SEED = 2024


# ---------------------------------------------------------------- metrics
def trajectory_metrics(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """x: [n, T, D] float32 -> {metric: [n]}"""
    eps = 1e-12
    steps = x[:, 1:] - x[:, :-1]                                  # [n, T-1, D]
    v = steps.norm(dim=-1)                                        # [n, T-1]
    total = v.sum(-1)
    p = v / total.clamp_min(eps)[:, None]
    entropy = -(p * torch.log(p.clamp_min(eps))).sum(-1)
    drift = x[:, -1] - x[:, 0]
    disp = drift.norm(dim=-1)
    align = torch.nn.functional.cosine_similarity(steps, drift[:, None].expand_as(steps), dim=-1).mean(-1)
    center = x.mean(1, keepdim=True)
    compact = (x - center).square().sum(-1).mean(-1).clamp_min(eps).sqrt()
    smooth = torch.nn.functional.cosine_similarity(x[:, :-1], x[:, 1:], dim=-1).mean(-1)
    cos_steps = torch.nn.functional.cosine_similarity(steps[:, :-1], steps[:, 1:], dim=-1)
    curv = cos_steps.clamp(-1 + 1e-7, 1 - 1e-7).acos().sum(-1)
    return {"total_distance": total, "velocity_entropy": entropy, "net_displacement": disp,
            "alignment": align, "compactness": compact, "straightness": disp / total.clamp_min(eps),
            "local_smoothness": smooth, "curvature": curv}


# ---------------------------------------------------------------- bases
def fit_pca(train_thoughts: np.ndarray, ranks, n_fit=PCA_FIT_PATHS, seed=SEED):
    """Per-position PCA on train thoughts. Returns {r: [T, D, r]} and explained-variance curves."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_thoughts), size=min(n_fit, len(train_thoughts)), replace=False)
    x = torch.from_numpy(np.asarray(train_thoughts[np.sort(idx)], dtype=np.float32))
    T = x.shape[1]
    bases, explained = {r: [] for r in ranks}, []
    for t in range(T):
        xt = x[:, t] - x[:, t].mean(0)
        cov = xt.T @ xt / (len(xt) - 1)
        evals, evecs = torch.linalg.eigh(cov.double())
        order = torch.argsort(evals, descending=True)
        evals, evecs = evals[order].float(), evecs[:, order].float()
        explained.append((evals.cumsum(0) / evals.sum()).tolist())
        for r in ranks:
            bases[r].append(evecs[:, :r])
    return {r: torch.stack(b) for r, b in bases.items()}, explained


def random_bases(ranks, T, D, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for r in ranks:
        qs = []
        for _ in range(T):
            q, _ = torch.linalg.qr(torch.randn(D, r, generator=g))
            qs.append(q)
        out[r] = torch.stack(qs)
    return out


def project(x: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """x [n, T, D], B [T, D, r] -> B_t B_t^T x_t for each t, in D coordinates."""
    coords = torch.einsum("ntd,tdr->ntr", x, B)
    return torch.einsum("ntr,tdr->ntd", coords, B)


def spaces(pca, rnd):
    out = {"full": None}
    for r in RANKS:
        out[f"pca{r}"] = pca[r]
    for r in RANKS:
        out[f"rand{r}"] = rnd[r]
    return out


def score_pool(thoughts: np.ndarray, space_bases: dict, chunk=8192):
    """thoughts [N, T, D] (memmap ok) -> DataFrame with one column per metric@space, plus energy kept."""
    cols: dict[str, list] = {}
    for s in range(0, len(thoughts), chunk):
        x = torch.from_numpy(np.asarray(thoughts[s:s + chunk], dtype=np.float32))
        norm2 = x.square().sum(-1).sum(-1)
        for name, B in space_bases.items():
            xs = x if B is None else project(x, B)
            for m, v in trajectory_metrics(xs).items():
                cols.setdefault(f"{m}@{name}", []).append(v.numpy())
            if B is not None:
                cols.setdefault(f"energy_kept@{name}", []).append((xs.square().sum(-1).sum(-1) / norm2).numpy())
    return pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})


# ---------------------------------------------------------------- evaluation
def within_q(score, cor):
    """Per-question AUROC (mixed questions only) -> array over mixed questions."""
    out = []
    for i in range(len(cor)):
        if 0 < cor[i].sum() < cor.shape[1]:
            out.append(S.auroc(score[i][cor[i]], score[i][~cor[i]]))
    return np.array(out)


def spearman_within(a, b):
    from scipy.stats import spearmanr
    vals = []
    for i in range(len(a)):
        if np.std(a[i]) > 0 and np.std(b[i]) > 0:
            vals.append(spearmanr(a[i], b[i]).statistic)
    return float(np.nanmean(vals)) if vals else float("nan")


def run(roots, out_root, data_dir=None):
    from .analyze import _scores_for, b2_name
    from .pool import is_done, load_thoughts
    from .reward import load_annotations
    out = Path(out_root) / "geometry"
    out.mkdir(parents=True, exist_ok=True)
    p = None
    for r in roots:
        f = Path(r) / "decisions" / "default_p.json"
        if f.exists():
            p = json.loads(f.read_text())["default_p"]
            break
    if p is None:
        raise SystemExit("default p not found: attach session-1 output")
    names = [b2_name(p, s) for s in R.SEEDS if is_done(roots, b2_name(p, s))]
    print(f"[geometry] p={p}, pools: {names}", flush=True)

    # bases
    _, train_th = load_annotations(roots, f"train__p{p:.2f}")
    print(f"[geometry] fitting PCA on {min(PCA_FIT_PATHS, len(train_th))} of {len(train_th)} train paths", flush=True)
    pca, explained = fit_pca(train_th, RANKS)
    del train_th
    T, D = pca[RANKS[0]].shape[0], pca[RANKS[0]].shape[1]
    rnd = random_bases(RANKS, T, D)
    sp = spaces(pca, rnd)
    diag = {"pca_explained_variance_at_rank": {
        f"t{t + 1}": {r: explained[t][r - 1] for r in RANKS} for t in range(T)}}

    # score every pool
    per_seed = []
    for name in names:
        meta, th = load_thoughts(roots, name)
        sc = score_pool(th, sp)
        full = _scores_for(roots, name)
        keep = ["qid", "sid", "correct", "answer"] + [c for c in REFERENCES if c in full.columns]
        df = pd.concat([meta[["qid", "sid"]].reset_index(drop=True), sc], axis=1)
        df = df.merge(full[keep], on=["qid", "sid"], how="left")
        df.to_parquet(out / f"scores__{name}.parquet", index=False)
        print(f"[geometry] scored {name}: {len(df)} paths, {sc.shape[1]} columns", flush=True)
        per_seed.append(df)

    configs = [f"{m}@{s}" for m in METRICS for s in sp]
    refs = [c for c in REFERENCES if c in per_seed[0].columns and per_seed[0][c].notna().all()]
    energy = {s: float(np.mean([d[f"energy_kept@{s}"].mean() for d in per_seed])) for s in sp if s != "full"}

    # AUROC (within question; sign applied so > 0.5 means the paper's direction works)
    rng = np.random.default_rng(R.ANALYSIS_SEED)
    auroc = {}
    for c in configs + refs:
        sign = SIGN.get(c.split("@")[0], 1)
        wq, pooled, corr_conf = [], [], []
        for d in per_seed:
            qids, codes, cor, cc, scd = S.to_arrays(d, [c] + (["conf"] if "conf" in d else []))
            v = sign * scd[c]
            wq.append(within_q(v, cor))
            pooled.append(S.pooled_auroc(v, cor))
            if "conf" in scd and c != "conf":
                corr_conf.append(spearman_within(v, scd["conf"]))
        allq = np.concatenate(wq)
        b = rng.integers(0, len(allq), size=(S.N_BOOT, len(allq)))
        auroc[c] = {"within_q": float(np.nanmean(allq)),
                    "lo": float(np.nanpercentile(np.nanmean(allq[b], 1), 2.5)),
                    "hi": float(np.nanpercentile(np.nanmean(allq[b], 1), 97.5)),
                    "pooled": float(np.mean(pooled)),
                    "spearman_with_conf": float(np.mean(corr_conf)) if corr_conf else None}

    # best-of-N selection
    sel = {}
    for N in R.N_LIST:
        accs = []
        for d in per_seed:
            d2 = d.copy()
            for c in configs:
                d2[c] = SIGN[c.split("@")[0]] * d2[c]
            qids, codes, cor, cc, scd = S.to_arrays(d2, configs + refs)
            acc, _, _, _ = S.selectors(codes, cor, cc, scd, N, rng, n_sub=100, best_of=configs + refs)
            acc["pass"] = S.pass_at_n(cor, N)
            accs.append(acc)
        sel[N] = {k: np.mean([a[k] for a in accs], 0) for k in accs[0]}
    Q = len(sel[R.N_LIST[0]]["majority"])
    bidx = S.boot_idx(Q, rng)
    main = sel[R.MAIN_N]
    results = {"p": p, "pools": names, "ranks": RANKS, "energy_kept": energy, "diagnostics": diag,
               "auroc": auroc, "selection": {}, "vs_majority_mainN": {}}
    for N, a in sel.items():
        results["selection"][str(N)] = {k: float(v.mean()) for k, v in a.items()}
    pvals = {}
    for k in configs + [f"best:{r}" for r in refs]:
        key = k if k.startswith("best:") else f"best:{k}"
        res = S.paired(main[key], main["majority"], bidx)
        res["efficiency"] = S.efficiency(main[key], main["majority"], main["pass"], bidx)
        results["vs_majority_mainN"][k] = res
        pvals[k] = res["p_boot"]
    holm = S.holm(pvals)
    for k in holm:
        results["vs_majority_mainN"][k]["p_holm"] = holm[k]
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    write_report(results, out / "report.md")
    print(f"[geometry] wrote {out}/report.md", flush=True)
    return results


def write_report(res, path):
    n = str(R.MAIN_N)
    sp = ["full"] + [f"pca{r}" for r in RANKS] + [f"rand{r}" for r in RANKS]
    L = ["# Geometry baselines (G1–G4)", "",
         f"Pools: {', '.join(res['pools'])} (p = {res['p']}). Every score is oriented so that higher = the "
         "direction the source paper says is better; AUROC > 0.5 means that direction holds.", "",
         f"## Within-question AUROC (mixed questions, {len(res['pools'])} seeds pooled)", "",
         "| score (source) | " + " | ".join(sp) + " |", "|---" * (len(sp) + 1) + "|"]
    for m in METRICS:
        L.append(f"| {m} ({SOURCE[m]}) | " + " | ".join(f"{res['auroc'][f'{m}@{s}']['within_q']:.3f}" for s in sp) + " |")
    L += ["", "References: " + ", ".join(f"{k} {v['within_q']:.3f}" for k, v in res["auroc"].items()
                                          if "@" not in k), "",
          f"## Best-of-N accuracy at N = {n} (%), seeds averaged", "",
          f"Majority {100 * res['selection'][n]['majority']:.1f} · Pass@{n} {100 * res['selection'][n]['pass']:.1f} · "
          + " · ".join(f"{k[5:]} {100 * v:.1f}" for k, v in res["selection"][n].items()
                       if k.startswith("best:") and "@" not in k), "",
          "| score | " + " | ".join(sp) + " |", "|---" * (len(sp) + 1) + "|"]
    for m in METRICS:
        L.append(f"| {m} | " + " | ".join(f"{100 * res['selection'][n][f'best:{m}@{s}']:.1f}" for s in sp) + " |")
    L += ["", f"## Top 10 configurations vs majority at N = {n} (paired bootstrap, Holm over all configurations)", "",
          "| configuration | gain (pts) | 95% CI | Holm p | efficiency |", "|---|---|---|---|---|"]
    top = sorted(res["vs_majority_mainN"].items(), key=lambda kv: -kv[1]["diff"])[:10]
    for k, v in top:
        L.append(f"| {k} | {100 * v['diff']:+.2f} | [{100 * v['lo']:+.2f}, {100 * v['hi']:+.2f}] | "
                 f"{v['p_holm']:.3f} | {v['efficiency']['eff']:.2f} |")
    L += ["", "## Energy kept by each subspace (share of ||h||², mean over paths)", "",
          " · ".join(f"{k} {v:.3f}" for k, v in res["energy_kept"].items()), "",
          "PCA variance explained per thought (train thoughts): "
          + " · ".join(f"{t}: " + "/".join(f"{x:.2f}" for x in v.values())
                       for t, v in res["diagnostics"]["pca_explained_variance_at_rank"].items())
          + f" (ranks {RANKS})", ""]
    Path(path).write_text("\n".join(L) + "\n")
