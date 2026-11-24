"""Offline analysis maths (CPU / numpy). Everything here works on saved pools only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata

N_SUBSETS = 200
N_BOOT = 2000


# ---------------------------------------------------------------- per-question arrays
def to_arrays(df: pd.DataFrame, score_cols=()):
    """df sorted by qid, sid with the same n per question.
    Returns qids [Q], codes [Q,n] (answer id within question), correct [Q,n] bool,
    code_correct list of [K_q] bool, scores {name: [Q,n]}."""
    df = df.sort_values(["qid", "sid"])
    qids = df["qid"].unique()
    n = len(df) // len(qids)
    assert n * len(qids) == len(df), "pool must have the same n for every question"
    ans = df["answer"].to_numpy().reshape(len(qids), n)
    cor = df["correct"].to_numpy().astype(bool).reshape(len(qids), n)
    codes = np.zeros((len(qids), n), dtype=np.int32)
    code_correct = []
    for i in range(len(qids)):
        u, inv = np.unique(ans[i], return_inverse=True)
        codes[i] = inv
        cc = np.zeros(len(u), dtype=bool)
        cc[inv[cor[i]]] = True
        code_correct.append(cc)
    scores = {c: df[c].to_numpy().astype(np.float64).reshape(len(qids), n) for c in score_cols}
    return qids, codes, cor, code_correct, scores


def pass_at_n(cor: np.ndarray, N: int) -> np.ndarray:
    """Unbiased estimator per question: 1 - C(n-c, N) / C(n, N)."""
    n = cor.shape[1]
    c = cor.sum(1)
    out = np.ones(len(c))
    for i, ci in enumerate(c):
        if n - ci < N:
            continue
        out[i] = 1.0 - np.prod(1.0 - N / np.arange(n - ci + 1, n + 1))
    return out


# ---------------------------------------------------------------- selectors on random subsets
def selectors(codes, cor, code_correct, scores: dict, N: int, rng: np.random.Generator,
              n_sub: int = N_SUBSETS, best_of=(), vote=()):
    """Expected accuracy per question of every selector, averaged over n_sub random subsets of size N.
    best_of: score names used as 'pick the highest-scoring attempt'.
    vote:    score names used as 'answer with the largest summed score' (score-weighted vote).
    Ties are broken uniformly at random. Also returns one fixed draw (binary) per selector for McNemar,
    the majority-vote tie rate and the mean number of distinct answers."""
    Q, n = cor.shape
    names = ["random", "majority"] + [f"best:{s}" for s in best_of] + [f"vote:{s}" for s in vote]
    acc = {k: np.zeros(Q) for k in names}
    one = {k: np.zeros(Q, dtype=bool) for k in names}
    tie = np.zeros(Q)
    distinct = np.zeros(Q)
    for i in range(Q):
        K = len(code_correct[i])
        idx = np.argsort(rng.random((n_sub, n)), axis=1)[:, :N]          # [S, N]
        cc = code_correct[i]
        onehot = np.eye(K, dtype=np.float64)[codes[i][idx]]               # [S, N, K]
        counts = onehot.sum(1)                                            # [S, K]
        noise = rng.random((n_sub, K)) * 0.5
        pick = (counts + noise).argmax(1)
        top = counts.max(1, keepdims=True)
        tie[i] = ((counts == top).sum(1) > 1).mean()
        distinct[i] = (counts > 0).sum(1).mean()
        res = {"random": cor[i][idx].mean(1), "majority": cc[pick]}
        for s in best_of:
            v = scores[s][i][idx] + rng.random((n_sub, N)) * 1e-9
            res[f"best:{s}"] = cor[i][idx][np.arange(n_sub), v.argmax(1)]
        for s in vote:
            w = (onehot * scores[s][i][idx][..., None]).sum(1) + rng.random((n_sub, K)) * 1e-9
            w = np.where(counts > 0, w, -np.inf)
            res[f"vote:{s}"] = cc[w.argmax(1)]
        for k in names:
            r = np.asarray(res[k], dtype=np.float64)
            acc[k][i] = r.mean()
            one[k][i] = r[0] > 0.5 if k != "random" else rng.random() < r.mean()
    return acc, one, tie, distinct


# ---------------------------------------------------------------- uncertainty
def boot_idx(Q, rng, B=N_BOOT):
    return rng.integers(0, Q, size=(B, Q))


def ci(values: np.ndarray, bidx: np.ndarray):
    m = values[bidx].mean(1)
    return float(values.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a: np.ndarray, b: np.ndarray, bidx: np.ndarray):
    """Mean difference a-b with bootstrap 95% CI and two-sided bootstrap p-value (question-level)."""
    d = a - b
    m = d[bidx].mean(1)
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)),
            "p_boot": float(min(1.0, max(p, 1.0 / len(m))))}


def mcnemar(a: np.ndarray, b: np.ndarray):
    """Exact McNemar on paired binary outcomes."""
    b01 = int((~a & b).sum())
    b10 = int((a & ~b).sum())
    if b01 + b10 == 0:
        return {"a_only": b10, "b_only": b01, "p": 1.0}
    return {"a_only": b10, "b_only": b01, "p": float(binomtest(b10, b01 + b10, 0.5).pvalue)}


def efficiency(sel, maj, pas, bidx):
    """(selector - majority) / (Pass@N - majority), computed on pooled means, bootstrapped."""
    num = (sel - maj)[bidx].mean(1)
    den = (pas - maj)[bidx].mean(1)
    r = num / np.where(np.abs(den) < 1e-12, np.nan, den)
    point = (sel - maj).mean() / (pas - maj).mean() if abs((pas - maj).mean()) > 1e-12 else np.nan
    return {"eff": float(point), "lo": float(np.nanpercentile(r, 2.5)), "hi": float(np.nanpercentile(r, 97.5))}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for k, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - k) * p))
        out[name] = running
    return out


# ---------------------------------------------------------------- scores as classifiers
def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def within_question_auroc(score: np.ndarray, cor: np.ndarray) -> tuple[float, int]:
    vals = [auroc(score[i][cor[i]], score[i][~cor[i]]) for i in range(len(cor))
            if 0 < cor[i].sum() < cor.shape[1]]
    return float(np.nanmean(vals)) if vals else np.nan, len(vals)


def pooled_auroc(score, cor):
    return auroc(score[cor], score[~cor])


def ece(prob: np.ndarray, label: np.ndarray, bins: int = 10) -> float:
    prob, label = prob.ravel(), label.ravel().astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if m.any():
            e += m.mean() * abs(prob[m].mean() - label[m].mean())
    return float(e)


def classifier_report(prob, label, thr=0.5):
    prob, label = prob.ravel(), label.ravel().astype(bool)
    pred = prob >= thr
    tp, fp, fn = (pred & label).sum(), (pred & ~label).sum(), (~pred & label).sum()
    return {"accuracy": float((pred == label).mean()),
            "precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1)),
            "auroc_pooled": pooled_auroc(prob, label), "ece": ece(prob, label)}


# ---------------------------------------------------------------- per-question breakdown
def breakdown(codes, cor, code_correct, det_correct: np.ndarray | None):
    Q, n = cor.shape
    c = cor.sum(1)
    always, never = c == n, c == 0
    mixed = ~always & ~never
    minority = np.zeros(Q, dtype=bool)
    top_wrong = np.zeros(Q)
    for i in range(Q):
        cnt = np.bincount(codes[i], minlength=len(code_correct[i]))
        wrong = cnt[~code_correct[i]]
        top_wrong[i] = wrong.max() if len(wrong) else 0
        minority[i] = c[i] > 0 and top_wrong[i] > c[i]
    out = {"always_correct": int(always.sum()), "never_correct": int(never.sum()), "mixed": int(mixed.sum()),
           "correct_minority": int(minority.sum()),
           "mean_correct_of_n": float(c.mean()), "mean_top_wrong_of_n": float(top_wrong.mean()),
           "top_wrong_beats_correct_rate": float((top_wrong > c).mean())}
    if det_correct is not None:
        out["recoverable"] = int((~det_correct & (c > 0)).sum())
    return out, {"mixed": mixed, "minority": minority}


def conversions(sel_one: np.ndarray, base: np.ndarray):
    return {"wrong_to_right": int((~base & sel_one).sum()), "right_to_wrong": int((base & ~sel_one).sum()),
            "net": int((~base & sel_one).sum() - (base & ~sel_one).sum())}


def answer_entropy(codes):
    ent = []
    for row in codes:
        p = np.bincount(row) / len(row)
        p = p[p > 0]
        ent.append(-(p * np.log(p)).sum())
    return float(np.mean(ent))


def thought_diversity(thoughts: np.ndarray, n: int) -> list[float]:
    """thoughts [Q*n, 6, D] sorted by (qid, sid). Mean pairwise cosine dissimilarity per step,
    via ||sum of unit vectors||^2 = n + sum_{i!=j} cos_ij."""
    T = thoughts.shape[1]
    out = np.zeros(T)
    Q = thoughts.shape[0] // n
    for start in range(0, Q, 64):
        x = np.asarray(thoughts[start * n:(start + 64) * n], dtype=np.float32)
        q = x.shape[0] // n
        x = x.reshape(q, n, T, -1)
        x = x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)
        s = x.sum(1)                                               # [q, T, D]
        mean_cos = ((s ** 2).sum(-1) - n) / (n * (n - 1))          # [q, T]
        out += (1 - mean_cos).sum(0)
    return (out / Q).tolist()
