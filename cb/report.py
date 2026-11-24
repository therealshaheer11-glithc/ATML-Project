"""Turn results.json into a readable report.md."""
from __future__ import annotations

from . import prereg as R


def _pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


def _ci(t):
    return f"{100 * t[0]:.1f} [{100 * t[1]:.1f}, {100 * t[2]:.1f}]"


def write_report(res, path):
    L = ["# Coconut baselines — results", ""]
    a = res.get("anchors", {})
    L += ["## B0/B1 anchors (Gate 1)", "", "| run | acc % [95% CI] | empty parse | unfinished | string≠numeric |",
          "|---|---|---|---|---|"]
    for k, v in a.items():
        if isinstance(v, dict) and "acc" in v:
            L.append(f"| {k} | {_ci(v['acc'])} | {_pct(v['parse_empty'])} | {_pct(v['unfinished'])} | "
                     f"{v['string_vs_numeric_disagree']} |")
    L += ["", f"Gate 1: `{a.get('gate1')}`  fp16 vs fp32: `{a.get('fp16_vs_fp32')}`",
          "Hand-check `B0_parse_check.csv` (fill the `hand_check_ok` column).", ""]
    b2 = res.get("b2", {})
    if b2:
        L += ["## B2 headroom (mean ± sd over seeds)", "",
              "| p | N=1 | " + " | ".join(f"Pass@{N}" for N in R.N_LIST[1:]) + " | distinct/64 | top-wrong>correct |",
              "|---" * (len(R.N_LIST) + 3) + "|"]
        for p, v in b2.items():
            cells = [f"{100 * v['acc_n1']['mean']:.1f}±{100 * v['acc_n1']['sd']:.1f}"]
            cells += [f"{100 * v[f'pass@{N}']['mean']:.1f}±{100 * v[f'pass@{N}']['sd']:.1f}" for N in R.N_LIST[1:]]
            cells += [f"{v['distinct_of_64']['mean']:.1f}", _pct(v['top_wrong_beats_correct']['mean'])]
            L.append(f"| {p} | " + " | ".join(cells) + " |")
        L += ["", f"Default p: **{res.get('default_p')}**  ({R.P_RULE})", ""]
    sel = res.get("selection", {})
    if sel:
        L += ["## B4/B5 selectors at the default p (accuracy %, seeds averaged per question)", ""]
        names = [k for k, v in sel["byN"][str(R.N_LIST[0])].items() if isinstance(v, dict) and "acc" in v]
        L += ["| N | Pass@N | " + " | ".join(names) + " | tie rate |", "|---" * (len(names) + 3) + "|"]
        for N, row in sel["byN"].items():
            L.append(f"| {N} | {100 * row['pass'][0]:.1f} | " +
                     " | ".join(f"{100 * row[k]['acc'][0]:.1f}" for k in names) +
                     f" | {_pct(row['majority_tie_rate'])} |")
        L += ["", f"**Primary**: {sel['primary']['comparison']}: `{sel['primary']['result']}`",
              f"Exploratory (Holm-adjusted p): `{sel['exploratory_holm']}`", "",
              "## B6 breakdown (seed 0, 64 attempts)", "", f"`{sel.get('breakdown_seed0')}`", "",
              f"Conversions vs deterministic Coconut (N=64): `{sel.get('conversions_vs_deterministic_seed0_N64')}`",
              "", "## Score quality (within-question AUROC is the honest one)", "",
              "| score | within-question AUROC | pooled AUROC | ECE |", "|---|---|---|---|"]
        for k, v in sel.get("score_quality_seed0", {}).items():
            L.append(f"| {k} | {v['within_question']:.3f} (n={v['n_mixed']}) | {v['pooled']:.3f} | "
                     f"{v.get('ece', float('nan')):.3f} |")
    if res.get("gates"):
        g = res["gates"]
        L += ["", "## Gates 2 and 3", "", f"Gate 2 (headroom ≥ 8 pts): **{g['gate2_pass']}** "
              f"({g['gate2_headroom_points']:.1f} pts)  ·  Gate 3 (no selector closes most of the gap): **{g['gate3_pass']}**"]
    if res.get("b3"):
        L += ["", "## B3 text CoT sampling", "", f"`{res['b3']}`"]
    if res.get("r11"):
        L += ["", "## R11 train vs test attempt accuracy", "", f"`{res['r11']}`"]
    if res.get("b2_you"):
        L += ["", "## Regime check on You et al.'s checkpoint", "", f"`{res['b2_you']}`"]
    open(path, "w").write("\n".join(L) + "\n")
