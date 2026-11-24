# Coconut baselines: updated plan and fixed decisions

Group 11 (Aitasaam, Ahmad, Kashif). This document replaces the "what to run" parts of *Coconut Baselines:
What to Run First, and Why*. It records every choice we fixed **before the first run**. The code enforces
these choices (`cb/prereg.py`). If we change any of them after seeing results, we log the change in the last section.

---

## 1. What changed from the original baselines plan

| Item | Original plan | Now |
|---|---|---|
| Analysis choices | partly open | **fixed in advance** (§3): default-p rule, tie rule, closed selector list, one primary test, Holm correction for the rest |
| Seed variance | "spread across 3 seeds" | 3 seeds = **sampling variance only**. One checkpoint, no retraining; stated as a limitation (§6) |
| Selection efficiency | point estimate | **bootstrapped** 95% CI, computed on pooled means |
| Logging | answers, probabilities, thoughts | plus **per-shard RNG seed and batch layout** (exact replay) and the **deterministic path** for every question |
| R12 (reward models) | full Wang et al. recipe | **compute-capped** (§3.9) |
| B0 | none | **new**: hand-check 100 parsed answers before trusting any number |
| Checkpoints | "official Coconut code" | released checkpoints (§3.1); Wang et al. never released theirs |
| No-CoT anchor | trained GPT-2 No-CoT | **dropped**: no released checkpoint; the literature value (16.5%) is quoted only |
| Second checkpoint | none | You et al.'s released Coconut, used as a cheap **regime check** (no retraining needed) |

---

## 2. The baselines as they now stand

| ID | Run | Answers | Output |
|---|---|---|---|
| **B0** | Hand-check 50 parsed answers from the deterministic run and 50 from the p=0.1 pool; also count string vs numeric disagreements | Is answer parsing right? | `B0_parse_check.csv` |
| **B1** | Deterministic Coconut (fp16 and fp32), greedy text CoT, Coconut p=0.1 at N=1 (from the B2 pool) | Does our pipeline reproduce the published numbers? | Gate 1 |
| **B2** | Coconut + dropout, p ∈ {0.05, 0.10, 0.20, 0.30} × seeds {0,1,2} × 64 attempts per question, thoughts saved | How much headroom? Which p? | Pass@N, distinct answers, thought diversity, default p |
| **B3** | Text CoT sampled at T ∈ {0.7, 1.0}, 64 attempts, seed 0 | Does test-time scaling help text CoT as much? | Pass@N, majority, mean-log-prob pick, tokens per attempt |
| **B4** | random, majority vote, answer confidence (best and vote) on the B2 pools | Do the free selectors find correct attempts? | accuracy for N ∈ {1,…,64} |
| **B5** | PRM-HE and ORM (Wang et al. recipe, capped) as best-of-N and score-weighted vote | What can a learned selector do? | accuracy, classifier metrics, within-question AUROC |
| **B6** | always / never / mixed, recoverable, correct-minority, conversions vs the deterministic answer | Where exactly is the headroom? | breakdown table |
| **R11** | 5,000 train questions × 16 attempts at the default p | How much easier are training questions? | train vs test attempt accuracy |
| **Regime check** | You et al.'s checkpoint: deterministic + p ∈ {0.1, 0.2}, seed 0 | Is our checkpoint closer to Wang (Pass@4 ≈ 38) or You (≈ 47)? | Pass@N table |

---

## 3. Decisions fixed before the first run

### 3.1 Checkpoints
- **Coconut:** `UdeSA/coconut-0.2B-gsm8k-aug` (`coconut/`). Retrained with the official recipe (GPT-2, c=2, 3 latent
  stages, 6 thoughts); the card reports 34.57% on the GSM8K test set.
- **Text CoT:** the same repo's `cot/` model (the stage-0 warm start of that Coconut); the card reports 43.14%. Using the
  matched pair means "Coconut vs CoT" compares models from one training run.
- **Regime check:** `ModalityDance/latent-tts-coconut` (You et al.).
- Data: GSM8k-aug from the pinned Internalize-CoT commit, converted exactly as in the official repo (1,319 test questions).

### 3.2 Sampler (identical to Wang et al.'s released code)
- Prompt = question + "\n" (no special tokens). It's encoded once per question with **dropout off**, and its KV cache is shared by all 64 attempts.
- **Dropout on** for the six latent forward passes (inputs: `<start-latent>`, h1…h5; outputs: thoughts h1…h6), at
  every GPT-2 dropout site (embedding, attention probabilities, attention output, MLP output), all set to p.
- h6 is fed back with **dropout off**, then `<end-latent>` and **greedy** answer decoding (max 16 tokens) with dropout off.
- Text CoT: max 64 new tokens (official setting), temperature sampling; T=0 means greedy.
- Inference in **fp16**. Gate 1 checks it against an fp32 run.

### 3.3 Answer parsing and correctness
- Official rule: take the text after the last `#`, remove commas, strip spaces. Correct = **string equality** with the
  normalized gold answer. A numeric comparison is logged alongside, and every disagreement is reported.
- Majority vote groups attempts by this parsed string. An empty parse counts as its own (wrong) answer.

### 3.4 Pools ("sample once, analyze many times")
- Every attempt is saved: question id, attempt id, shard, row, raw text, parsed answer, correct, finished flag, number
  of answer tokens, sum log-prob, **confidence = mean max-probability over non-EOS answer tokens (Wang's definition)**,
  answer token ids, and the six thoughts (final layer, fp16).
- **Replay:** each shard draws its dropout masks from a generator seeded by a hash of (pool name, shard), and its exact
  batch layout is stored in `spec.json`. `replay_shard` regenerates it, optionally with an intervention on any thought.
- The deterministic run saves every question's **deterministic path** (thoughts + answer).

### 3.5 Default p (decided by rule, written once to `decisions/default_p.json`)
> Highest Pass@32 averaged over the 3 seeds. If several p values are within 1.0 point of the best, take the one with
> the higher N=1 accuracy.

The rule uses the oracle Pass@32, which treats every selector the same, so it can't favour any selector.

### 3.6 Selectors: the closed list
Nothing outside this list is tested in the baselines.
- **random pick**, **majority vote**;
- **best-of-N** (highest-scoring attempt) for: confidence, PRM-HE last step, PRM-HE min, PRM-HE mean, ORM;
- **score-weighted vote** (answer with the largest summed score) for: confidence, PRM-HE last, ORM.

Rules:
- **Ties in majority vote:** uniformly random among the tied answers. The tie rate is reported for every N.
- **Smaller N:** Pass@N uses the unbiased estimator. Selectors use **200 random subsets** of size N per question.
- N ∈ {1, 2, 4, 8, 16, 32, 64}.
- The Stage-2 weighted-voting variants with knobs (Borda power, softmax temperature, top-k) need a calibration split.
  They are **not** part of the baselines.

### 3.7 Main comparison and tests
- **Primary (the one confirmatory test):** PRM-HE (last step) best-of-N **vs** majority vote, **N = 32**, default p.
- Per-question accuracy is first averaged over the 3 seeds. The test is a **question-level paired bootstrap** (2,000 resamples), reported as mean
  difference, 95% CI and p-value. Exact **McNemar** on one fixed subset draw of seed 0 is reported as a secondary check.
- **Everything else** vs majority at N=32 is **exploratory**, with p-values **Holm-corrected** across that family.
- All accuracies carry question-level bootstrap 95% CIs. With 1,319 questions, a difference under about 2 points is treated as noise
  unless the paired CI excludes 0.

### 3.8 Statistics details
- **Selection efficiency** = (selector − majority) / (Pass@N − majority), computed on pooled means and bootstrapped. It isn't
  averaged per question, because the ratio is unstable when the gap is small.
- **Within-question AUROC** (mixed questions only, averaged) is the headline measure of whether a score separates attempts.
  Pooled AUROC and ECE are shown only for contrast.
- **Thought diversity** = mean pairwise cosine dissimilarity between attempts' thoughts, per step (seed 0).

### 3.9 R12 compute budget (reward models)
| Step | Cap | Notes |
|---|---|---|
| Annotation | ≤ **100,000** of 385,620 train questions (fixed shuffle, seed 1234), plus all 500 validation questions | 5 attempts per question, deduplicated by answer, 10 continuations per prefix, at the default p. Whatever finishes in session 1, plus ≤ 4 h in session 2 |
| PRM training | ≤ **6.5 h** on one T4 | ≤ 10 epochs, batch 128, lr 1e-4, warmup 500, AdamW wd 0.01, fp16; the LR schedule is fitted to the time cap |
| ORM training | ≤ **6.5 h** on the other T4, in parallel | correct/incorrect balanced 1:1 |
| Scoring | default-p pools, 3 seeds | minutes |

`runs/bench.json` holds the measured throughput, from which the code projects hours for every run on the actual GPU. The number of
questions actually annotated and the epochs actually trained are reported with the B5 results.

### 3.10 Gates
1. **Gate 1 (anchors):** deterministic Coconut within 34.1 ± 1.5, greedy CoT within 42.9 ± 1.5, fp16 vs fp32 within
   0.3 points, and B0 hand-check ≥ 98/100 correct. **If it fails, stop and debug.**
2. **Gate 2 (headroom):** at the default p, Pass@32 − majority@32 ≥ 8 points. If not, the premise is weak, so we revisit the plan.
3. **Gate 3 (selection gap):** if any selector's efficiency CI at N=32 lies above 0.5 (it closes most of the gap),
   our method has little left to add, so we revisit the plan.
4. **Regime:** report our Pass@4 next to Wang's (≈ 38) and You's (≈ 47).

---

## 4. Session plan (Kaggle, 2 × T4, 12 h each)

| Session | Contents (in order) | Must finish before moving on |
|---|---|---|
| 1 | setup, benchmark → B1 anchors → B2 (12 pools) → B3 → regime check → **default p** → R11 → annotation (until time runs out) | B1 + B2 |
| 2 | finish annotation (≤ 4 h) → PRM ∥ ORM training (≤ 6.5 h each) → scoring → final report | session-1 output attached |

Each session uses both GPUs: the shards of every pool are split between two worker processes.
Everything is time-capped and resumable at the shard level. Rerunning a session continues where it stopped.
Expected disk use: about 9–10 GB of thoughts in session 1 (12 B2 pools), well under Kaggle's 20 GB output limit.

---

## 5. Deviations from the papers (deliberate)
- **Checkpoint:** a public retrain of Coconut, not Wang et al.'s (unreleased). Our absolute numbers will differ from theirs.
  Gate 1 and the regime check show by how much.
- **PRM loss:** each annotated path trains all six step heads in one forward pass, with class-balanced weights. Wang instead
  samples single (path, step) items balanced 1:1. Same objective, about six times cheaper.
- **Annotation scale:** at most 100k questions, not 385k. Training is time-capped (§3.9).
- **Annotation, step 6:** with all six thoughts fixed nothing is random, so we decode once instead of ten identical times.
  Like Wang's code, the fixed prefix is re-encoded with dropout off.
- **No-CoT anchor:** dropped (no checkpoint).

## 6. Limitations we state up front
- **One Coconut checkpoint, no retraining.** The ± across 3 seeds is sampling variance for that one model. It says nothing
  about variance across training runs (Hao et al.'s ±1.5 is the latter). The regime check on You et al.'s checkpoint is a
  partial, descriptive look at a second model.
- Reward models are trained at reduced scale (§3.9). A weak B5 result is therefore an upper bound on how badly learned selectors
  do, not a definitive one.
- The PRM and ORM are trained on the training set, which Coconut was trained on. R11 measures how much easier those questions are.
- The code passed CPU tests against Hugging Face GPT-2 and against an official-style Coconut loop, plus a full CPU dry run of both
  sessions. **Its first real GPU run is session 1.** Gate 1 exists to catch anything the CPU tests can't.

## 7. Geometry baselines G1–G4 (added 2026-11-24, before any geometry result was seen)
No training and no new model runs: computed on CPU from the thoughts saved in session 1 (default p, 3 seeds).
- **Scores (8):** total distance, velocity entropy, net displacement, alignment (proposal; Li et al., Vilas et al.),
  and compactness, straightness, local smoothness, curvature (Wang et al.; our code matches their `geometry.py` exactly).
- **Spaces (9):** full 768 dims; PCA and random subspaces at ranks r ∈ {32, 64, 128, 256}. A subspace is applied per
  thought position as hₜ → BₜBₜᵀhₜ in 768 coordinates (the same form Bₜ will take). PCA is fit per position on 40k
  train-question thoughts (no test data, no labels); random bases are QR of Gaussian noise (seed 2024).
- **Direction of "good"** fixed from the papers: lower total distance and velocity entropy, higher net displacement and
  alignment (Vilas, Li); higher for all four Wang metrics (the direction their geometric loss pushes correct paths).
- **Reported:** within-question AUROC (headline), pooled AUROC, correlation with confidence, best-of-N accuracy for
  N = 1…64, and gain over majority at N = 32 with paired bootstrap; Holm correction across all 72 configurations.
  References: majority, confidence, PRM-HE, ORM, Pass@N. All exploratory; §3.7's primary test is unchanged.
- **Bₜ** (gradient subspace) and the causal check come later, plugged into the same code.

## 8. Changes after data
*(none yet; log date, what changed, why, and which results were already seen)*
