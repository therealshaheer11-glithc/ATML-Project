"""Pre-registered analysis choices. Fixed BEFORE the first run (see DECISIONS.md).
Changing anything here after results exist must be logged in DECISIONS.md under 'Changes after data'."""

# ---- sampling grid
P_GRID = [0.05, 0.10, 0.20, 0.30]       # B2 dropout rates
SEEDS = [0, 1, 2]                        # sampling seeds (one checkpoint: no training-run variance)
N_POOL = 64                              # attempts per test question
N_LIST = [1, 2, 4, 8, 16, 32, 64]
TEMPS = [0.7, 1.0]                       # B3 text-CoT temperatures (seed 0 only)
TRAIN_SUBSET, N_TRAIN_POOL = 5000, 16    # R11 train-vs-test gap

# ---- default p rule
P_RULE = ("highest Pass@32 (mean over 3 seeds); if several p are within 1.0 point of the best Pass@32, "
          "take the one with the higher N=1 accuracy")
P_TIE_PASS32 = 0.01

# ---- selectors (the full, closed list; nothing else is tested in the baselines)
SCORES = ["conf", "prm_he_last", "prm_he_min", "prm_he_mean", "orm"]
BEST_OF = ["conf", "prm_he_last", "prm_he_min", "prm_he_mean", "orm"]   # pick highest-scoring attempt
VOTE = ["conf", "prm_he_last", "orm"]                                   # answer with largest summed score
# majority vote: random tie-break between tied answers; tie rate reported. 200 random subsets per N.

# ---- tests
MAIN_N = 32
PRIMARY = "best:prm_he_last"             # primary: PRM-HE (last step) best-of-N vs majority, N=32, default p
ANALYSIS_SEED = 12345
# Every other selector vs majority at N=32 = exploratory family, Holm-corrected.

# ---- gates
TARGET_COCONUT, TARGET_COT = 34.1, 42.9  # Hao et al.
GATE_TOL = 1.5                           # points
FP16_TOL = 0.3                           # points: fp16 vs fp32 deterministic Coconut
GATE2_MIN_GAP = 0.08                     # Pass@32 - majority@32 at the default p
# Gate 3: fails if any selector's efficiency 95% CI lies above 0.5 at N=32

# ---- R12 compute budget (reward models)
ANNOTATE_MAX_QUESTIONS = 100_000         # of 385,620 GSM8k-aug train questions (fixed shuffle, seed 1234)
ANNOTATE_SESSION2_MAX_H = 4.0            # extra annotation allowed in session 2 before training starts
RM_TRAIN_MAX_H = 6.5                     # per model; PRM on GPU0 and ORM on GPU1 in parallel
