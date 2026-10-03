"""
Central configuration for the guardrail adversarial robustness project.

Edit this file before running experiments. Nothing here requires code changes
elsewhere — run_experiment.py, transfer_study.py, and defense/*.py all read
from this module.
"""

import os

# ---------------------------------------------------------------------------
# Local surrogate classifiers (white/gray-box targets)
# ---------------------------------------------------------------------------
# Primary surrogate (used when a single model is requested, e.g. by
# defense/adversarial_finetune.py which fine-tunes one model at a time).
LOCAL_SURROGATE_MODEL = "unitary/toxic-bert"

# Full registry of local surrogates used for the multi-model study
# (broader coverage + surrogate-to-surrogate transfer matrix). All are
# small, CPU-friendly, open-weight moderation/toxicity classifiers spanning
# different architectures and training data, so cross-model transfer results
# are meaningful rather than just measuring near-duplicate models.
LOCAL_SURROGATE_REGISTRY = {
    "toxic_bert": "unitary/toxic-bert",
    "toxic_comment_distilbert": "martin-ha/toxic-comment-model",
    "roberta_hate_dynabench": "facebook/roberta-hate-speech-dynabench-r4-target",
}

LOCAL_SURROGATE_THRESHOLD = 0.5                # score >= threshold -> "flagged"

# Sentence-embedding model used for the perturbation-cost (semantic
# similarity) metric. Also CPU-friendly.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Black-box API targets
# ---------------------------------------------------------------------------
USE_API_TARGETS = False  # flip to True once you have API keys set as env vars

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PERSPECTIVE_API_KEY = os.environ.get("PERSPECTIVE_API_KEY")

ENABLED_API_TARGETS = {
    "openai_moderation": bool(OPENAI_API_KEY),
    "perspective": bool(PERSPECTIVE_API_KEY),
}

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "jigsaw_toxicity_pred"   # or "hatexplain"
N_SAMPLES = 200                         # subset size for a manageable run
RANDOM_SEED = 42

# Fraction of each sample drawn from toxic examples (stratified sampling).
# Jigsaw is only ~10% toxic, so without this a 200-example sample leaves
# ~20 examples to attack. Set to None for a plain random sample.
TOXIC_FRACTION = 0.5

# The original `jigsaw_toxicity_pred` hub entry needs a manual Kaggle
# download and is a legacy loading script (unsupported in datasets>=4).
# These parquet mirrors are tried in order instead; text/label columns are
# auto-detected. Alternatively set JIGSAW_CSV=/path/to/kaggle/train.csv.
JIGSAW_HF_MIRRORS = [
    "anitamaxvim/jigsaw-toxic-comments",
    "tcapelle/jigsaw-toxic-comment-classification-challenge",
]
JIGSAW_LOCAL_CSV = None

# Only --mock runs may silently fall back to the 8-sentence offline sample.
ALLOW_OFFLINE_FALLBACK = False

# ---------------------------------------------------------------------------
# Attack search budget
# ---------------------------------------------------------------------------
MAX_QUERIES_PER_EXAMPLE = 40   # query budget for the greedy char-attack search
MAX_PARAPHRASE_CANDIDATES = 5  # candidates generated per example
MIN_SEMANTIC_SIMILARITY = 0.75  # reject perturbations below this cosine sim
TRIGGER_HOLDOUT_FRACTION = 0.3  # fraction of flagged examples reserved as held-out for the universal trigger study

# Hard ceiling on total real API queries in a single invocation of
# blackbox_direct_attack_study.py, across ALL examples, seeds, and targets
# combined -- a responsible-use safeguard independent of what --n_samples /
# --max_queries / --n_seeds happen to multiply out to. Chosen to comfortably
# fit within free-tier limits for both OpenAI Moderation and Perspective API.
MAX_TOTAL_API_QUERIES_PER_RUN = 500

# ---------------------------------------------------------------------------
# Ablation sweeps (used by ablations.py)
# ---------------------------------------------------------------------------
QUERY_BUDGET_SWEEP = [5, 10, 20, 40, 80]
SIMILARITY_THRESHOLD_SWEEP = [0.6, 0.7, 0.75, 0.8, 0.9]
PERTURBATION_RATE_SWEEP = [0.05, 0.1, 0.2, 0.3, 0.4]

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_CI = 0.95
SIGNIFICANCE_ALPHA = 0.05

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
ATTACK_RESULTS_PATH = os.path.join(RESULTS_DIR, "attack_results.jsonl")
MULTI_MODEL_RESULTS_PATH = os.path.join(RESULTS_DIR, "multi_model_results.jsonl")
TRANSFER_RESULTS_PATH = os.path.join(RESULTS_DIR, "transfer_results.json")
SURROGATE_TRANSFER_MATRIX_PATH = os.path.join(RESULTS_DIR, "surrogate_transfer_matrix.json")
MULTI_MODEL_STATS_PATH = os.path.join(RESULTS_DIR, "multi_model_stats.json")
DEFENSE_RESULTS_PATH = os.path.join(RESULTS_DIR, "defense_results.json")
ABLATION_RESULTS_PATH = os.path.join(RESULTS_DIR, "ablation_results.json")
STATS_RESULTS_PATH = os.path.join(RESULTS_DIR, "stats_results.json")
TRIGGER_STUDY_RESULTS_PATH = os.path.join(RESULTS_DIR, "trigger_study_results.json")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
REPORT_PATH = os.path.join(RESULTS_DIR, "report.md")
