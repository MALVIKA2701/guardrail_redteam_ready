# Adversarial Robustness Evaluation of LLM Guardrail / Moderation Models

A research framework for systematically evaluating the adversarial robustness
of safety guardrail / content-moderation classifiers — the layer that sits in
front of (or alongside) LLMs to catch disallowed content. Most published
jailbreak research attacks the *generator* (the LLM itself). This project
instead targets the *classifier* that is supposed to catch bad outputs, on
the hypothesis that these models are typically smaller, less robustly
trained, and far less studied than the LLMs they guard.

Built to demonstrate research maturity for a graduate application: a clear
threat model, multiple complementary attack families, cross-model and
black-box transfer studies, statistically rigorous evaluation (bootstrap CIs,
paired significance testing), ablations, a defense contribution, and
reproducible engineering (tests, CI, auto-generated reporting) — not just a
single attack script with one headline number.

See `ARCHITECTURE.md` for a full pipeline diagram and design rationale.

## Research Design

### Threat model
- **Gray/white-box surrogates** (full logit access, CPU-friendly, open
  weights): a registry of three architecturally distinct toxicity/moderation
  classifiers (`config.LOCAL_SURROGATE_REGISTRY`) — `unitary/toxic-bert`,
  `martin-ha/toxic-comment-model`, and
  `facebook/roberta-hate-speech-dynabench-r4-target`.
- **Black-box targets** (score/decision only, no gradients): OpenAI
  Moderation API and Perspective API.

### Attack families (three granularities + one "universal" variant)
| Attack | File | Granularity | Technique family |
|---|---|---|---|
| Character perturbation | `attacks/char_attacks.py` | character | homoglyphs, leetspeak, zero-width chars, typos + greedy search |
| Word substitution | `attacks/word_substitution_attack.py` | word | TextFooler-style importance-ranked synonym substitution |
| Paraphrase | `attacks/paraphrase_attack.py` | sentence | LLM-based meaning-preserving rewrites |
| Genetic search | `attacks/genetic_search_attack.py` | mixed (word+char) | population-based, score-guided black-box optimization; usable directly against black-box APIs, not just local surrogates |
| Universal trigger | `attacks/trigger_search_attack.py` | cross-example | black-box beam search for one reusable suffix that degrades many inputs at once, evaluated on a held-out split |

Testing three granularities plus a universal-trigger variant lets the
project answer a richer question than "can this be evaded?" — namely *at
what level of the input* guardrail models are most fragile, and whether
vulnerabilities are per-example or systemic.

### Transfer studies (two levels)
- **Cross-model transfer** (`multi_model_study.py`): attacks are optimized
  against one local surrogate, then replayed against the other surrogates
  with zero adaptation, producing a full transfer matrix. Tests whether
  guardrail architectures share a common blind spot.
- **Black-box API transfer** (`transfer_study.py`): the same surrogate-optimized
  attacks are replayed against production moderation APIs, testing whether
  the vulnerability generalizes beyond open-weight research models.
- **Direct black-box optimization** (`blackbox_direct_attack_study.py`): a
  score-guided genetic algorithm (`attacks/genetic_search_attack.py`) is run
  directly against the production APIs, querying their real continuous
  score every generation, rather than only replaying a surrogate-optimized
  attack. This is the stronger, more realistic black-box threat model, and
  its ASR is meant to be compared against `transfer_study.py`'s transfer
  rate for the same API to see how much attack strength surrogate-replay
  alone leaves on the table.

### Universal trigger generalization (important methodological point)
The universal trigger (`attacks/trigger_search_attack.py`) is optimized on
one batch of examples and evaluated on a **disjoint held-out batch it never
saw during search** (`universal_trigger_study.py`, controlled by
`config.TRIGGER_HOLDOUT_FRACTION`). The headline "universal trigger" claim
must be reported using the **held-out success rate**, not the optimize-set
score — optimizing and evaluating on the same examples would measure
overfitting to that batch, not genuine generalization. The study also
reports a `generalization_gap` (optimize-set success rate minus held-out
success rate) so overfitting is visible rather than hidden.

### Statistical rigor (`stats.py`)
- Bootstrap confidence intervals on Attack Success Rate (not bare point
  estimates).
- Paired McNemar significance tests between attack methods on shared
  examples, so claims like "word substitution significantly outperforms
  character substitution" are backed by a p-value.
- **Multiple-comparisons correction (Holm-Bonferroni)** applied across every
  pairwise test run: within `compare_attacks()` for a single classifier, and
  across the FULL family of (model × attack-pair) tests in
  `compare_attacks_across_models()` for the multi-model study. Running k
  pairwise comparisons uncorrected inflates the family-wise false-positive
  rate — e.g. 3 pairwise tests at alpha=0.05 uncorrected gives roughly a 14%
  chance of at least one spurious "significant" result even with no real
  difference anywhere. Every comparison reports both
  `significant_uncorrected` and `significant_after_correction` so the
  difference is visible.
- **Effect sizes alongside every corrected p-value.** A significant
  McNemar result only says two attacks differ; it says nothing about how
  much. Every pairwise comparison also reports an odds ratio (from McNemar's
  own 2x2 table, Haldane-Anscombe corrected) and a bootstrap confidence
  interval on the raw ASR difference (`paired_asr_difference_ci`), so a
  tiny, practically meaningless gap that happens to be "significant" is
  visibly tiny, not just flagged as significant.
- **Seed variance for stochastic search procedures.** The universal trigger
  search (`universal_trigger_study.py`) and the direct black-box genetic
  attack (`blackbox_direct_attack_study.py`) are both stochastic. Both now
  run multiple independent seeds and report mean/std/min/max (via
  `stats.seed_variance_summary`) instead of a single run's number — a
  single seed can't distinguish a genuinely strong/generalizing result from
  one lucky search.

### Ablations (`ablations.py`)
- ASR vs. query budget (does success rate saturate, and where?)
- ASR vs. minimum semantic-similarity threshold (what's the robustness/
  stealth trade-off?)

### Defense (`defense/adversarial_finetune.py`)
Fine-tunes a surrogate on a mix of clean + successfully-attacked examples
(adversarial training) and reports before/after ASR, so the project
contributes a mitigation alongside every attack, not attacks in isolation.
Critically, it also reports **clean-data accuracy, false-positive rate, and
false-negative rate on a fresh, disjoint, labeled evaluation set** before
and after fine-tuning — a defense that drops ASR by flagging everything is
not a real improvement, and this makes that failure mode visible rather
than hiding behind a single "ASR went down" headline.

### Perturbation cost (`metrics/perturbation_cost.py`)
Every attack result is paired with normalized character edit distance and
semantic similarity, so "success" that mangles text into gibberish is
distinguishable from success that preserves fluent, natural language.

### Reporting
- `visualize_results.py` — generates all plots (ASR bar chart with CI error
  bars, cross-model transfer heatmap, ablation curves, perturbation-cost
  histograms, defense before/after comparison) as PNGs.
- `generate_report.py` — aggregates every result file into a single
  `results/report.md` with embedded figures, ready to hand to a supervisor
  or paste into a portfolio writeup.

## Engineering quality

- **`tests/`** — 59 unit tests covering attack logic, metrics, and
  statistics, all running fully offline against `MockClassifier` and
  pure-Python fallbacks (no model downloads or network access required).
  Run with `pytest tests/ -v`.
- **`.github/workflows/ci.yml`** — runs the offline test suite automatically
  on every push/PR.
- **`ARCHITECTURE.md`** — pipeline diagram and design-decision rationale.

## Quickstart (fast run)

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build if you have a GPU
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
./run_fast.sh
```

Needs network access to huggingface.co on the first run (models + dataset
are cached afterwards). If the Jigsaw HF mirrors are unavailable, download
Kaggle's `train.csv` for the Toxic Comment Classification Challenge and run
with `JIGSAW_CSV=/path/to/train.csv ./run_fast.sh`.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

You will additionally need, depending on which components you use:
- Network access to **huggingface.co** (downloads the surrogate classifiers,
  the sentence-embedding model, and the toxicity datasets)
- An **`OPENAI_API_KEY`** env var for the OpenAI Moderation black-box target
- A **`PERSPECTIVE_API_KEY`** env var for the Perspective API black-box target

If you don't have API keys yet, leave `config.USE_API_TARGETS = False` — the
local-surrogate and cross-model-transfer results (the bulk of the project)
don't need them.

## Running the full pipeline

```bash
# 0. Sanity-check the pipeline works with zero dependencies
python run_experiment.py --mock --n_samples 8

# 1. Real attack evaluation against the primary surrogate
python run_experiment.py --dataset jigsaw_toxicity_pred --n_samples 200 --attacks char,word,genetic,paraphrase

# 2. Cross-model study (all three surrogates + transfer matrix + corrected stats)
python multi_model_study.py --n_samples 100 --attacks char,word,paraphrase

# 3. Universal trigger, with proper optimize/held-out split
python universal_trigger_study.py --n_samples 150

# 4. Black-box API transfer (surrogate-optimized attacks replayed on APIs)
python transfer_study.py --results results/attack_results.jsonl

# 5. Direct black-box optimization against the APIs themselves (stronger, costs real queries)
python blackbox_direct_attack_study.py --n_samples 20 --max_queries 60

# 6. Statistical analysis (bootstrap CIs, Holm-corrected McNemar tests)
python stats.py --results results/attack_results.jsonl

# 7. Ablations
python ablations.py --n_samples 60

# 8. Defense: adversarial fine-tuning + before/after ASR AND clean-data cost
python defense/adversarial_finetune.py --results results/attack_results.jsonl

# 9. Generate all figures + the consolidated report
python visualize_results.py
python generate_report.py
```

Open `results/report.md` for the final consolidated writeup with embedded
figures.

## Dataset

Uses public, already-labeled toxicity datasets — no novel harmful content is
authored as part of this project:
- Jigsaw Toxic Comment Classification (HF: `jigsaw_toxicity_pred`)
- HateXplain (HF: `hatexplain`)

`data/load_data.py` handles downloading and a small fixed local fallback
sample (clearly marked, non-toxic placeholder strings) for offline
smoke-testing the pipeline without network access.

## Verified

The full pipeline (all three attacks, stats, ablations, visualization, report
generation) has been run end-to-end offline using `MockClassifier` and the
rule-based paraphrase fallback, confirming every script executes without
errors and produces the expected output files. The attack search logic
itself was separately verified to work correctly (e.g. driving a flagged
score from 0.98 → 0.05 while preserving semantic similarity, under a
permissive similarity function).

**Known caveat in offline/fallback mode only:** without `sentence-transformers`
installed, `SemanticSimilarity` falls back to a crude word-level Jaccard
metric, which is too strict for character-level perturbations (changing one
character inside a word makes that whole word "not match"). This makes ASR
look artificially low (near 0%) only when running fully offline in `--mock`
mode. Once you install the real dependencies with network access, real
sentence embeddings are used and this resolves — real experiments/paper
results should always be run with the full dependencies installed.

## Responsible use / ethics note

- All attacks are evaluated against classifiers, not used to actually publish
  harmful content anywhere.
- No novel harmful text is authored as attack payloads beyond what already
  exists in public labeled datasets used for robustness research.
- The universal trigger search uses only semantically-neutral filler phrases
  (e.g. "by the way", "for context") as its candidate pool — no
  task-specific harmful content.
- Findings involving any third-party production API are intended for
  responsible disclosure to the provider, per standard adversarial ML
  research norms.
- The defense contribution (adversarial fine-tuning) is reported alongside
  every attack result so the project is not attack-only.

### Direct black-box attacks against production APIs (`blackbox_direct_attack_study.py`)

This is the one part of the project that queries a live, third-party
production system rather than an open-weight model you can run and inspect
locally. It's legitimate adversarial-robustness research, but it's the part
most worth being explicit about:

- **Capped query budget.** `config.MAX_TOTAL_API_QUERIES_PER_RUN` is a hard
  ceiling on total real API queries per script invocation, enforced by a
  `QueryBudgetGuard` regardless of what `--n_samples`, `--max_queries`, and
  `--n_seeds` are set to. This keeps usage well within free-tier / reasonable
  limits and prevents runaway query volume against a system you don't own.
- **No targeting of live user content.** Every input scored comes from
  existing public toxicity-benchmark datasets (Jigsaw, HateXplain), never
  from real users' live messages or content. There is no attempt, intent, or
  mechanism in this project to cause a moderation failure on anything other
  than research-dataset text scored for measurement purposes.
- **Coordinated disclosure.** Any specific successful adversarial example or
  discovered trigger string is intended to be responsibly disclosed to the
  relevant provider (OpenAI, Google/Jigsaw for Perspective) before any
  publication, consistent with standard adversarial-ML and security research
  norms — the goal is characterizing and fixing a weakness, not publishing a
  working exploit.
- **Explicit opt-in required.** The script refuses to run against real APIs
  unless invoked with `--i-accept-responsible-use`, which forces a
  conscious acknowledgment of the above before any live queries are sent.

## Suggested paper/portfolio structure mapped to this codebase

1. Intro/motivation — this README's opening paragraph
2. Threat model — "Threat model" section above, `ARCHITECTURE.md`
3. Methods — `attacks/*.py` docstrings, the attack-family table above
4. Experimental setup — `config.py`, `data/load_data.py`
5. Results — `results/report.md`, `results/figures/*.png`
6. Statistical validation — `stats.py` output
7. Ablations — `ablations.py` output
8. Defense — `defense/adversarial_finetune.py` before/after table
9. Ethics — "Responsible use / ethics note" section above
