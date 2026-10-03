# Changes (fixes applied before first real run)

1. **Dataset loading (blocker).** `jigsaw_toxicity_pred` on the HF Hub is a
   legacy loading script that needs a manual Kaggle download and is
   unsupported in `datasets>=4`. It always failed and the code *silently*
   fell back to 8 placeholder sentences, so "real" runs would have reported
   results on 8 made-up examples. Now: loads from parquet mirrors (or a local
   Kaggle CSV via `JIGSAW_CSV`), and fails loudly unless `--mock` is used.
   HateXplain now loads from its auto-converted parquet branch.
2. **Stratified sampling.** Jigsaw is ~10% toxic, so `--n_samples 200` gave
   ~20 attackable examples. `config.TOXIC_FRACTION = 0.5` fixes that.
3. **Scoring bug for 2 of 3 surrogates.** `LocalSurrogateClassifier.score`
   took the max over *all* labels, including `non-toxic` / `nothate`. For
   2-class models that's always >= 0.5, so every input was "flagged" and ASR
   was stuck at 0% for `martin-ha/toxic-comment-model` and
   `facebook/roberta-hate-speech-dynabench-r4-target`. Benign labels are now
   excluded. Also: score caching (faster) and automatic GPU use.
4. **ASR definition.** Attacks now run only on true positives (labeled toxic
   AND flagged), the standard definition.
5. **Defense.** (a) toxic-bert's multi-label head + integer labels crashed the
   Trainer's BCE loss -> now a proper binary single-label head with named
   labels; (b) training data had no benign examples, teaching "flag
   everything" -> benign negatives added 1:1, duplicates removed;
   (c) after-defense ASR was measured on the training examples (leakage) ->
   now fresh, disjoint examples, with the original model attacked on the same
   examples for an apples-to-apples before/after.
6. `USE_API_TARGETS` is now actually respected; `accelerate` added to
   requirements (needed by the HF Trainer).
7. `run_fast.sh` + README quickstart; 7 new offline tests (59 total, all pass).
