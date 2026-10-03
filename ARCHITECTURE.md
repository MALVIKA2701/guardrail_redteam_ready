# Architecture

```
                        ┌─────────────────────┐
                        │  data/load_data.py   │   Jigsaw / HateXplain
                        │  (public datasets)   │   (+ offline fallback)
                        └──────────┬───────────┘
                                   │  seed texts (flagged as toxic)
                                   ▼
        ┌──────────────────────────────────────────────────┐
        │                    ATTACKS                        │
        │  attacks/char_attacks.py         (char-level)     │
        │  attacks/word_substitution_attack.py (word-level) │
        │  attacks/paraphrase_attack.py    (sentence-level) │
        │  attacks/trigger_search_attack.py (universal)     │
        └───────────────────────┬────────────────────────────┘
                                 │ perturbed candidates
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │                    TARGETS                        │
        │  targets/local_classifier.py                       │
        │    - toxic_bert, toxic_comment_distilbert,          │
        │      roberta_hate_dynabench (registry)              │
        │  targets/api_moderation.py                          │
        │    - OpenAI Moderation API (black-box)               │
        │    - Perspective API (black-box)                     │
        └───────────────────────┬────────────────────────────┘
                                 │ scores / flagged decisions
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │                  EVALUATION                        │
        │  evaluate.py           -> per-example ASR records  │
        │  multi_model_study.py  -> cross-model transfer      │
        │  transfer_study.py     -> black-box API transfer    │
        │  ablations.py          -> query budget / similarity │
        │  stats.py              -> bootstrap CI, McNemar     │
        └───────────────────────┬────────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │                    DEFENSE                         │
        │  defense/adversarial_finetune.py                    │
        │    fine-tune surrogate on adversarial examples,      │
        │    re-run evaluate.py, report before/after ASR       │
        └───────────────────────┬────────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │                   REPORTING                        │
        │  visualize_results.py -> results/figures/*.png      │
        │  generate_report.py   -> results/report.md          │
        └──────────────────────────────────────────────────┘
```

## Design principles

1. **Every component is independently runnable and independently testable.**
   `tests/` covers attack logic, metrics, and statistics using
   `MockClassifier` and pure-Python fallbacks -- no network access or heavy
   ML dependencies required to verify the *algorithmic* correctness of the
   project (see `.github/workflows/ci.yml`).

2. **Three attack granularities, one interface.** `char_attacks.py`,
   `word_substitution_attack.py`, and `paraphrase_attack.py` all expose
   `score_fn`/`similarity_fn`-based functions returning the same result
   schema (`best_text`, `best_score`, `orig_score`, `queries_used`,
   `success`, `similarity`), so `evaluate.py` can run and compare them
   uniformly, and `stats.py` can run paired significance tests between any
   two of them.

3. **Two distinct threat models are tested, not conflated.** Per-example
   attacks (character/word/paraphrase) answer "how fragile is this specific
   input's classification?" The universal trigger search
   (`trigger_search_attack.py`) answers a different, arguably more
   consequential question: "is there a single reusable string that degrades
   the classifier broadly?"

4. **Transfer is tested at two levels.** `multi_model_study.py` measures
   surrogate-to-surrogate transfer (does an attack crafted against one open
   model fool a different open model?), and `transfer_study.py` measures
   surrogate-to-black-box-API transfer (does it fool a production system?).
   Reporting both separately lets you distinguish "guardrail models share a
   architecture-level weakness" from "even production-hardened APIs are
   vulnerable."

5. **No result is reported without uncertainty.** `stats.py` wraps every ASR
   with a bootstrap confidence interval and provides paired McNemar
   significance tests between attack methods, so claims like "word
   substitution beats character substitution" are backed by a p-value, not
   just two point estimates that happen to differ.

6. **The defense is not an afterthought.** `defense/adversarial_finetune.py`
   is a first-class script producing its own before/after comparison, so the
   project's contribution includes a mitigation, not just an attack.
