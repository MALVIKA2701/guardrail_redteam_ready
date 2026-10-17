# Extension: Can a guardrail tell when it is being fooled?

*Added by Samiksha Prasad as an extension of the original study by Malvika
Rajeev Kappadan and Samiksha Prasad. The original attack, transfer, statistics
and defense code is unchanged; this extension lives in `deferral/`,
`deferral_study.py` and `tests/test_deferral.py`.*

## The question

The main study shows that moderation classifiers can be evaded: small
character, word or paraphrase changes push a toxic text's score below the
0.5 block threshold, and it is let through.

Real moderation systems are rarely purely binary. Uncertain cases are often
sent to human review. That raises a question the binary evaluation cannot
answer:

> If the guardrail is allowed to say "I'm not sure, a human should check
> this", how much of the evasion problem goes away, how much human review
> does that cost, and does it still hold when the attacker knows about it?

## Method

**Three-way policy.** `deferral/selective_guardrail.py`:

| Condition | Decision |
|---|---|
| score ≥ 0.5 | block |
| deferral score ≥ lower edge | defer to human review |
| otherwise | allow |

Two deferral signals are supported:

- `single`: the primary classifier's own score, i.e. a confidence band just
  below the threshold.
- `ensemble`: the maximum score across the three registry classifiers, i.e.
  "defer if any model is suspicious". This reuses the project's finding that
  attacks only partly transfer between architectures.

**Calibrating the band to a review budget.** The operator chooses a budget B,
the largest share of benign traffic they are willing to send to human
review (for example 5%, 10% or 20%). The lower edge is chosen on a
calibration split of benign texts using a split-conformal quantile, so that
on new benign data the expected review rate is at most B. The realized rate
is then measured on a disjoint test split.

**The adaptive attacker.** The word-substitution and genetic attacks in this
repo stop as soon as the score drops below 0.5, so their successful examples
sit just under the threshold by construction. A band below 0.5 would catch
many of them for that reason alone, which would overstate the band's value.

The study therefore also runs an adaptive attacker that knows the band and
keeps going until the deferral score is below the lower edge. It is built
without editing any attack code: the attacks receive a monotone re-scaling
of the score that maps the lower edge to 0.5, so their own stopping rule now
fires at the edge.

**What is reported, per budget and attack** (`results/deferral_results_<signal>.json`):

- Lower edge, and realized review rate on test benign text with a bootstrap CI
- Attack success rate (ASR) with no band (the original binary result)
- ASR of the standard attack under the band (optimistic)
- ASR of the adaptive attack under the band, with a bootstrap CI (the honest number)
- Paired difference from the no-band ASR with its CI, McNemar's test, and
  Holm-Bonferroni correction across all budget × attack comparisons
- Extra queries and semantic similarity the adaptive attacker needs

## Running it (free, no API keys)

All three classifiers are open-weight and run on a CPU, so this needs no
paid API. A free Google Colab notebook is enough.

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# offline smoke test
python deferral_study.py --mock --n_samples 8

# main run: single-model confidence band
python deferral_study.py --n_samples 400 --attacks char,word,genetic --budgets 0,0.05,0.1,0.2

# ensemble signal
python deferral_study.py --n_samples 400 --attacks char,word,genetic --signal ensemble
```

`--budgets 0` is included on purpose: a budget of 0 sets the edge to 0.5 and
reproduces the original binary guardrail, so every table carries its own
baseline.

## How to read the results

- If adaptive ASR stays well below the no-band ASR at a small budget, a
  modest amount of human review buys real robustness, even against an
  attacker who knows the policy.
- If adaptive ASR climbs back to the no-band level, the band only helped
  against attacks that stopped early. That is still a useful, publishable
  negative result: it shows that evaluating a deferral defense against
  non-adaptive attacks overstates it.
- Comparing `single` with `ensemble` shows whether disagreement between
  models is a better "I'm not sure" signal than one model's own score.
- Higher adaptive query counts and lower similarity mean the band makes
  attacks more expensive or more visible, even where they still succeed.

## Limitations

- The conformal guarantee is on the benign review rate, and only holds if
  real benign traffic resembles the calibration split.
- The adaptive attacker reuses this repo's search procedures. A stronger
  optimizer could do better, so adaptive ASR is a lower bound on what a
  determined attacker achieves.
- In ensemble mode each attack query scores all three models, so query
  counts are not directly comparable with single-model runs.
