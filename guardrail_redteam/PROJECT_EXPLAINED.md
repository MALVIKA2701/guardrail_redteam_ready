# Project Explained — Reference Doc

*Adversarial Robustness Evaluation of LLM Guardrail / Moderation Models*

Use this as a pre-interview refresher. It's organized as: one-paragraph pitch
→ pipeline walkthrough → likely questions with answers → things to admit
honestly if pushed.

---

## The one-paragraph pitch

LLM safety systems typically have two layers: the LLM itself, and a smaller,
separate **guardrail/moderation classifier** that checks inputs or outputs
for disallowed content. Almost all published jailbreak research attacks
layer one (the LLM). This project asks whether layer two — the classifier
— is actually robust, on the hypothesis that it's smaller, trained on
narrower data, and far less adversarially studied than the LLM it protects.
The project systematically attacks these classifiers using established
adversarial-NLP techniques across three input granularities plus a
universal-trigger variant, measures whether the vulnerability transfers
across different classifier architectures and into production black-box
APIs, quantifies results with proper statistics (not bare percentages), and
closes the loop with a defense (adversarial fine-tuning) evaluated with the
same rigor as the attack.

---

## Pipeline walkthrough (what happens, in order, and why)

1. **Data** — public labeled toxicity datasets (Jigsaw, HateXplain). No
   novel harmful text authored.
2. **Targets** — 3 open-weight local classifiers (white/gray-box, full
   score access) + 2 production APIs (OpenAI Moderation, Perspective;
   black-box, decision/score only).
3. **Attacks** — four families:
   - *Character-level*: homoglyphs, leetspeak, zero-width chars, typos +
     greedy search.
   - *Word-level*: TextFooler-style — rank words by importance (via
     leave-one-out deletion), substitute the most important ones with
     synonyms.
   - *Sentence-level*: paraphrase the whole input, keep whichever candidate
     most reduces the score.
   - *Universal trigger*: black-box beam search for ONE reusable suffix that
     reduces the average score across MANY examples at once — a
     fundamentally different, more consequential threat model than
     per-example attacks.
4. **Perturbation cost** — every result reports edit distance + semantic
   similarity alongside success, so "success via gibberish" is
   distinguishable from "success while staying fluent."
5. **Transfer studies** — three angles:
   - Cross-model: attack model A, replay on models B/C untouched, with
     Holm-Bonferroni-corrected significance tests across the full family of
     (model × attack-pair) comparisons.
   - Cross-API (replay): attack the cheap local surrogate, replay on
     production black-box APIs.
   - Cross-API (direct): run a score-guided genetic algorithm directly
     against the production APIs' real continuous scores, to measure the
     stronger threat model of a determined black-box adversary rather than
     only surrogate-transfer.
6. **Statistics** — bootstrap confidence intervals on ASR; paired McNemar
   tests between attack methods (paired because attacks run on the *same*
   examples); Holm-Bonferroni correction across every family of pairwise
   tests, since uncorrected multiple comparisons inflate the false-positive
   rate.
7. **Ablations** — ASR vs. query budget; ASR vs. similarity threshold.
8. **Universal trigger generalization** — the trigger is optimized on one
   batch and evaluated on a disjoint held-out batch it never saw, with the
   generalization gap reported explicitly, so "universal" means "transfers
   to unseen examples," not "fit this specific batch."
9. **Defense** — adversarial fine-tuning on clean + attacked examples;
   before/after ASR comparison AND before/after clean-data accuracy /
   false-positive rate, so a defense that "works" by flagging everything is
   visible as a failure, not hidden behind a lower ASR number.
10. **Engineering** — 39 offline unit tests (MockClassifier, no GPU/API
    needed), CI on every push, auto-generated report + figures.

---

## Likely questions and answers

**Q: Why attack the classifier instead of the LLM?**
A: It's the understudied half of the safety stack. LLM jailbreaks get heavy
research attention; the smaller classifier in front of (or behind) the LLM
is often the actual last line of defense in production, yet gets almost no
adversarial scrutiny.

**Q: Isn't this just red-teaming? What's the actual research contribution?**
A: Three things beyond a single attack script: (1) systematic comparison
across attack granularities rather than one ad hoc trick, (2) a two-level
transfer study (cross-model AND cross-black-box-API) testing whether the
weakness is architectural or a one-off, (3) a defense evaluated with the
same statistical rigor as the attack, so the project isn't attack-only.

**Q: Why not use gradient-based attacks (e.g. GCG-style)?**
A: Those require white-box gradient access to the target and realistically
need GPU compute. Real attackers facing production APIs only get
query/decision access — so black-box, query-based search is both the more
realistic threat model and matches CPU-only constraints.

**Q: How do you know a "successful" attack isn't just garbled nonsense?**
A: Every result is paired with normalized edit distance and semantic
similarity (via sentence embeddings). Success with high edit distance and
low similarity is flagged as a weak/uninteresting result; success with high
similarity and low edit distance is the one that actually threatens real
moderation.

**Q: Why McNemar's test specifically, not just comparing two ASR
percentages?**
A: The attacks are run on the *same* set of examples, so the two success
sequences are paired, not independent samples. McNemar's test is the
correct paired test for this; comparing raw percentages (or using an
unpaired test) would ignore the pairing and could give a misleading
significance conclusion.

**Q: You're running a lot of pairwise significance tests — doesn't that
inflate your false-positive rate?**
A: Yes, and the project corrects for it. Every family of pairwise McNemar
tests — both within a single classifier's attack comparisons and across the
full (model × attack-pair) grid in the multi-model study — goes through
Holm-Bonferroni correction. Holm-Bonferroni was chosen over plain Bonferroni
because it's more powerful (less conservative) while still controlling the
family-wise error rate exactly, and over Benjamini-Hochberg because I want
to avoid any false positive across the family, not just control the
expected proportion of them. Every reported comparison shows both the
uncorrected and corrected significance flag so the difference is visible.

**Q: Is your "universal" trigger actually universal, or did it just overfit
to the examples you searched over?**
A: This is checked directly: the trigger is optimized on one batch of
flagged examples and evaluated on a disjoint held-out batch it never saw
during search. The number that matters is the held-out success rate, not
the optimize-set score, and the project explicitly reports the gap between
them. A large gap would mean the trigger fit that specific batch rather than
generalizing — which is exactly the failure mode this split is designed to
catch.

**Q: Your defense dropped ASR — but did it just start flagging everything?**
A: Checked explicitly. Alongside before/after ASR, the defense evaluation
reports clean-data accuracy, false-positive rate, and false-negative rate on
a fresh, disjoint, labeled set (both toxic and benign examples). If the
fine-tuned model's false-positive rate jumped, that would mean the
robustness gain came at the cost of usefully distinguishing benign from
toxic content, not from genuine robustness.

**Q: Your black-box attacks against the API — are you just replaying
something optimized elsewhere, or actually attacking the API directly?**
A: Both, and the project distinguishes them. `transfer_study.py` tests
surrogate-transfer: attack the cheap local model, replay the result against
the API. `blackbox_direct_attack_study.py` goes further — it runs a
score-guided genetic algorithm directly against the API's own continuous
score every generation, which is what a real black-box adversary targeting
that API would actually do, since both OpenAI Moderation and Perspective
expose continuous scores, not just accept/reject. Comparing the two ASRs
tells you how much attack strength surrogate-replay alone leaves on the
table.

**Q: Why a genetic algorithm for the black-box optimization rather than
just extending the greedy search?**
A: Greedy search commits early to one perturbation path and can get stuck
in a local optimum. A population-based genetic algorithm keeps several
candidate variants alive across generations, using crossover and mutation
to explore word-substitution and character-perturbation moves jointly
rather than in one fixed order. It also generalizes cleanly to black-box
targets because it only ever needs a continuous score — the same interface
the API targets and the local classifiers both expose.

**Q: A corrected p-value tells you two attacks differ significantly — but
how do you know the difference actually matters?**
A: Every pairwise comparison also reports an effect size, not just a
p-value: an odds ratio from McNemar's own 2x2 table, and a bootstrap
confidence interval on the raw ASR difference between the two attacks. A
statistically significant but practically tiny gap (say, a 2-point ASR
difference with a wide CI straddling zero) is visible as such, rather than
being reported as just "significant" and left ambiguous about magnitude.

**Q: Your universal trigger and direct black-box attack are both
stochastic searches — how do you know the result isn't just one lucky run?**
A: Both are run across multiple independent random seeds (the trigger
search re-splits optimize/holdout AND re-searches per seed; the black-box
genetic attack re-runs the full search per seed), and I report mean, std,
and min/max across seeds rather than a single number. High variance across
seeds would mean the result depends on which examples happened to be
sampled or searched, not a stable property of the attack — that's exactly
the failure mode this guards against, and it's what makes the held-out
generalization claim actually credible rather than a one-off.

**Q: You're running attacks directly against live production APIs — what
stops that from being irresponsible?**
A: Three things, all enforced in code, not just stated as intent: (1) a
hard ceiling on total real API queries per run (`QueryBudgetGuard`,
independent of sample size or seed count) that keeps usage within
reasonable/free-tier limits, (2) every input scored comes from existing
public research datasets, never live user content — there's no mechanism
in the project that could cause a real moderation failure on anything but
research-dataset text used for measurement, and (3) the script refuses to
run against real APIs at all unless you explicitly pass a flag confirming
you've read the responsible-disclosure expectations. Any specific
successful trigger or adversarial example would be disclosed to the
provider before any publication, per standard security research norms.

**Q: What's the universal trigger search actually testing, and why does it
matter more than the per-example attacks?**
A: Per-example attacks answer "is this specific input fragile?" The
universal trigger answers "is there one reusable string that degrades the
classifier broadly?" If such a trigger exists, an attacker doesn't need to
craft a custom attack per message — they reuse one string over and over.
That's a bigger real-world risk than any single per-example success.

**Q: What ethical safeguards are in place?**
A: No novel harmful content authored — all seed text comes from existing
public academic datasets. The universal trigger's candidate pool is
deliberately neutral filler phrases ("by the way", "for context"), not
harmful language. Any finding against a real production API is intended for
responsible disclosure, not publication as a usable exploit. The defense
experiment is included specifically so the project isn't attack-only.

**Q: What would you do with more time or compute?**
A: Add white-box gradient-based attacks (needs GPU + logit access, e.g. a
proper adversarial-suffix search), expand the local-model registry further,
and add human evaluation of "does this still read as toxic to a person"
rather than relying solely on an embedding-similarity proxy for meaning
preservation.

**Q: What's the single most interesting finding you'd highlight?**
A: (Answer this with your actual numbers once you've run it — see the
honesty note below.) Structurally, the most interesting *possible* finding
is high cross-model or cross-API transfer combined with low perturbation
cost: that would mean a cheap, fluent, reusable attack against an open
research model also threatens production systems.

---

## Be honest if pushed on this

**If asked "have you actually run this and gotten results?"** — be direct:
the codebase is fully built and tested (26/26 unit tests passing, verified
attack logic works correctly on a mock classifier), but real experimental
numbers require running it with real model/API access, which wasn't
available in the environment where it was built. Don't claim findings you
don't have. If you've since run it for real, replace this section's
examples with your actual numbers.

**If asked for a specific ASR/transfer-rate number** — don't guess or make
one up. Say "let me pull up the actual results" or "I haven't run that
specific ablation yet" rather than inventing a plausible-sounding figure.
