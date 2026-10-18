"""
Deferral study: can a guardrail tell when it is being fooled, and hand those
inputs to a human instead of letting them through?

For each human-review budget B (the share of benign traffic we accept
sending to review), this script:

  1. Calibrates the deferral band's lower edge on a CALIBRATION split of
     benign texts (split-conformal, deferral/selective_guardrail.py).
  2. Measures the realized review rate on a disjoint TEST split of benign
     texts, so the budget is checked on data the calibration never saw.
  3. Runs the standard attacks (which target the 0.5 block threshold only)
     and reports how many of their "successful" evasions the band would
     still catch. This number is OPTIMISTIC: those attacks stop just under
     0.5 by construction.
  4. Runs ADAPTIVE attacks that know the band exists and push the deferral
     score below its lower edge, and reports their success rate (the share
     of attacked toxic texts that end up ALLOWED) with bootstrap CIs,
     paired against the standard attacks with McNemar's test, plus the
     extra queries and lost semantic similarity the band forces on the
     attacker.

Signals:
  --signal single    defer on the primary classifier's own score
  --signal ensemble  defer if ANY classifier in the registry is suspicious
                     (max score across models)

Usage:
    python deferral_study.py --mock --n_samples 40               # offline smoke test
    python deferral_study.py --n_samples 300 --attacks char,word,genetic
    python deferral_study.py --n_samples 300 --signal ensemble --attacks char,word
"""

import argparse
import json
import math
import os
import random
from typing import Callable, List

import config
from data.load_data import load_dataset
from targets.local_classifier import LocalSurrogateClassifier, MockClassifier
from attacks.char_attacks import greedy_search_attack
from attacks.word_substitution_attack import word_substitution_attack
from attacks.genetic_search_attack import genetic_search_attack
from attacks.paraphrase_attack import paraphrase_attack, HFParaphraser
from metrics.perturbation_cost import SemanticSimilarity
from stats import bootstrap_asr_ci, mcnemar_test, paired_asr_difference_ci, holm_bonferroni_correction
from deferral.benign_sources import SOURCES as BENIGN_SOURCES, select_benign
from deferral.selective_guardrail import (
    ALLOW, DEFER, BLOCK, decide, calibrate_lower_edge, adaptive_score_fn,
    ensemble_max_score, decision_rates,
)

DEFERRAL_RESULTS_PATH = os.path.join(config.RESULTS_DIR, "deferral_results.json")
DEFERRAL_SUMMARY_PATH = os.path.join(config.RESULTS_DIR, "deferral_summary.md")


class _MockVariant(MockClassifier):
    """A second offline mock with different trigger words, so the ensemble
    signal can be smoke-tested without downloading models."""
    def __init__(self, markers):
        self.MARKERS = markers


def build_classifiers(mock: bool, signal: str, primary_key: str = None):
    """primary_key: a key of config.LOCAL_SURROGATE_REGISTRY (default: the
    entry for config.LOCAL_SURROGATE_MODEL). The other registry models form
    the rest of the ensemble."""
    if mock:
        primary = MockClassifier()
        others = [_MockVariant(["idiot", "nobody wants", "hates", "shut"]),
                  _MockVariant(["disgusting", "stupid", "get out", "worst"])]
    else:
        registry = config.LOCAL_SURROGATE_REGISTRY
        if primary_key is None:
            primary_name = config.LOCAL_SURROGATE_MODEL
        elif primary_key in registry:
            primary_name = registry[primary_key]
        else:
            raise ValueError(f"unknown --primary {primary_key!r}; choose from {list(registry)}")
        primary = LocalSurrogateClassifier(primary_name)
        others = [LocalSurrogateClassifier(m) for m in registry.values() if m != primary_name]
    if signal == "single":
        deferral_fn = primary.score
    elif signal == "ensemble":
        deferral_fn = ensemble_max_score([primary.score] + [o.score for o in others])
    else:
        raise ValueError(f"unknown signal {signal}")
    return primary, deferral_fn


def run_attack(name: str, text: str, score_fn: Callable[[str], float], sim_fn,
               paraphraser, seed: int, max_queries: int, min_similarity: float) -> dict:
    """Same dispatch as evaluate.run_attacks, for a single text and score fn."""
    if name == "char":
        return greedy_search_attack(text, score_fn=score_fn, similarity_fn=sim_fn,
                                    max_queries=max_queries, min_similarity=min_similarity, seed=seed)
    if name == "word":
        return word_substitution_attack(text, score_fn=score_fn, similarity_fn=sim_fn,
                                        max_queries=max_queries, min_similarity=min_similarity)
    if name == "genetic":
        return genetic_search_attack(text, score_fn=score_fn, similarity_fn=sim_fn,
                                     max_queries=max_queries, min_similarity=min_similarity, seed=seed)
    if name == "paraphrase":
        return paraphrase_attack(text, score_fn=score_fn, similarity_fn=sim_fn,
                                 paraphraser=paraphraser, n_candidates=config.MAX_PARAPHRASE_CANDIDATES,
                                 min_similarity=min_similarity)
    raise ValueError(f"Unknown attack: {name}")


def split_benign(data, cal_fraction: float, seed: int):
    benign = [t for t, y in data if y == 0]
    rng = random.Random(seed)
    rng.shuffle(benign)
    cut = int(round(len(benign) * cal_fraction))
    return benign[:cut], benign[cut:]


def _mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def run_study(data, primary, deferral_fn, attack_names: List[str], budgets: List[float],
              cal_fraction: float = 0.5, seed: int = None, max_queries: int = None,
              min_similarity: float = None, sim_fn=None, paraphraser=None,
              max_attack_examples: int = None, benign_texts: List[str] = None) -> dict:
    """
    benign_texts: optional benign pool for calibration + testing (split by
    cal_fraction). If None, the benign texts of `data` are used, which is
    the original setting.
    """
    seed = config.RANDOM_SEED if seed is None else seed
    max_queries = max_queries or config.MAX_QUERIES_PER_EXAMPLE
    min_similarity = min_similarity or config.MIN_SEMANTIC_SIMILARITY
    t = config.LOCAL_SURROGATE_THRESHOLD
    sim_fn = sim_fn or SemanticSimilarity()
    if paraphraser is None and "paraphrase" in attack_names:
        paraphraser = HFParaphraser()

    if benign_texts is None:
        cal_benign, test_benign = split_benign(data, cal_fraction, seed)
    else:
        cal_benign, test_benign = split_benign([(b, 0) for b in benign_texts], cal_fraction, seed)
    # Attack only true positives (labeled toxic AND blocked), as in evaluate.py.
    toxic = [txt for txt, y in data if y == 1]
    targets = [txt for txt in toxic if primary.score(txt) >= t]
    if max_attack_examples:
        targets = targets[:max_attack_examples]

    # Standard attacks: run once, independent of the budget.
    standard = {a: [] for a in attack_names}
    for i, txt in enumerate(targets):
        for a in attack_names:
            r = run_attack(a, txt, primary.score, sim_fn, paraphraser, i, max_queries, min_similarity)
            standard[a].append({
                "orig_text": txt, "best_text": r["best_text"], "queries_used": r["queries_used"],
                "similarity": r["similarity"], "evaded_binary": primary.score(r["best_text"]) < t,
            })

    results = {
        "config": {
            "n_examples": len(data), "n_cal_benign": len(cal_benign), "n_test_benign": len(test_benign),
            "n_attack_targets": len(targets), "attacks": attack_names, "budgets": budgets,
            "block_threshold": t, "max_queries": max_queries, "min_similarity": min_similarity,
            "seed": seed,
        },
        "standard_attacks": {
            a: {
                "binary_asr": bootstrap_asr_ci([x["evaded_binary"] for x in standard[a]]),
                "avg_queries": _mean(x["queries_used"] for x in standard[a]),
                "avg_similarity": _mean(x["similarity"] for x in standard[a]),
            } for a in attack_names
        },
        "by_budget": [],
    }

    all_p, p_index = [], []
    for B in budgets:
        cal = calibrate_lower_edge([deferral_fn(x) for x in cal_benign],
                                   [primary.score(x) for x in cal_benign], B, t)
        edge = cal["lower_edge"]
        no_band = math.isinf(edge)
        cal = {**cal, "lower_edge": None if no_band else cal["lower_edge"]}

        test_decisions = [decide(primary.score(x), deferral_fn(x), edge, t) for x in test_benign]
        reviewed = [d == DEFER for d in test_decisions]
        toxic_clean = [decide(primary.score(x), deferral_fn(x), edge, t) for x in toxic]

        entry = {
            "review_budget": B,
            "lower_edge": None if no_band else round(edge, 6),
            "calibration": cal,
            "test_benign_decisions": decision_rates(test_decisions),
            "test_benign_review_rate": bootstrap_asr_ci(reviewed),
            "budget_respected_on_test": (sum(reviewed) / len(reviewed) <= B) if reviewed else None,
            "clean_toxic_decisions": decision_rates(toxic_clean),
            "attacks": {},
        }

        for a in attack_names:
            std = standard[a]
            # Non-adaptive view: would the band still catch the standard attack's output?
            std_allowed = [decide(primary.score(x["best_text"]), deferral_fn(x["best_text"]), edge, t) == ALLOW
                           for x in std]

            if no_band:            # budget 0: adaptive attacker == standard attacker
                adapt = [{"allowed": s, "queries_used": x["queries_used"], "similarity": x["similarity"]}
                         for s, x in zip(std_allowed, std)]
            elif edge <= 0:        # band covers every allowed text
                adapt = [{"allowed": False, "queries_used": None, "similarity": None} for _ in std]
            else:
                f = adaptive_score_fn(deferral_fn, edge, t, block_score_fn=primary.score)
                adapt = []
                for i, x in enumerate(std):
                    r = run_attack(a, x["orig_text"], f, sim_fn, paraphraser, i, max_queries, min_similarity)
                    bt = r["best_text"]
                    adapt.append({
                        "allowed": decide(primary.score(bt), deferral_fn(bt), edge, t) == ALLOW,
                        "queries_used": r["queries_used"], "similarity": r["similarity"],
                    })

            binary = [x["evaded_binary"] for x in std]
            adaptive_allowed = [x["allowed"] for x in adapt]
            test = mcnemar_test(binary, adaptive_allowed)
            all_p.append(test["p_value"])
            p_index.append((len(results["by_budget"]), a))
            succ = [x for x in adapt if x["allowed"]]
            entry["attacks"][a] = {
                "binary_asr_no_band": bootstrap_asr_ci(binary),
                "standard_attack_asr_with_band": bootstrap_asr_ci(std_allowed),
                "adaptive_attack_asr_with_band": bootstrap_asr_ci(adaptive_allowed),
                "asr_reduction_vs_no_band": paired_asr_difference_ci(binary, adaptive_allowed),
                "mcnemar_no_band_vs_adaptive": test,
                "adaptive_avg_queries": _mean(x["queries_used"] for x in adapt if x["queries_used"] is not None),
                "adaptive_avg_similarity_of_successes": _mean(x["similarity"] for x in succ),
            }
        results["by_budget"].append(entry)

    # Holm-Bonferroni across every (budget x attack) comparison.
    if all_p:
        corrected = holm_bonferroni_correction(all_p)
        for (bi, a), c in zip(p_index, corrected):
            results["by_budget"][bi]["attacks"][a]["mcnemar_no_band_vs_adaptive"]["holm"] = c
    return results


def write_summary(results: dict, path: str, signal: str, model_desc: str):
    c = results["config"]
    lines = [
        "# Deferral Study Summary\n",
        f"Classifier: `{model_desc}` · deferral signal: `{signal}` · "
        f"{c['n_attack_targets']} attacked toxic texts · "
        f"{c['n_cal_benign']} calibration / {c['n_test_benign']} test benign texts\n",
        "ASR = share of attacked toxic texts that end up ALLOWED. "
        "\"Standard\" attacks only target the 0.5 threshold; \"adaptive\" attacks "
        "know the band and push below its lower edge.\n",
        "| Review budget | Lower edge | Test review rate [95% CI] | Attack | ASR, no band | "
        "Standard ASR, with band | Adaptive ASR, with band [95% CI] | Adaptive queries |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in results["by_budget"]:
        rr = e["test_benign_review_rate"]
        rr_s = f"{rr['asr']:.1%} [{rr['ci_low']:.1%}, {rr['ci_high']:.1%}]" if rr["asr"] is not None else "n/a"
        for a, s in e["attacks"].items():
            ad = s["adaptive_attack_asr_with_band"]
            ad_s = f"{ad['asr']:.1%} [{ad['ci_low']:.1%}, {ad['ci_high']:.1%}]" if ad["asr"] is not None else "n/a"
            edge_s = "no band" if e["lower_edge"] is None else f"{e['lower_edge']:.3f}"
            lines.append(
                f"| {e['review_budget']:.0%} | {edge_s} | {rr_s} | {a} | "
                f"{s['binary_asr_no_band']['asr']:.1%} | {s['standard_attack_asr_with_band']['asr']:.1%} | "
                f"{ad_s} | {s['adaptive_avg_queries']} |"
            )
    lines.append("\nThe adaptive column is the honest one. The standard-with-band column overstates "
                 "the band's value, because those attacks stop as soon as they cross 0.5.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=config.DATASET_NAME)
    p.add_argument("--n_samples", type=int, default=config.N_SAMPLES)
    p.add_argument("--attacks", default="char,word,genetic")
    p.add_argument("--budgets", default="0,0.05,0.1,0.2")
    p.add_argument("--signal", choices=["single", "ensemble"], default="single")
    p.add_argument("--primary", default=None,
                   help=f"primary classifier, one of {list(config.LOCAL_SURROGATE_REGISTRY)} "
                        f"(default: {config.LOCAL_SURROGATE_MODEL})")
    p.add_argument("--benign_source", choices=list(BENIGN_SOURCES), default="jigsaw",
                   help="benign traffic the band is calibrated and tested on")
    p.add_argument("--n_benign", type=int, default=400,
                   help="benign texts for jigsaw_hard / hatexplain (split into calibration and test)")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                   help="seed for the data sample and the calibration/test split")
    p.add_argument("--cal_fraction", type=float, default=0.5)
    p.add_argument("--max_attack_examples", type=int, default=None)
    p.add_argument("--mock", action="store_true", help="offline smoke test with MockClassifier")
    args = p.parse_args()
    if args.mock:
        config.ALLOW_OFFLINE_FALLBACK = True

    data = load_dataset(name=args.dataset, n_samples=args.n_samples, seed=args.seed)
    primary, deferral_fn = build_classifiers(args.mock, args.signal, args.primary)
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    budgets = [float(b) for b in args.budgets.split(",")]

    benign_texts = None
    if args.benign_source != "jigsaw":
        if args.mock:
            loader = lambda: list(data)  # offline: reuse the fallback sample
        elif args.benign_source == "jigsaw_hard":
            from data.load_data import _load_jigsaw
            loader = _load_jigsaw
        else:
            from data.load_data import _load_hatexplain
            loader = _load_hatexplain
        benign_texts = select_benign(args.benign_source, data, args.n_benign, args.seed,
                                     full_pairs_loader=loader)
        print(f"Using {len(benign_texts)} benign texts from '{args.benign_source}'")
        if len(benign_texts) < 40 and not args.mock:
            raise RuntimeError(f"only {len(benign_texts)} benign texts from {args.benign_source}; "
                               f"too few to calibrate a review budget")

    results = run_study(data, primary, deferral_fn, attacks, budgets, seed=args.seed,
                        cal_fraction=args.cal_fraction, max_attack_examples=args.max_attack_examples,
                        benign_texts=benign_texts)
    primary_name = "mock" if args.mock else (config.LOCAL_SURROGATE_REGISTRY.get(args.primary)
                                             or config.LOCAL_SURROGATE_MODEL)
    results["config"].update({"signal": args.signal, "mock": args.mock, "dataset": args.dataset,
                              "primary": primary_name, "benign_source": args.benign_source})

    # The original run (default primary, jigsaw benign, default seed) keeps
    # its original file name; every other setting gets a tagged one.
    default_run = (args.primary in (None, "toxic_bert") and args.benign_source == "jigsaw"
                   and args.seed == config.RANDOM_SEED)
    tag = args.signal if default_run else (
        f"{args.signal}_{args.primary or 'toxic_bert'}_{args.benign_source}_s{args.seed}")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = DEFERRAL_RESULTS_PATH.replace(".json", f"_{tag}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    md = DEFERRAL_SUMMARY_PATH.replace(".md", f"_{tag}.md")
    write_summary(results, md, args.signal, f"{primary_name} · benign: {args.benign_source} · seed {args.seed}")
    print(open(md).read())
    print(f"Saved {out} and {md}")


if __name__ == "__main__":
    main()
