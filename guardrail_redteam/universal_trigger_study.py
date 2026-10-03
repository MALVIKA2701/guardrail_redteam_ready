"""
Universal trigger study, done properly: the trigger is optimized ONLY on an
optimize-set of flagged examples, then evaluated ONLY on a disjoint
held-out set it never saw during search. The headline "universal trigger"
number this project reports is the HELD-OUT success rate, not the
optimize-set score -- optimizing and evaluating on the same examples would
just measure overfitting to that specific batch, not a genuine generalizing
vulnerability.

Usage:
    python universal_trigger_study.py --n_samples 100
"""

import argparse
import json
import os
import random

import config
from data.load_data import load_dataset
from targets.local_classifier import LocalSurrogateClassifier
from attacks.trigger_search_attack import search_universal_trigger, evaluate_trigger_transfer
from stats import seed_variance_summary


def split_optimize_holdout(flagged_texts, holdout_fraction: float = None, seed: int = None):
    holdout_fraction = holdout_fraction if holdout_fraction is not None else config.TRIGGER_HOLDOUT_FRACTION
    seed = seed if seed is not None else config.RANDOM_SEED
    rng = random.Random(seed)
    shuffled = list(flagged_texts)
    rng.shuffle(shuffled)
    n_holdout = max(1, int(len(shuffled) * holdout_fraction))
    holdout = shuffled[:n_holdout]
    optimize = shuffled[n_holdout:]
    return optimize, holdout


def run_single_seed(optimize_set, holdout_set, classifier, seed: int) -> dict:
    """
    Runs the search+evaluation once for a given seed. The split itself is
    ALSO reseeded per run (not just the search), so seed variance reflects
    both "did the search get lucky" and "did the optimize/holdout split
    happen to be favorable" -- the two things that could make a single run
    look stronger than the procedure really is.
    """
    search_result = search_universal_trigger(
        optimize_set,
        score_fn=classifier.score,
        max_queries=config.MAX_QUERIES_PER_EXAMPLE * len(optimize_set),
        seed=seed,
    )
    holdout_result = evaluate_trigger_transfer(
        search_result["trigger"], holdout_set, classifier.score,
    )
    optimize_result = evaluate_trigger_transfer(
        search_result["trigger"], optimize_set, classifier.score,
    )
    return {
        "seed": seed,
        "trigger": search_result["trigger"],
        "optimize_set_success_rate": optimize_result["holdout_success_rate"],
        "held_out_success_rate": holdout_result["holdout_success_rate"],
        "search_queries_used": search_result["queries_used"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=config.N_SAMPLES)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--n_seeds", type=int, default=5,
        help="Number of independent (re-split + re-search) runs. A single "
             "seed cannot distinguish a genuinely strong trigger from one "
             "lucky search run -- report mean/spread across seeds instead.",
    )
    args = parser.parse_args()
    if args.mock:
        config.ALLOW_OFFLINE_FALLBACK = True

    data = load_dataset(n_samples=args.n_samples)

    if args.mock:
        from targets.local_classifier import MockClassifier
        classifier = MockClassifier()
    else:
        classifier = LocalSurrogateClassifier()

    flagged = [t for t, _ in data if classifier.score(t) >= config.LOCAL_SURROGATE_THRESHOLD]
    print(f"{len(flagged)} / {len(data)} examples currently flagged by the classifier.")

    per_seed_results = []
    for seed in range(args.n_seeds):
        optimize_set, holdout_set = split_optimize_holdout(flagged, seed=config.RANDOM_SEED + seed)
        if len(optimize_set) < 3 or len(holdout_set) < 1:
            print(f"[seed {seed}] Not enough flagged examples for a meaningful split. "
                  f"Increase --n_samples. Skipping this seed.")
            continue
        print(f"\n=== Seed {seed}: {len(optimize_set)} optimize / {len(holdout_set)} held-out ===")
        result = run_single_seed(optimize_set, holdout_set, classifier, seed=seed)
        print(f"  trigger={result['trigger']!r}  "
              f"held_out_ASR={result['held_out_success_rate']}  "
              f"optimize_ASR={result['optimize_set_success_rate']}")
        per_seed_results.append(result)

    if not per_seed_results:
        print("No seed produced a usable split -- increase --n_samples and re-run.")
        return

    held_out_values = [r["held_out_success_rate"] for r in per_seed_results if r["held_out_success_rate"] is not None]
    optimize_values = [r["optimize_set_success_rate"] for r in per_seed_results if r["optimize_set_success_rate"] is not None]

    held_out_summary = seed_variance_summary(held_out_values)
    optimize_summary = seed_variance_summary(optimize_values)
    generalization_gap_mean = (
        round(optimize_summary["mean"] - held_out_summary["mean"], 4)
        if optimize_summary["mean"] is not None and held_out_summary["mean"] is not None
        else None
    )

    report = {
        "per_seed_results": per_seed_results,
        "held_out_success_rate_across_seeds": held_out_summary,
        "optimize_set_success_rate_across_seeds": optimize_summary,
        "generalization_gap_mean": generalization_gap_mean,
        "note": (
            "The 'universal trigger' claim should be reported as "
            "held_out_success_rate_across_seeds (mean +/- std / min-max), "
            "NOT a single seed's number and NOT optimize_set_success_rate. "
            "High variance across seeds (large std, wide min-max) means the "
            "result depends heavily on which examples happened to be "
            "sampled/searched, not a stable property of the trigger."
        ),
    }
    print("\n=== Aggregate across seeds ===")
    print(json.dumps({
        "held_out_success_rate_across_seeds": held_out_summary,
        "optimize_set_success_rate_across_seeds": optimize_summary,
        "generalization_gap_mean": generalization_gap_mean,
    }, indent=2))

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(config.TRIGGER_STUDY_RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved trigger study results to {config.TRIGGER_STUDY_RESULTS_PATH}")


if __name__ == "__main__":
    main()
