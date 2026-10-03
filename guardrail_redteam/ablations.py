"""
Ablation studies: how does Attack Success Rate change as we vary the query
budget, the semantic-similarity floor, and (for char attacks) the
perturbation rate? These sweeps are what let the paper/portfolio say
something like "ASR saturates after ~20 queries" or "requiring similarity
>= 0.9 cuts ASR roughly in half" instead of reporting a single number at one
arbitrarily chosen setting.

Usage:
    python ablations.py --n_samples 60
"""

import argparse
import json
import os

import config
from data.load_data import load_dataset
from targets.local_classifier import LocalSurrogateClassifier
from attacks.char_attacks import greedy_search_attack
from metrics.perturbation_cost import SemanticSimilarity


def sweep_query_budget(data, classifier, sim_fn, budgets=None):
    budgets = budgets or config.QUERY_BUDGET_SWEEP
    results = []
    flagged = [(t, l) for t, l in data if classifier.score(t) >= config.LOCAL_SURROGATE_THRESHOLD]
    for budget in budgets:
        successes = 0
        for text, _ in flagged:
            r = greedy_search_attack(
                text, score_fn=classifier.score, similarity_fn=sim_fn,
                max_queries=budget, min_similarity=config.MIN_SEMANTIC_SIMILARITY,
            )
            successes += int(r["success"])
        asr = successes / len(flagged) if flagged else None
        results.append({"query_budget": budget, "n": len(flagged), "asr": round(asr, 4) if asr is not None else None})
        print(f"  query_budget={budget}: ASR={asr:.2%}" if asr is not None else f"  query_budget={budget}: n=0")
    return results


def sweep_similarity_threshold(data, classifier, sim_fn, thresholds=None):
    thresholds = thresholds or config.SIMILARITY_THRESHOLD_SWEEP
    results = []
    flagged = [(t, l) for t, l in data if classifier.score(t) >= config.LOCAL_SURROGATE_THRESHOLD]
    for thresh in thresholds:
        successes = 0
        for text, _ in flagged:
            r = greedy_search_attack(
                text, score_fn=classifier.score, similarity_fn=sim_fn,
                max_queries=config.MAX_QUERIES_PER_EXAMPLE, min_similarity=thresh,
            )
            successes += int(r["success"])
        asr = successes / len(flagged) if flagged else None
        results.append({"min_similarity": thresh, "n": len(flagged), "asr": round(asr, 4) if asr is not None else None})
        print(f"  min_similarity={thresh}: ASR={asr:.2%}" if asr is not None else f"  min_similarity={thresh}: n=0")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=60)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    if args.mock:
        config.ALLOW_OFFLINE_FALLBACK = True

    data = load_dataset(n_samples=args.n_samples)

    if args.mock:
        from targets.local_classifier import MockClassifier
        classifier = MockClassifier()
        sim_fn = lambda a, b: SemanticSimilarity._jaccard_fallback(a, b)
    else:
        classifier = LocalSurrogateClassifier()
        sim_fn = SemanticSimilarity()

    print("Sweeping query budget...")
    budget_results = sweep_query_budget(data, classifier, sim_fn)

    print("Sweeping similarity threshold...")
    similarity_results = sweep_similarity_threshold(data, classifier, sim_fn)

    all_results = {
        "query_budget_sweep": budget_results,
        "similarity_threshold_sweep": similarity_results,
    }

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(config.ABLATION_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved ablation results to {config.ABLATION_RESULTS_PATH}")


if __name__ == "__main__":
    main()
