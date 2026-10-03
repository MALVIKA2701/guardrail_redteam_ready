"""
Multi-model study: runs the full attack suite against every model in
config.LOCAL_SURROGATE_REGISTRY (not just the single primary surrogate), and
builds a surrogate-to-surrogate transfer matrix -- do adversarial examples
crafted against one guardrail model fool a *different* guardrail model, even
before we get to the black-box API transfer study?

This is what turns "I attacked one classifier" into "I characterized a class
of vulnerability across guardrail architectures," which is a much stronger
empirical claim for a portfolio project.

Usage:
    python multi_model_study.py --n_samples 100 --attacks char,word,paraphrase
"""

import argparse
import json
import os
from typing import Dict, List

import config
from data.load_data import load_dataset
from targets.local_classifier import LocalSurrogateClassifier
from evaluate import run_attacks, summarize
from stats import compare_attacks_across_models


def run_multi_model_study(
    data,
    registry: Dict[str, str] = None,
    attack_names: List[str] = ("char", "word", "paraphrase"),
) -> Dict[str, dict]:
    registry = registry or config.LOCAL_SURROGATE_REGISTRY
    all_records = {}
    classifiers = {}

    for name, model_id in registry.items():
        print(f"\n=== Attacking surrogate '{name}' ({model_id}) ===")
        clf = LocalSurrogateClassifier(model_name=model_id)
        classifiers[name] = clf
        records = run_attacks(data, clf, attack_names=attack_names)
        all_records[name] = records
        summary = summarize(records)
        print(json.dumps(summary["overall"], indent=2))

    return all_records, classifiers


def build_cross_model_transfer_matrix(
    all_records: Dict[str, List[dict]],
    classifiers: Dict[str, "LocalSurrogateClassifier"],
) -> Dict[str, Dict[str, float]]:
    """
    For every pair (source_model, target_model), measures what fraction of
    source_model's successful adversarial examples still evade target_model.
    Diagonal entries (source == target) are trivially 1.0 by construction and
    included only for reference.
    """
    matrix = {}
    for source_name, records in all_records.items():
        successes = [r for r in records if r["success"]]
        matrix[source_name] = {}
        for target_name, target_clf in classifiers.items():
            if not successes:
                matrix[source_name][target_name] = None
                continue
            if source_name == target_name:
                matrix[source_name][target_name] = 1.0
                continue
            transferred = 0
            for r in successes:
                if target_clf.score(r["best_text"]) < config.LOCAL_SURROGATE_THRESHOLD:
                    transferred += 1
            matrix[source_name][target_name] = round(transferred / len(successes), 4)
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=config.N_SAMPLES)
    parser.add_argument("--attacks", default="char,word,paraphrase")
    args = parser.parse_args()

    attack_names = [a.strip() for a in args.attacks.split(",") if a.strip()]
    data = load_dataset(n_samples=args.n_samples)

    all_records, classifiers = run_multi_model_study(data, attack_names=attack_names)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(config.MULTI_MODEL_RESULTS_PATH, "w") as f:
        for model_name, records in all_records.items():
            for r in records:
                f.write(json.dumps({"model": model_name, **r}) + "\n")
    print(f"\nSaved per-model attack records to {config.MULTI_MODEL_RESULTS_PATH}")

    print("\n=== Building cross-model transfer matrix ===")
    matrix = build_cross_model_transfer_matrix(all_records, classifiers)
    print(json.dumps(matrix, indent=2))
    with open(config.SURROGATE_TRANSFER_MATRIX_PATH, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"Saved cross-model transfer matrix to {config.SURROGATE_TRANSFER_MATRIX_PATH}")

    print("\n=== Pairwise attack comparisons within each model, "
          "Holm-Bonferroni corrected across the FULL family of tests ===")
    stats_result = compare_attacks_across_models(all_records)
    print(f"Total tests in family: {stats_result['n_tests_in_family']}")
    print(json.dumps(stats_result["corrected_across_full_family"], indent=2))
    with open(config.MULTI_MODEL_STATS_PATH, "w") as f:
        json.dump(stats_result, f, indent=2)
    print(f"Saved corrected multi-model statistics to {config.MULTI_MODEL_STATS_PATH}")


if __name__ == "__main__":
    main()
