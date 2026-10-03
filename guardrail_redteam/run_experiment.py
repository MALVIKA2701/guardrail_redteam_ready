"""
Entry point: loads data, runs the local-surrogate attack evaluation, saves
results and a markdown summary.

Usage:
    python run_experiment.py --dataset jigsaw_toxicity_pred --n_samples 200 --attacks char,paraphrase
"""

import argparse
import json
import os

import config
from data.load_data import load_dataset
from targets.local_classifier import LocalSurrogateClassifier, MockClassifier
from evaluate import run_attacks, summarize, save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=config.DATASET_NAME)
    parser.add_argument("--n_samples", type=int, default=config.N_SAMPLES)
    parser.add_argument("--attacks", default="char,word,paraphrase")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the dependency-free MockClassifier instead of the real "
             "HF model. For offline pipeline smoke-testing only.",
    )
    args = parser.parse_args()
    if args.mock:
        config.ALLOW_OFFLINE_FALLBACK = True

    print(f"Loading dataset '{args.dataset}' (n={args.n_samples})...")
    data = load_dataset(name=args.dataset, n_samples=args.n_samples)
    print(f"Loaded {len(data)} examples.")

    if args.mock:
        print("Using MockClassifier (offline smoke-test mode).")
        classifier = MockClassifier()
    else:
        print(f"Loading local surrogate classifier: {config.LOCAL_SURROGATE_MODEL}")
        classifier = LocalSurrogateClassifier()

    attack_names = [a.strip() for a in args.attacks.split(",") if a.strip()]
    print(f"Running attacks: {attack_names}")

    records = run_attacks(data, classifier, attack_names=attack_names)
    save_results(records)

    summary = summarize(records)
    print(json.dumps(summary, indent=2))

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    md_path = os.path.join(config.RESULTS_DIR, "summary.md")
    with open(md_path, "w") as f:
        f.write("# Attack Evaluation Summary\n\n")
        f.write(f"Dataset: `{args.dataset}`, n={len(data)}, "
                f"classifier: `{'mock' if args.mock else config.LOCAL_SURROGATE_MODEL}`\n\n")
        f.write("| Attack | n | ASR | Avg Queries | Avg Edit Dist | Avg Sim (successes) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, stats in summary.items():
            if stats.get("n", 0) == 0:
                continue
            f.write(
                f"| {name} | {stats['n']} | {stats['attack_success_rate']:.2%} | "
                f"{stats['avg_queries_used']} | {stats['avg_normalized_edit_distance']} | "
                f"{stats['avg_semantic_similarity_of_successes']} |\n"
            )
    print(f"Markdown summary written to {md_path}")


if __name__ == "__main__":
    main()
