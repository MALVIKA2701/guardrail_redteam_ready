"""
Transfer study: takes the adversarial texts generated against the local
white/gray-box surrogate (evaluate.py output) and re-queries them against
black-box API targets with zero additional adaptation, to measure how much
attack success transfers from a cheap local model to production systems.

This is the paper's core empirical question: are guardrail models a shared
attack surface, or does each need to be attacked independently?

Usage:
    python transfer_study.py --results results/attack_results.jsonl
"""

import argparse
import json
import os

import config
from targets.api_moderation import get_enabled_api_targets
from evaluate import load_results


def run_transfer_study(records, api_targets: dict) -> dict:
    """
    For every successful surrogate attack, re-check the perturbed text
    against each black-box API target and record whether it still evades.
    """
    successes = [r for r in records if r["success"]]
    print(f"{len(successes)} surrogate-successful adversarial examples to test for transfer.")

    matrix = {name: {"n": 0, "transferred": 0} for name in api_targets}

    for r in successes:
        for name, target in api_targets.items():
            try:
                flagged = target.is_flagged(r["best_text"])
                matrix[name]["n"] += 1
                if not flagged:
                    matrix[name]["transferred"] += 1
            except Exception as e:
                print(f"  [warn] {name} query failed for example {r['example_idx']}: {e!r}")

    results = {}
    for name, counts in matrix.items():
        n = counts["n"]
        results[name] = {
            "n_tested": n,
            "n_transferred": counts["transferred"],
            "transfer_rate": round(counts["transferred"] / n, 4) if n else None,
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=config.ATTACK_RESULTS_PATH)
    args = parser.parse_args()

    records = load_results(args.results)
    api_targets = get_enabled_api_targets()

    if not api_targets:
        print(
            "No API targets enabled (set OPENAI_API_KEY / PERSPECTIVE_API_KEY "
            "env vars and USE_API_TARGETS = True in config.py). "
            "Nothing to do -- this script only measures transfer to black-box "
            "APIs, not local-surrogate results (see evaluate.py for those)."
        )
        return

    results = run_transfer_study(records, api_targets)
    print(json.dumps(results, indent=2))

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(config.TRANSFER_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved transfer study results to {config.TRANSFER_RESULTS_PATH}")


if __name__ == "__main__":
    main()
