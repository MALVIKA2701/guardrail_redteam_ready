"""
Direct black-box attack: runs the score-guided genetic search DIRECTLY
against production moderation APIs (querying their real continuous score on
every candidate), rather than optimizing against the cheap local surrogate
and merely replaying the result (which is what transfer_study.py measures).

This is the stronger, more realistic black-box threat model: a real
adversary attacking OpenAI Moderation or Perspective directly would use
exactly this kind of score-guided search, not a one-shot replay of an
attack built elsewhere. Comparing this script's ASR against
transfer_study.py's transfer rate tells you how much attack strength was
left on the table by only testing transfer.

=== Responsible use of this script ===
This is the one part of the project that queries a live, third-party
production system you don't own. Before running it:
  1. A hard ceiling on total API queries per run is enforced
     (config.MAX_TOTAL_API_QUERIES_PER_RUN) regardless of --n_samples,
     --max_queries, or --n_seeds -- this keeps usage well within free-tier /
     reasonable-use limits and prevents runaway cost or query volume.
  2. This script only ever scores TEXT FROM THE RESEARCH DATASET (existing
     public toxicity-benchmark examples) -- it does not attempt to cause a
     moderation failure on any real user's live content, and produces no
     output intended for deployment anywhere.
  3. Findings should be responsibly disclosed to the API provider (OpenAI /
     Google Perspective) before any publication of specific successful
     adversarial examples or trigger strings.
  4. You must pass --i-accept-responsible-use to run this against real
     APIs, confirming you've read the above.

Usage:
    python blackbox_direct_attack_study.py --n_samples 20 --max_queries 60 --n_seeds 3 --i-accept-responsible-use
"""

import argparse
import json
import os

import config
from data.load_data import load_dataset
from targets.api_moderation import get_enabled_api_targets
from attacks.genetic_search_attack import genetic_search_attack
from metrics.perturbation_cost import SemanticSimilarity
from stats import seed_variance_summary


class QueryBudgetExceeded(Exception):
    pass


class QueryBudgetGuard:
    """
    Wraps a target's .score() to enforce a hard ceiling on total real API
    queries across the ENTIRE script invocation (all examples, all seeds,
    all targets combined) -- independent of what --n_samples, --max_queries,
    or --n_seeds happen to multiply out to.
    """

    def __init__(self, score_fn, cap: int):
        self._score_fn = score_fn
        self.cap = cap
        self.count = 0

    def __call__(self, text: str) -> float:
        if self.count >= self.cap:
            raise QueryBudgetExceeded(
                f"Query budget of {self.cap} exhausted -- stopping to respect "
                f"the responsible-use cap on real API queries."
            )
        self.count += 1
        return self._score_fn(text)


def run_direct_blackbox_attack(data, guarded_score_fn, sim_fn, max_queries, seed):
    """
    Runs the genetic search directly against a black-box target's real score
    function (wrapped in a QueryBudgetGuard), for ONE random seed.
    """
    results = []
    for idx, (text, label) in enumerate(data):
        try:
            base_score = guarded_score_fn(text)
        except QueryBudgetExceeded as e:
            print(f"  [budget] {e}")
            break
        except Exception as e:
            print(f"  [warn] scoring failed for example {idx}: {e!r}")
            continue

        if base_score < 0.5:
            continue

        try:
            result = genetic_search_attack(
                text,
                score_fn=guarded_score_fn,
                similarity_fn=sim_fn,
                max_queries=max_queries,
                min_similarity=config.MIN_SEMANTIC_SIMILARITY,
                seed=seed * 10_000 + idx,  # distinct RNG stream per (seed, example)
            )
        except QueryBudgetExceeded as e:
            print(f"  [budget] {e}")
            break

        results.append({"example_idx": idx, "orig_text": text, **result})
        print(f"  example {idx}: {base_score:.3f} -> {result['best_score']:.3f} "
              f"(success={result['success']}, queries={result['queries_used']})")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=20,
                         help="Kept small by default since this costs real API queries.")
    parser.add_argument("--max_queries", type=int, default=60,
                         help="Per-example query budget for the genetic search.")
    parser.add_argument(
        "--n_seeds", type=int, default=3,
        help="Independent runs with different RNG seeds. A single seed's ASR "
             "can't distinguish a genuinely strong attack from a lucky "
             "search -- report mean/spread across seeds instead.",
    )
    parser.add_argument(
        "--i-accept-responsible-use", action="store_true", dest="accept",
        help="Required to actually run against real APIs -- see the module "
             "docstring's responsible-use section.",
    )
    args = parser.parse_args()

    api_targets = get_enabled_api_targets()
    if not api_targets:
        print(
            "No API targets enabled (set OPENAI_API_KEY / PERSPECTIVE_API_KEY "
            "env vars AND USE_API_TARGETS = True in config.py). This script optimizes DIRECTLY against black-box "
            "APIs and has nothing to attack without them -- see "
            "transfer_study.py for the surrogate-transfer alternative that "
            "doesn't need API keys to run against the local classifier."
        )
        return

    if not args.accept:
        print(
            "This script queries live, third-party production moderation "
            "APIs. Before running it for real, read the responsible-use "
            "section in this file's docstring (query cap, no live-content "
            "targeting, responsible disclosure). Re-run with "
            "--i-accept-responsible-use once you've done so."
        )
        return

    data = load_dataset(n_samples=args.n_samples)
    sim_fn = SemanticSimilarity()

    all_results = {}
    for name, target in api_targets.items():
        print(f"\n=== Direct black-box genetic attack against '{name}' "
              f"(query cap: {config.MAX_TOTAL_API_QUERIES_PER_RUN}) ===")
        guard = QueryBudgetGuard(target.score, cap=config.MAX_TOTAL_API_QUERIES_PER_RUN)

        seed_asrs = []
        per_seed_records = []
        for seed in range(args.n_seeds):
            if guard.count >= guard.cap:
                print(f"  [budget] Query cap reached before seed {seed}; stopping early.")
                break
            print(f"\n  --- seed {seed} (queries used so far: {guard.count}/{guard.cap}) ---")
            results = run_direct_blackbox_attack(data, guard, sim_fn, args.max_queries, seed=seed)
            n = len(results)
            asr = sum(r["success"] for r in results) / n if n else None
            if asr is not None:
                seed_asrs.append(asr)
            per_seed_records.append({"seed": seed, "n_attacked": n, "attack_success_rate": asr, "records": results})

        asr_summary = seed_variance_summary(seed_asrs)
        all_results[name] = {
            "queries_used_total": guard.count,
            "query_cap": guard.cap,
            "asr_across_seeds": asr_summary,
            "per_seed": per_seed_records,
        }
        if asr_summary["mean"] is not None:
            print(f"\n'{name}' direct-attack ASR across {asr_summary['n_seeds']} seeds: "
                  f"mean={asr_summary['mean']:.2%}  std={asr_summary['std']:.2%}  "
                  f"min={asr_summary['min']:.2%}  max={asr_summary['max']:.2%}")
        else:
            print(f"'{name}': no flagged examples found across any seed")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, "blackbox_direct_attack_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {out_path}")
    print(
        "\nCompare asr_across_seeds.mean against transfer_study.py's "
        "transfer_rate for the same API: a meaningfully higher direct-attack "
        "ASR shows that surrogate-transfer alone understated how vulnerable "
        "the production API actually is to a determined black-box adversary. "
        "A high std/wide min-max means the result is seed-sensitive and "
        "should be reported as a range, not a single number."
    )


if __name__ == "__main__":
    main()
