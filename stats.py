"""
Statistical rigor layer: bootstrap confidence intervals on Attack Success
Rate, and paired significance testing (McNemar's test) between attack
methods, so results are reported as "X% [95% CI: a-b]" and "attack A beats
attack B, p=..." rather than bare point estimates.

Usage:
    python stats.py --results results/attack_results.jsonl
"""

import argparse
import json
import math
import random
from collections import defaultdict
from typing import List

import config


def bootstrap_asr_ci(
    successes: List[bool],
    n_iterations: int = None,
    ci: float = None,
    seed: int = 0,
) -> dict:
    """
    Bootstrap confidence interval on Attack Success Rate.

    Resamples `successes` (a list of booleans) with replacement n_iterations
    times, computes ASR each time, and returns the point estimate plus the
    percentile CI.
    """
    n_iterations = n_iterations or config.BOOTSTRAP_ITERATIONS
    ci = ci or config.BOOTSTRAP_CI
    rng = random.Random(seed)
    n = len(successes)
    if n == 0:
        return {"asr": None, "ci_low": None, "ci_high": None, "n": 0}

    point_estimate = sum(successes) / n
    boot_asrs = []
    for _ in range(n_iterations):
        sample = [successes[rng.randrange(n)] for _ in range(n)]
        boot_asrs.append(sum(sample) / n)
    boot_asrs.sort()
    alpha = (1 - ci) / 2
    lo_idx = int(alpha * n_iterations)
    hi_idx = int((1 - alpha) * n_iterations) - 1
    return {
        "asr": round(point_estimate, 4),
        "ci_low": round(boot_asrs[lo_idx], 4),
        "ci_high": round(boot_asrs[hi_idx], 4),
        "n": n,
        "confidence_level": ci,
    }


def holm_bonferroni_correction(p_values: List[float], alpha: float = None) -> List[dict]:
    """
    Holm-Bonferroni step-down correction for family-wise error rate control.

    More powerful than plain Bonferroni while still controlling FWER exactly
    (no independence assumption needed, unlike Benjamini-Hochberg's FDR
    control). Appropriate here because we want to avoid ANY false positive
    across the whole family of pairwise attack-comparison tests, not just
    control the expected proportion of false positives.

    Returns a list (same order as input) of dicts with the original p-value,
    the Holm-adjusted p-value, and whether it's significant after correction.
    """
    alpha = alpha or config.SIGNIFICANCE_ALPHA
    m = len(p_values)
    if m == 0:
        return []

    # Sort by p-value ascending, keeping track of original index
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [None] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        # Holm adjustment: p * (m - rank), enforced monotonically non-decreasing
        candidate = p * (m - rank)
        running_max = max(running_max, candidate)
        adjusted[orig_idx] = min(running_max, 1.0)

    return [
        {
            "p_value": round(p_values[i], 6),
            "p_value_holm_adjusted": round(adjusted[i], 6),
            "significant_after_correction": adjusted[i] < alpha,
        }
        for i in range(m)
    ]


def paired_asr_difference_ci(
    successes_a: List[bool],
    successes_b: List[bool],
    n_iterations: int = None,
    ci: float = None,
    seed: int = 0,
) -> dict:
    """
    Effect size for two paired attack-success sequences: the ASR difference
    (attack A's ASR minus attack B's ASR) with its own bootstrap CI, computed
    by resampling EXAMPLE PAIRS with replacement (preserving the pairing)
    rather than resampling each attack's successes independently.

    A significant McNemar p-value only tells you the two attacks differ;
    this tells you how much, which is what a reviewer actually needs to
    judge whether the difference is practically meaningful.
    """
    n_iterations = n_iterations or config.BOOTSTRAP_ITERATIONS
    ci = ci or config.BOOTSTRAP_CI
    assert len(successes_a) == len(successes_b)
    n = len(successes_a)
    if n == 0:
        return {"asr_difference": None, "ci_low": None, "ci_high": None, "n": 0}

    point = sum(successes_a) / n - sum(successes_b) / n
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        a = sum(successes_a[i] for i in idx) / n
        b = sum(successes_b[i] for i in idx) / n
        diffs.append(a - b)
    diffs.sort()
    alpha = (1 - ci) / 2
    lo = diffs[int(alpha * n_iterations)]
    hi = diffs[int((1 - alpha) * n_iterations) - 1]
    return {
        "asr_difference": round(point, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "n": n,
        "confidence_level": ci,
    }


def seed_variance_summary(values: List[float]) -> dict:
    """
    Summarizes a metric (e.g. ASR) computed across multiple random seeds of
    a stochastic search procedure, so a single lucky/unlucky run isn't
    mistaken for a stable result.
    """
    n = len(values)
    if n == 0:
        return {"n_seeds": 0, "mean": None, "std": None, "min": None, "max": None, "values": []}
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance ** 0.5
    return {
        "n_seeds": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "values": [round(v, 4) for v in values],
    }


def mcnemar_test(successes_a: List[bool], successes_b: List[bool]) -> dict:
    """
    McNemar's test for paired binary outcomes -- the standard test for
    "does attack A succeed on significantly more of the SAME examples than
    attack B", accounting for the pairing (same underlying example) rather
    than treating the two attacks' results as independent samples.

    Requires successes_a and successes_b to be aligned (same example order,
    same length).
    """
    assert len(successes_a) == len(successes_b), "Paired lists must be same length"
    b = sum(1 for a, bb in zip(successes_a, successes_b) if a and not bb)  # A succeeds, B fails
    c = sum(1 for a, bb in zip(successes_a, successes_b) if not a and bb)  # B succeeds, A fails

    if b + c == 0:
        return {
            "statistic": 0.0, "p_value": 1.0, "b_a_only": b, "c_b_only": c,
            "significant_at_0.05": False, "odds_ratio": 1.0, "note": "no discordant pairs",
        }

    # Continuity-corrected McNemar's chi-square statistic (chi2, df=1)
    statistic = ((abs(b - c) - 1) ** 2) / (b + c)
    p_value = _chi2_sf_df1(statistic)
    # Odds ratio (Haldane-Anscombe +0.5 correction to avoid division by zero
    # when b or c is 0) -- the standard effect size for McNemar's test,
    # answering "how much more often does A succeed where B fails, versus
    # the reverse" rather than just "is the difference nonzero."
    odds_ratio = (b + 0.5) / (c + 0.5)
    return {
        "statistic": round(statistic, 4),
        "p_value": round(p_value, 6),
        "b_a_only": b,
        "c_b_only": c,
        "odds_ratio": round(odds_ratio, 4),
        "significant_at_0.05": p_value < config.SIGNIFICANCE_ALPHA,
    }


def _chi2_sf_df1(x: float) -> float:
    """Survival function of chi-square distribution with 1 df, i.e. P(X > x).
    Equivalent to 2 * (1 - Phi(sqrt(x))) using the standard normal CDF Phi,
    since chi2(df=1) = Z^2 for standard normal Z. Implemented without scipy
    so this module has zero extra dependencies."""
    if x <= 0:
        return 1.0
    z = math.sqrt(x)
    # 1 - Phi(z) via the complementary error function identity
    upper_tail = 0.5 * math.erfc(z / math.sqrt(2))
    return 2 * upper_tail


def compare_attacks(records: List[dict]) -> dict:
    """
    Groups records by example_idx and attack, then reports bootstrap CIs per
    attack plus pairwise McNemar tests between every pair of attacks that
    share the same set of example_idx values. Applies Holm-Bonferroni
    correction across ALL pairwise tests run here, since running k pairwise
    comparisons without correction inflates the family-wise false-positive
    rate -- e.g. with 3 attacks (3 pairwise tests) at alpha=0.05 uncorrected,
    ~14% chance of at least one spurious "significant" result even if no
    real difference exists anywhere.
    """
    by_attack = defaultdict(dict)  # attack -> {example_idx: success}
    for r in records:
        by_attack[r["attack"]][r["example_idx"]] = r["success"]

    ci_results = {}
    for attack, per_example in by_attack.items():
        ci_results[attack] = bootstrap_asr_ci(list(per_example.values()))

    pairwise = {}
    attack_names = sorted(by_attack.keys())
    comparison_names = []
    raw_p_values = []
    for i, a in enumerate(attack_names):
        for b in attack_names[i + 1:]:
            shared_idx = sorted(set(by_attack[a]) & set(by_attack[b]))
            if not shared_idx:
                continue
            succ_a = [by_attack[a][idx] for idx in shared_idx]
            succ_b = [by_attack[b][idx] for idx in shared_idx]
            result = mcnemar_test(succ_a, succ_b)
            result["effect_size"] = paired_asr_difference_ci(succ_a, succ_b)
            name = f"{a}_vs_{b}"
            pairwise[name] = result
            comparison_names.append(name)
            raw_p_values.append(result["p_value"])

    if raw_p_values:
        corrections = holm_bonferroni_correction(raw_p_values)
        for name, corr in zip(comparison_names, corrections):
            pairwise[name]["p_value_holm_adjusted"] = corr["p_value_holm_adjusted"]
            pairwise[name]["significant_after_correction"] = corr["significant_after_correction"]
            # Replace the uncorrected flag with an explicit note so results
            # can't be read as already-corrected by mistake.
            pairwise[name]["significant_uncorrected"] = pairwise[name].pop("significant_at_0.05")

    return {"bootstrap_ci_by_attack": ci_results, "pairwise_mcnemar": pairwise}


def compare_attacks_across_models(records_by_model: dict) -> dict:
    """
    Same idea as compare_attacks, but for the multi-model study: runs
    pairwise McNemar tests within EACH model, then applies ONE Holm-Bonferroni
    correction across the FULL family of (model x attack-pair) tests
    together. This is the correct scope for the correction -- if you instead
    corrected separately per model, you'd still inflate the overall
    false-positive rate across the whole study.

    Args:
        records_by_model: {model_name: [attack_records, ...]}

    Returns:
        dict: {"per_model": {model_name: {attack_pair: mcnemar_result_without_correction}},
               "corrected": {"model_name::attack_pair": corrected_result}}
    """
    per_model_raw = {}
    all_names = []
    all_p_values = []

    for model_name, records in records_by_model.items():
        by_attack = defaultdict(dict)
        for r in records:
            by_attack[r["attack"]][r["example_idx"]] = r["success"]

        attack_names = sorted(by_attack.keys())
        model_results = {}
        for i, a in enumerate(attack_names):
            for b in attack_names[i + 1:]:
                shared_idx = sorted(set(by_attack[a]) & set(by_attack[b]))
                if not shared_idx:
                    continue
                succ_a = [by_attack[a][idx] for idx in shared_idx]
                succ_b = [by_attack[b][idx] for idx in shared_idx]
                result = mcnemar_test(succ_a, succ_b)
                result["effect_size"] = paired_asr_difference_ci(succ_a, succ_b)
                pair_name = f"{a}_vs_{b}"
                model_results[pair_name] = result
                all_names.append(f"{model_name}::{pair_name}")
                all_p_values.append(result["p_value"])
        per_model_raw[model_name] = model_results

    corrected = {}
    if all_p_values:
        corrections = holm_bonferroni_correction(all_p_values)
        for name, corr in zip(all_names, corrections):
            corrected[name] = corr

    return {
        "per_model_raw": per_model_raw,
        "corrected_across_full_family": corrected,
        "n_tests_in_family": len(all_p_values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=config.ATTACK_RESULTS_PATH)
    args = parser.parse_args()

    with open(args.results) as f:
        records = [json.loads(line) for line in f]

    results = compare_attacks(records)
    print(json.dumps(results, indent=2))

    import os
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(config.STATS_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved statistical analysis to {config.STATS_RESULTS_PATH}")


if __name__ == "__main__":
    main()
