import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import (
    bootstrap_asr_ci, mcnemar_test, compare_attacks, holm_bonferroni_correction,
    compare_attacks_across_models, paired_asr_difference_ci, seed_variance_summary,
)


def test_paired_asr_difference_ci_identical_sequences_gives_zero_diff():
    a = [True, False, True, False]
    b = [True, False, True, False]
    result = paired_asr_difference_ci(a, b, n_iterations=200, seed=0)
    assert result["asr_difference"] == 0.0


def test_paired_asr_difference_ci_detects_positive_difference():
    a = [True, True, True, False]
    b = [False, False, True, False]
    result = paired_asr_difference_ci(a, b, n_iterations=500, seed=0)
    assert result["asr_difference"] > 0
    assert result["ci_low"] <= result["asr_difference"] <= result["ci_high"]


def test_paired_asr_difference_ci_empty():
    result = paired_asr_difference_ci([], [], n_iterations=100)
    assert result["asr_difference"] is None
    assert result["n"] == 0


def test_seed_variance_summary_basic():
    result = seed_variance_summary([0.2, 0.4, 0.3])
    assert result["n_seeds"] == 3
    assert result["min"] == 0.2
    assert result["max"] == 0.4
    assert abs(result["mean"] - 0.3) < 1e-6


def test_seed_variance_summary_empty():
    result = seed_variance_summary([])
    assert result["n_seeds"] == 0
    assert result["mean"] is None


def test_mcnemar_test_includes_odds_ratio():
    a = [True] * 10 + [False] * 5
    b = [False] * 10 + [False] * 5
    result = mcnemar_test(a, b)
    assert "odds_ratio" in result
    assert result["odds_ratio"] > 1  # a succeeds where b fails, not vice versa


def test_mcnemar_test_odds_ratio_no_discordant_pairs():
    a = [True, False]
    b = [True, False]
    result = mcnemar_test(a, b)
    assert result["odds_ratio"] == 1.0


def test_compare_attacks_includes_effect_size():
    records = [
        {"attack": "char", "example_idx": 0, "success": True},
        {"attack": "char", "example_idx": 1, "success": True},
        {"attack": "word", "example_idx": 0, "success": False},
        {"attack": "word", "example_idx": 1, "success": False},
    ]
    result = compare_attacks(records)
    comp = result["pairwise_mcnemar"]["char_vs_word"]
    assert "effect_size" in comp
    assert "asr_difference" in comp["effect_size"]


def test_holm_bonferroni_no_correction_needed_single_test():
    result = holm_bonferroni_correction([0.03])
    assert result[0]["p_value_holm_adjusted"] == 0.03
    assert result[0]["significant_after_correction"] is True


def test_holm_bonferroni_corrects_multiple_comparisons():
    # 3 identical borderline p-values: uncorrected all "significant" at 0.05,
    # but Holm should push at least the smallest-rank one above 0.05
    # (0.04 * 3 = 0.12 for the most significant one under Holm).
    result = holm_bonferroni_correction([0.04, 0.04, 0.04])
    assert all(r["p_value_holm_adjusted"] > 0.05 for r in result)
    assert all(r["significant_after_correction"] is False for r in result)


def test_holm_bonferroni_preserves_order_and_length():
    pvals = [0.5, 0.001, 0.2]
    result = holm_bonferroni_correction(pvals)
    assert len(result) == len(pvals)
    for i, r in enumerate(result):
        assert r["p_value"] == pvals[i]


def test_holm_bonferroni_empty_input():
    assert holm_bonferroni_correction([]) == []


def test_compare_attacks_applies_correction_fields():
    records = [
        {"attack": "char", "example_idx": 0, "success": True},
        {"attack": "char", "example_idx": 1, "success": True},
        {"attack": "word", "example_idx": 0, "success": False},
        {"attack": "word", "example_idx": 1, "success": False},
    ]
    result = compare_attacks(records)
    comp = result["pairwise_mcnemar"]["char_vs_word"]
    assert "p_value_holm_adjusted" in comp
    assert "significant_after_correction" in comp
    assert "significant_uncorrected" in comp


def test_compare_attacks_across_models_corrects_full_family():
    records_by_model = {
        "model_a": [
            {"attack": "char", "example_idx": 0, "success": True},
            {"attack": "char", "example_idx": 1, "success": False},
            {"attack": "word", "example_idx": 0, "success": False},
            {"attack": "word", "example_idx": 1, "success": False},
        ],
        "model_b": [
            {"attack": "char", "example_idx": 0, "success": True},
            {"attack": "char", "example_idx": 1, "success": True},
            {"attack": "word", "example_idx": 0, "success": True},
            {"attack": "word", "example_idx": 1, "success": False},
        ],
    }
    result = compare_attacks_across_models(records_by_model)
    assert result["n_tests_in_family"] == 2  # one char_vs_word test per model
    assert "model_a::char_vs_word" in result["corrected_across_full_family"]
    assert "model_b::char_vs_word" in result["corrected_across_full_family"]


def test_bootstrap_asr_ci_all_successes():
    result = bootstrap_asr_ci([True] * 50, n_iterations=200, seed=0)
    assert result["asr"] == 1.0
    assert result["ci_low"] <= 1.0 <= result["ci_high"] + 1e-9


def test_bootstrap_asr_ci_all_failures():
    result = bootstrap_asr_ci([False] * 50, n_iterations=200, seed=0)
    assert result["asr"] == 0.0


def test_bootstrap_asr_ci_empty():
    result = bootstrap_asr_ci([], n_iterations=200)
    assert result["asr"] is None
    assert result["n"] == 0


def test_bootstrap_asr_ci_mixed_produces_reasonable_interval():
    successes = [True, False] * 25
    result = bootstrap_asr_ci(successes, n_iterations=500, seed=1)
    assert 0.3 < result["asr"] < 0.7
    assert result["ci_low"] < result["asr"] < result["ci_high"]


def test_mcnemar_no_discordant_pairs():
    a = [True, False, True]
    b = [True, False, True]
    result = mcnemar_test(a, b)
    assert result["p_value"] == 1.0


def test_mcnemar_detects_difference_with_many_discordant_pairs():
    # A succeeds on 40 examples B fails on; B succeeds on 2 examples A fails on
    a = [True] * 40 + [False] * 2 + [True] * 10
    b = [False] * 40 + [True] * 2 + [True] * 10
    result = mcnemar_test(a, b)
    assert result["p_value"] < 0.05
    assert result["significant_at_0.05"] is True


def test_compare_attacks_groups_by_attack_and_example():
    records = [
        {"attack": "char", "example_idx": 0, "success": True},
        {"attack": "char", "example_idx": 1, "success": False},
        {"attack": "word", "example_idx": 0, "success": False},
        {"attack": "word", "example_idx": 1, "success": False},
    ]
    result = compare_attacks(records)
    assert "char" in result["bootstrap_ci_by_attack"]
    assert "word" in result["bootstrap_ci_by_attack"]
    assert "char_vs_word" in result["pairwise_mcnemar"]
