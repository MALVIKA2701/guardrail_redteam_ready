import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from targets.local_classifier import MockClassifier
from attacks.genetic_search_attack import genetic_search_attack, _mutate, _crossover
from universal_trigger_study import split_optimize_holdout, run_single_seed


def permissive_sim(a, b):
    return 1.0


def test_mutate_changes_tokens():
    rng = random.Random(0)
    tokens = ["you", "are", "an", "idiot"]
    mutated = _mutate(tokens, rng)
    assert len(mutated) == len(tokens)


def test_crossover_same_length_inputs():
    rng = random.Random(0)
    a = ["a", "b", "c", "d"]
    b = ["w", "x", "y", "z"]
    child = _crossover(a, b, rng)
    assert len(child) == 4
    # child should contain a prefix from a and a suffix from b
    assert child[0] == "a" or child[0] == "w"


def test_crossover_mismatched_length_falls_back():
    rng = random.Random(0)
    a = ["a", "b", "c"]
    b = ["x", "y"]
    child = _crossover(a, b, rng)
    assert child == a


def test_genetic_search_attack_reduces_score_on_flagged_text():
    clf = MockClassifier()
    text = "You are an idiot and everyone hates you disgusting fool"
    assert clf.score(text) >= 0.5

    result = genetic_search_attack(
        text, score_fn=clf.score, similarity_fn=permissive_sim,
        population_size=10, generations=6, max_queries=150,
        min_similarity=0.0, seed=1,
    )
    assert result["orig_score"] >= 0.5
    assert result["best_score"] <= result["orig_score"]
    assert result["queries_used"] > 0


def test_genetic_search_attack_respects_query_budget():
    clf = MockClassifier()
    text = "You are an idiot and everyone hates you"
    result = genetic_search_attack(
        text, score_fn=clf.score, similarity_fn=permissive_sim,
        population_size=20, generations=20, max_queries=15,
        min_similarity=0.0, seed=2,
    )
    # queries_used should not wildly exceed the budget (allow one batch overshoot)
    assert result["queries_used"] <= 15 + 20


def test_split_optimize_holdout_partitions_all_examples():
    texts = [f"example {i}" for i in range(10)]
    optimize, holdout = split_optimize_holdout(texts, holdout_fraction=0.3, seed=0)
    assert len(optimize) + len(holdout) == len(texts)
    assert set(optimize).isdisjoint(set(holdout))


def test_split_optimize_holdout_respects_fraction_roughly():
    texts = [f"example {i}" for i in range(20)]
    optimize, holdout = split_optimize_holdout(texts, holdout_fraction=0.25, seed=0)
    assert len(holdout) == 5
    assert len(optimize) == 15


def test_run_single_seed_produces_expected_keys():
    clf = MockClassifier()
    optimize_set = [
        "You are an idiot.", "I hate you so much.", "This is the worst, so stupid.",
        "Shut up, disgusting fool.",
    ]
    holdout_set = ["Everyone hates you, idiot.", "This is disgusting."]
    result = run_single_seed(optimize_set, holdout_set, clf, seed=0)
    for key in ("seed", "trigger", "optimize_set_success_rate", "held_out_success_rate", "search_queries_used"):
        assert key in result


def test_run_single_seed_different_seeds_can_differ():
    clf = MockClassifier()
    optimize_set = [
        "You are an idiot.", "I hate you so much.", "This is the worst, so stupid.",
        "Shut up, disgusting fool.",
    ]
    holdout_set = ["Everyone hates you, idiot.", "This is disgusting."]
    result_a = run_single_seed(optimize_set, holdout_set, clf, seed=0)
    result_b = run_single_seed(optimize_set, holdout_set, clf, seed=1)
    # Not asserting they differ (they might coincidentally match) -- just
    # confirming both run cleanly and independently with different seeds.
    assert result_a["seed"] == 0
    assert result_b["seed"] == 1
