import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from targets.local_classifier import MockClassifier
from attacks.char_attacks import (
    homoglyph_substitute, leetspeak_substitute, insert_zero_width,
    random_typo, greedy_search_attack,
)
from attacks.word_substitution_attack import word_substitution_attack, _get_synonyms
from attacks.trigger_search_attack import search_universal_trigger, evaluate_trigger_transfer
from attacks.paraphrase_attack import rule_based_paraphrase, paraphrase_attack, HFParaphraser


def permissive_sim(a, b):
    return 1.0


def test_homoglyph_substitute_changes_text_with_high_rate():
    text = "hate is bad"
    rng = random.Random(1)
    out = homoglyph_substitute(text, rate=1.0, rng=rng)
    assert out != text
    assert len(out) == len(text)  # 1:1 char substitution


def test_leetspeak_substitute_deterministic_with_seed():
    text = "e a t o s"
    rng = random.Random(2)
    out = leetspeak_substitute(text, rate=1.0, rng=rng)
    assert out != text


def test_insert_zero_width_increases_length():
    text = "hello world"
    rng = random.Random(3)
    out = insert_zero_width(text, rate=1.0, rng=rng)
    assert len(out) > len(text)


def test_random_typo_can_modify_text():
    text = "this is a reasonably long test sentence"
    rng = random.Random(4)
    out = random_typo(text, rate=0.5, rng=rng)
    assert isinstance(out, str)


def test_greedy_search_attack_reduces_score_on_flagged_text():
    clf = MockClassifier()
    text = "You are an idiot and everyone hates you."
    assert clf.score(text) >= 0.5

    result = greedy_search_attack(
        text, score_fn=clf.score, similarity_fn=permissive_sim,
        max_queries=40, min_similarity=0.0, seed=1,
    )
    assert result["orig_score"] >= 0.5
    assert result["best_score"] <= result["orig_score"]
    assert result["success"] is True
    assert result["queries_used"] > 0


def test_greedy_search_attack_noop_on_already_benign_text():
    clf = MockClassifier()
    text = "Thanks so much for your help with the report."
    result = greedy_search_attack(
        text, score_fn=clf.score, similarity_fn=permissive_sim,
        max_queries=40, min_similarity=0.0, seed=1,
    )
    assert result["orig_score"] < 0.5
    assert result["success"] is False  # can't "succeed" attacking non-flagged text


def test_word_substitution_attack_runs_and_returns_expected_keys():
    clf = MockClassifier()
    text = "You are an idiot and everyone hates you."
    result = word_substitution_attack(
        text, score_fn=clf.score, similarity_fn=permissive_sim,
        max_queries=60, min_similarity=0.0,
    )
    for key in ("best_text", "best_score", "orig_score", "queries_used", "success", "similarity"):
        assert key in result


def test_get_synonyms_fallback_returns_known_word():
    syns = _get_synonyms("idiot")
    assert isinstance(syns, list)
    assert len(syns) > 0


def test_rule_based_paraphrase_returns_candidates():
    candidates = rule_based_paraphrase("this is bad and stupid", n_candidates=5, rng=random.Random(0))
    assert isinstance(candidates, list)
    assert len(candidates) > 0


def test_paraphrase_attack_with_rule_based_fallback():
    clf = MockClassifier()
    text = "You are an idiot and everyone hates you."

    class DummyParaphraser:
        def paraphrase(self, text, n_candidates=5):
            return rule_based_paraphrase(text, n_candidates=n_candidates, rng=random.Random(0))

    result = paraphrase_attack(
        text, score_fn=clf.score, similarity_fn=permissive_sim,
        paraphraser=DummyParaphraser(), n_candidates=5, min_similarity=0.0,
    )
    for key in ("best_text", "best_score", "orig_score", "queries_used", "success", "similarity"):
        assert key in result


def test_universal_trigger_search_reduces_average_score():
    clf = MockClassifier()
    texts = [
        "You are an idiot.",
        "I hate you so much.",
        "This is the worst, so stupid.",
    ]
    result = search_universal_trigger(
        texts, score_fn=clf.score, beam_width=3, max_trigger_tokens=2, max_queries=200, seed=0,
    )
    assert "trigger" in result
    assert result["avg_score_with_trigger"] <= result["avg_score_without_trigger"]


def test_evaluate_trigger_transfer_returns_expected_keys():
    clf = MockClassifier()
    holdout = ["I hate this, disgusting.", "You are so stupid, shut up."]
    result = evaluate_trigger_transfer("by the way", holdout, clf.score)
    assert "holdout_success_rate" in result
    assert result["n_holdout"] == len(holdout)
