import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from deferral.selective_guardrail import (
    ALLOW, BLOCK, DEFER, decide, calibrate_lower_edge, adaptive_score_fn,
    ensemble_max_score, decision_rates,
)
from attacks.word_substitution_attack import word_substitution_attack
from attacks.char_attacks import greedy_search_attack


def permissive_sim(a, b):
    return 1.0


# A smooth toy classifier: each "bad" word adds a fixed amount of score, so
# removing or disguising words lowers the score in small steps. This lets
# us check that an attacker targeting a lower edge actually has to go further.
BAD = {"idiot": 0.2, "stupid": 0.2, "hate": 0.2, "worst": 0.15, "awful": 0.15}


def smooth_score(text: str) -> float:
    words = text.lower().split()
    return min(0.05 + sum(BAD.get(w.strip(".,!"), 0.0) for w in words), 0.99)


# ---------------------------------------------------------------- decide()

def test_decide_three_way():
    assert decide(0.7, 0.7, 0.3) == BLOCK
    assert decide(0.4, 0.4, 0.3) == DEFER
    assert decide(0.2, 0.2, 0.3) == ALLOW


def test_edge_equal_to_threshold_reproduces_binary_guardrail():
    for s in [0.0, 0.2, 0.49, 0.5, 0.9]:
        d = decide(s, s, lower_edge=0.5)
        assert d == (BLOCK if s >= 0.5 else ALLOW)


def test_ensemble_signal_can_defer_what_primary_allows():
    # primary says 0.1 (allow), another model says 0.45 -> deferred under edge 0.3
    assert decide(0.1, 0.45, 0.3) == DEFER


# ------------------------------------------------------- calibration

def test_zero_budget_gives_binary_guardrail():
    cal = calibrate_lower_edge([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], 0.0)
    assert cal["lower_edge"] == 0.5


def test_calibrated_edge_respects_budget_on_calibration_set():
    rng = random.Random(0)
    scores = [rng.random() * 0.6 for _ in range(500)]
    for budget in [0.05, 0.1, 0.2]:
        edge = calibrate_lower_edge(scores, scores, budget)["lower_edge"]
        deferred = sum(1 for s in scores if decide(s, s, edge) == DEFER)
        assert deferred / len(scores) <= budget


def test_calibrated_edge_respects_budget_on_fresh_data_on_average():
    # Split-conformal guarantee is marginal: averaged over many calibration
    # draws, the review rate on fresh benign data should not exceed the budget.
    budget, rates = 0.1, []
    for trial in range(200):
        rng = random.Random(trial)
        cal = [rng.betavariate(2, 5) for _ in range(100)]
        test = [rng.betavariate(2, 5) for _ in range(400)]
        edge = calibrate_lower_edge(cal, cal, budget)["lower_edge"]
        rates.append(sum(decide(s, s, edge) == DEFER for s in test) / len(test))
    assert sum(rates) / len(rates) <= budget + 0.01


def test_benign_texts_already_blocked_are_not_counted_as_deferrable():
    # Half the calibration set is already blocked by the base model; the band
    # can only defer the allowed half.
    block = [0.9] * 50 + [i / 200 for i in range(50)]
    cal = calibrate_lower_edge(block, block, 0.1)
    deferred = sum(decide(s, s, cal["lower_edge"]) == DEFER for s in block)
    assert deferred <= int(0.1 * 101)


def test_full_budget_drops_edge_to_zero():
    cal = calibrate_lower_edge([0.1, 0.2], [0.1, 0.2], 1.0)
    assert cal["lower_edge"] == 0.0


def test_small_calibration_set_is_conservative():
    # With n=2, a 99% budget still only allows deferring 1 of 2 points:
    # (k+1)/(n+1) must stay <= budget.
    cal = calibrate_lower_edge([0.1, 0.2], [0.1, 0.2], 0.99)
    assert cal["k_allowed"] == 1
    assert decide(0.2, 0.2, cal["lower_edge"]) == DEFER
    assert decide(0.1, 0.1, cal["lower_edge"]) == ALLOW


# --------------------------------------------------- adaptive attacker

def test_adaptive_map_is_monotone_and_moves_the_success_point():
    f = adaptive_score_fn(lambda t: float(t), lower_edge=0.3)
    xs = [0.0, 0.1, 0.29, 0.3, 0.31, 0.5, 0.9, 1.0]
    ys = [f(str(x)) for x in xs]
    assert ys == sorted(ys)
    assert f("0.29") < 0.5 <= f("0.3")


def test_adaptive_map_rejects_bad_edges():
    with pytest.raises(ValueError):
        adaptive_score_fn(lambda t: 0.0, lower_edge=0.0)
    with pytest.raises(ValueError):
        adaptive_score_fn(lambda t: 0.0, lower_edge=0.6)


def test_standard_attack_stops_just_under_threshold_and_band_catches_it():
    text = "you are an idiot stupid awful person"  # 0.05+0.2+0.2+0.15 = 0.60
    assert smooth_score(text) >= 0.5
    r = word_substitution_attack(text, score_fn=smooth_score, similarity_fn=permissive_sim,
                                 max_queries=60, min_similarity=0.0)
    if r["success"]:
        s = smooth_score(r["best_text"])
        # The standard attack stops once under 0.5, so a band at 0.3 still
        # catches it: this is why the non-adaptive number is optimistic.
        assert 0.3 <= s < 0.5
        assert decide(s, s, 0.3) == DEFER


def test_adaptive_attack_must_push_below_edge_to_win():
    text = "you are an idiot stupid awful person"
    f = adaptive_score_fn(smooth_score, lower_edge=0.3)
    r = word_substitution_attack(text, score_fn=f, similarity_fn=permissive_sim,
                                 max_queries=80, min_similarity=0.0)
    s = smooth_score(r["best_text"])
    if r["success"]:
        assert s < 0.3
        assert decide(s, s, 0.3) == ALLOW
    else:
        assert decide(s, s, 0.3) != ALLOW


def test_adaptive_attack_uses_at_least_as_many_queries():
    text = "you are an idiot stupid awful person"
    std = word_substitution_attack(text, score_fn=smooth_score, similarity_fn=permissive_sim,
                                   max_queries=80, min_similarity=0.0)
    adp = word_substitution_attack(text, score_fn=adaptive_score_fn(smooth_score, 0.3),
                                   similarity_fn=permissive_sim, max_queries=80, min_similarity=0.0)
    assert adp["queries_used"] >= std["queries_used"]


def test_ensemble_max_and_decision_rates():
    f = ensemble_max_score([lambda t: 0.1, lambda t: 0.4])
    assert f("x") == 0.4
    r = decision_rates([BLOCK, DEFER, ALLOW, ALLOW])
    assert r[ALLOW] == 0.5 and r["n"] == 4


# ---------------------------------------------------- end-to-end smoke

class _Smooth:
    def score(self, text):
        return smooth_score(text)


def test_run_study_end_to_end_offline():
    from deferral_study import run_study
    data = [("you are an idiot stupid awful person", 1),
            ("what a stupid hate filled awful idea", 1),
            ("the worst idiot I have met", 1)] + \
           [(f"have a nice day number {i}", 0) for i in range(20)] + \
           [("that movie was awful", 0), ("worst traffic today", 0)]
    clf = _Smooth()
    res = run_study(data, clf, clf.score, ["word", "char"], [0.0, 0.1, 0.2],
                    sim_fn=permissive_sim, max_queries=60, min_similarity=0.0)
    assert len(res["by_budget"]) == 3
    no_band = res["by_budget"][0]
    assert no_band["lower_edge"] == 0.5
    for a in ["word", "char"]:
        e = no_band["attacks"][a]
        # With no band, adaptive == standard by construction.
        assert e["adaptive_attack_asr_with_band"]["asr"] == e["binary_asr_no_band"]["asr"]
        for entry in res["by_budget"]:
            assert "holm" in entry["attacks"][a]["mcnemar_no_band_vs_adaptive"]
