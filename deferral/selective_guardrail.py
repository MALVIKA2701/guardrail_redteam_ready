"""
Selective guardrail: a three-way moderation policy that can DEFER to a human.

The base guardrail in this project makes a binary decision: block if the
classifier's toxicity score is >= the threshold (0.5), allow otherwise. An
evasion attack "succeeds" the moment it pushes the score just under 0.5.

This module asks a different question: instead of only blocking or allowing,
can the guardrail recognise inputs it is unsure about and send them to human
review? The policy is

    score >= block_threshold                     -> BLOCK
    deferral_score >= lower_edge (and not BLOCK) -> DEFER  (human review)
    otherwise                                    -> ALLOW

`deferral_score` is the primary classifier's own score by default (a
"confidence band" just below the threshold), or the maximum score across an
ensemble of classifiers ("defer if ANY model is suspicious").

Two design points matter for an honest evaluation:

1. The lower edge is calibrated on held-out BENIGN text so that a chosen
   human-review budget (e.g. at most 10% of benign traffic sent to review)
   holds on new data, using a split-conformal quantile (see
   `calibrate_lower_edge`).

2. Several attacks in this repo stop as soon as the score falls below 0.5,
   so their successful examples sit just under the threshold by
   construction. A band below 0.5 would "catch" them for that reason alone.
   The real test is an ADAPTIVE attacker who knows the band exists and keeps
   going until the deferral score drops below the lower edge. That attacker
   is built here without modifying any attack code, by handing the attacks
   a monotone re-scaling of the score (`adaptive_score_fn`).

Everything in this module is pure Python (no numpy/torch), so it runs in
the offline test suite.
"""

import math
from typing import Callable, Dict, List, Sequence

BLOCK = "block"
DEFER = "defer"
ALLOW = "allow"


def decide(score: float, deferral_score: float, lower_edge: float,
           block_threshold: float = 0.5) -> str:
    """Three-way decision for a single input."""
    if score >= block_threshold:
        return BLOCK
    if deferral_score >= lower_edge:
        return DEFER
    return ALLOW


def calibrate_lower_edge(benign_deferral_scores: Sequence[float],
                         benign_block_scores: Sequence[float],
                         review_budget: float,
                         block_threshold: float = 0.5) -> dict:
    """
    Chooses the band's lower edge from a CALIBRATION set of benign texts so
    that the probability a new benign text is sent to human review is at
    most `review_budget`, under exchangeability (split-conformal argument).

    Only benign texts the base guardrail would ALLOW (block score below the
    threshold) can be newly deferred; benign texts it already blocks are
    false positives of the base model and are unaffected by the band.

    Let n be the number of calibration benign texts and k the number of
    them the band is allowed to defer. With k = floor(review_budget * (n + 1))
    - 1 (clipped at 0), placing the edge just above the (k+1)-th largest
    allowed benign deferral score defers at most k of n calibration points,
    and the conformal rank argument gives P(new benign deferred) <=
    (k + 1) / (n + 1) <= review_budget.

    Returns the edge plus bookkeeping. A budget of 0 returns an edge equal
    to the block threshold, which reproduces the original binary guardrail.
    """
    assert len(benign_deferral_scores) == len(benign_block_scores)
    n = len(benign_deferral_scores)
    if review_budget <= 0 or n == 0:
        return {"lower_edge": block_threshold, "k_allowed": 0, "n_calibration": n,
                "review_budget": review_budget}

    allowed = sorted(
        (d for d, s in zip(benign_deferral_scores, benign_block_scores) if s < block_threshold),
        reverse=True,
    )
    k = max(int(math.floor(review_budget * (n + 1))) - 1, 0)
    if k >= len(allowed):
        # Budget is larger than the number of benign texts the band could
        # ever defer: the edge can drop to 0 without exceeding it.
        edge = 0.0
    else:
        # Smallest edge that still keeps the (k+1)-th largest score OUT of
        # the band. nextafter makes the comparison `>= edge` strict for it.
        edge = math.nextafter(allowed[k], math.inf)
    edge = min(edge, block_threshold)
    return {"lower_edge": edge, "k_allowed": k, "n_calibration": n,
            "review_budget": review_budget}


def adaptive_score_fn(deferral_score_fn: Callable[[str], float], lower_edge: float,
                      block_threshold: float = 0.5) -> Callable[[str], float]:
    """
    Wraps a deferral score so that the existing attacks, which all treat
    "score < block_threshold" as success, instead keep searching until the
    deferral score falls below `lower_edge`.

    The map is monotone increasing and sends lower_edge -> block_threshold,
    so any comparison an attack makes between two candidates' scores comes
    out the same; what moves is the stopping rule. (Heuristics built on
    score DIFFERENCES, such as word-importance ranking, can reorder slightly
    because the map is piecewise linear; that only changes which equally
    valid path the search takes, not what counts as success.)

        s <  e : f(s) = t * s / e
        s >= e : f(s) = t + (1 - t) * (s - e) / (1 - e)
    """
    e, t = lower_edge, block_threshold
    if not (0 < e <= t < 1):
        raise ValueError(f"need 0 < lower_edge <= block_threshold < 1, got e={e}, t={t}")

    def f(text: str) -> float:
        s = deferral_score_fn(text)
        if s < e:
            return t * s / e
        return t + (1 - t) * (s - e) / (1 - e)

    return f


def ensemble_max_score(score_fns: Sequence[Callable[[str], float]]) -> Callable[[str], float]:
    """Deferral score for an ensemble: the most suspicious model wins."""
    def f(text: str) -> float:
        return max(fn(text) for fn in score_fns)
    return f


def evaluate_policy_on_texts(texts: Sequence[str], block_score_fn, deferral_score_fn,
                             lower_edge: float, block_threshold: float = 0.5) -> List[str]:
    """Decisions for a list of texts."""
    return [decide(block_score_fn(t), deferral_score_fn(t), lower_edge, block_threshold)
            for t in texts]


def decision_rates(decisions: Sequence[str]) -> Dict[str, float]:
    n = len(decisions)
    if n == 0:
        return {BLOCK: None, DEFER: None, ALLOW: None, "n": 0}
    return {
        BLOCK: round(sum(d == BLOCK for d in decisions) / n, 4),
        DEFER: round(sum(d == DEFER for d in decisions) / n, 4),
        ALLOW: round(sum(d == ALLOW for d in decisions) / n, 4),
        "n": n,
    }
