"""
Score-guided black-box attack via a genetic algorithm over the continuous
score surface exposed by both local classifiers AND production moderation
APIs (OpenAI Moderation, Perspective both return continuous scores, not just
accept/reject decisions).

Why this matters: the other attacks in this project (char_attacks,
word_substitution_attack) are designed around and validated against the
LOCAL surrogate, then merely REPLAYED against black-box API targets in
transfer_study.py -- that tests transfer, but leaves attack strength on the
table if the goal is "how strong an attack can a realistic black-box
adversary mount directly against the production system." A genetic
algorithm treats score_fn as a fitness function and works identically
whether score_fn is a local classifier's .score() or an API's .score() --
it only ever needs query access to a continuous score, which is exactly
what a black-box adversary targeting these APIs actually has.

This is a genuine strength upgrade over pure greedy/beam search: population
n-gram of variants stays diverse across generations rather than committing
early to one perturbation path, so it explores fluent word-substitution AND
character-perturbation moves jointly instead of confusing them within a
single greedy pass.
"""

import random
from typing import Callable, List

from attacks.char_attacks import ATTACK_PRIMITIVES
from attacks.word_substitution_attack import _get_synonyms, _preserve_case


def _mutate(tokens: List[str], rng: random.Random) -> List[str]:
    """Applies one random mutation: either a word-level synonym swap or a
    character-level perturbation to a single random token."""
    tokens = list(tokens)
    if not tokens:
        return tokens
    idx = rng.randrange(len(tokens))
    word = tokens[idx]

    if rng.random() < 0.5 and word.isalpha() and len(word) >= 3:
        synonyms = _get_synonyms(word)
        if synonyms:
            tokens[idx] = _preserve_case(word, rng.choice(synonyms))
            return tokens

    # fall back to a character-level primitive on this token only
    _, fn = rng.choice(ATTACK_PRIMITIVES)
    tokens[idx] = fn(word, rate=0.5, rng=rng)
    return tokens


def _crossover(parent_a: List[str], parent_b: List[str], rng: random.Random) -> List[str]:
    """Single-point crossover between two equal-length-ish token sequences."""
    if len(parent_a) != len(parent_b) or len(parent_a) < 2:
        return list(parent_a)
    point = rng.randrange(1, len(parent_a))
    return parent_a[:point] + parent_b[point:]


def genetic_search_attack(
    text: str,
    score_fn: Callable[[str], float],
    similarity_fn: Callable[[str, str], float],
    population_size: int = 12,
    generations: int = 8,
    mutation_rate: float = 0.5,
    max_queries: int = 100,
    min_similarity: float = 0.75,
    seed: int = 0,
) -> dict:
    """
    Population-based, score-guided black-box search. Works identically
    against a local classifier or a black-box API -- the only requirement is
    that score_fn returns a continuous score, which both target types in
    this project provide.

    Fitness = classifier score (lower is better / more evasive), with any
    candidate below the similarity floor penalized to fitness=1.0 so it's
    effectively excluded from selection without needing a separate filtering
    pass.
    """
    rng = random.Random(seed)
    orig_score = score_fn(text)
    queries_used = 1
    tokens = text.split()

    def fitness(candidate_tokens: List[str]) -> float:
        nonlocal queries_used
        candidate_text = " ".join(candidate_tokens)
        sim = similarity_fn(text, candidate_text)
        if sim < min_similarity:
            return 1.0, candidate_text, sim
        score = score_fn(candidate_text)
        queries_used += 1
        return score, candidate_text, sim

    # Initialize population with single random mutations of the original.
    population = [list(tokens) for _ in range(population_size)]
    for i in range(1, population_size):
        population[i] = _mutate(population[i], rng)

    best_text, best_score, best_sim = text, orig_score, 1.0

    for gen in range(generations):
        if queries_used >= max_queries:
            break

        scored = []
        for indiv in population:
            if queries_used >= max_queries:
                break
            score, cand_text, sim = fitness(indiv)
            scored.append((score, indiv, cand_text, sim))
            if score < best_score:
                best_text, best_score, best_sim = cand_text, score, sim

        if not scored:
            break
        scored.sort(key=lambda x: x[0])

        if best_score < 0.5:
            break

        # Elitism: keep top half, breed the rest via crossover + mutation.
        survivors = [s[1] for s in scored[: max(2, population_size // 2)]]
        next_gen = list(survivors)
        while len(next_gen) < population_size:
            parent_a, parent_b = rng.sample(survivors, 2) if len(survivors) >= 2 else (survivors[0], survivors[0])
            child = _crossover(parent_a, parent_b, rng)
            if rng.random() < mutation_rate:
                child = _mutate(child, rng)
            next_gen.append(child)
        population = next_gen

    return {
        "best_text": best_text,
        "best_score": best_score,
        "orig_score": orig_score,
        "queries_used": queries_used,
        "success": best_score < 0.5 and orig_score >= 0.5,
        "similarity": best_sim,
    }
