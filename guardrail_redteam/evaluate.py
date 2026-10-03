"""
Core evaluation harness. Runs the configured attacks over a set of seed
texts against the local surrogate classifier, and reports Attack Success
Rate (ASR) plus perturbation-cost stats.
"""

import json
import os
from typing import List, Tuple

import config
from attacks.char_attacks import greedy_search_attack
from attacks.paraphrase_attack import paraphrase_attack, HFParaphraser
from attacks.word_substitution_attack import word_substitution_attack
from attacks.genetic_search_attack import genetic_search_attack
from metrics.perturbation_cost import SemanticSimilarity, normalized_edit_distance


def run_attacks(
    data: List[Tuple[str, int]],
    classifier,
    attack_names: List[str] = ("char", "word", "paraphrase"),
    max_queries: int = None,
    min_similarity: float = None,
    only_true_positives: bool = True,
) -> List[dict]:
    """
    Runs the requested attacks against every example in `data` that is
    labeled toxic AND currently flagged by the classifier (true positives),
    and returns one result record per (example, attack) pair.

    Restricting to true positives is the standard ASR definition: "evading"
    the classifier on a benign-labeled text it wrongly flagged would just be
    fixing a false positive, not a successful attack.
    """
    max_queries = max_queries or config.MAX_QUERIES_PER_EXAMPLE
    min_similarity = min_similarity or config.MIN_SEMANTIC_SIMILARITY
    sim_fn = SemanticSimilarity()
    paraphraser = HFParaphraser() if "paraphrase" in attack_names else None

    records = []
    for idx, (text, label) in enumerate(data):
        base_score = classifier.score(text)
        if base_score < config.LOCAL_SURROGATE_THRESHOLD:
            continue  # only attack examples the classifier currently flags
        if only_true_positives and label != 1:
            continue  # ...and that are actually labeled toxic

        for attack_name in attack_names:
            if attack_name == "char":
                result = greedy_search_attack(
                    text,
                    score_fn=classifier.score,
                    similarity_fn=sim_fn,
                    max_queries=max_queries,
                    min_similarity=min_similarity,
                    seed=idx,
                )
            elif attack_name == "word":
                result = word_substitution_attack(
                    text,
                    score_fn=classifier.score,
                    similarity_fn=sim_fn,
                    max_queries=max_queries,
                    min_similarity=min_similarity,
                )
            elif attack_name == "genetic":
                result = genetic_search_attack(
                    text,
                    score_fn=classifier.score,
                    similarity_fn=sim_fn,
                    max_queries=max_queries,
                    min_similarity=min_similarity,
                    seed=idx,
                )
            elif attack_name == "paraphrase":
                result = paraphrase_attack(
                    text,
                    score_fn=classifier.score,
                    similarity_fn=sim_fn,
                    paraphraser=paraphraser,
                    n_candidates=config.MAX_PARAPHRASE_CANDIDATES,
                    min_similarity=min_similarity,
                )
            else:
                raise ValueError(f"Unknown attack: {attack_name}")

            record = {
                "example_idx": idx,
                "attack": attack_name,
                "orig_text": text,
                "orig_label": label,
                **result,
                "edit_distance": normalized_edit_distance(text, result["best_text"]),
            }
            records.append(record)

    return records


def summarize(records: List[dict]) -> dict:
    """Aggregate ASR and perturbation-cost stats, overall and per attack."""
    summary = {"overall": _summarize_subset(records)}
    for attack_name in sorted(set(r["attack"] for r in records)):
        subset = [r for r in records if r["attack"] == attack_name]
        summary[attack_name] = _summarize_subset(subset)
    return summary


def _summarize_subset(records: List[dict]) -> dict:
    if not records:
        return {"n": 0}
    n = len(records)
    successes = [r for r in records if r["success"]]
    asr = len(successes) / n
    avg_queries = sum(r["queries_used"] for r in records) / n
    avg_edit_dist = sum(r["edit_distance"] for r in records) / n
    avg_sim = sum(r["similarity"] for r in records) / n
    return {
        "n": n,
        "attack_success_rate": round(asr, 4),
        "avg_queries_used": round(avg_queries, 2),
        "avg_normalized_edit_distance": round(avg_edit_dist, 4),
        "avg_semantic_similarity_of_successes": round(
            sum(r["similarity"] for r in successes) / len(successes), 4
        ) if successes else None,
    }


def save_results(records: List[dict], path: str = None):
    path = path or config.ATTACK_RESULTS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} attack records to {path}")


def load_results(path: str = None) -> List[dict]:
    path = path or config.ATTACK_RESULTS_PATH
    with open(path) as f:
        return [json.loads(line) for line in f]
