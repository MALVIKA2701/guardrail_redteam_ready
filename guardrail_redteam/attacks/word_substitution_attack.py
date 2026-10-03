"""
Word-level synonym substitution attack, in the style of TextFooler
(Jin et al. 2020): for each word in the input, generate synonym candidates,
rank words by how much replacing them reduces the classifier's score, and
greedily substitute the highest-impact words first, subject to a semantic
similarity budget.

This is a well-established published technique. Including it alongside the
character-level and paraphrase attacks gives the project three distinct,
complementary attack families operating at three different granularities
(character / word / sentence) -- a much stronger basis for comparing which
perturbation granularity guardrail models are most vulnerable to.
"""

import random
import re
from typing import Callable, Dict, List

# Offline synonym table used when WordNet (via nltk) isn't available. Covers
# common words that tend to trigger toxicity classifiers, as a fallback so
# the pipeline remains runnable without network access to nltk_data.
_FALLBACK_SYNONYMS: Dict[str, List[str]] = {
    "bad": ["poor", "awful", "terrible", "substandard", "inferior"],
    "good": ["great", "fine", "positive", "solid", "decent"],
    "stupid": ["foolish", "unwise", "senseless", "silly"],
    "hate": ["dislike", "despise", "resent", "loathe"],
    "hates": ["dislikes", "despises", "resents", "loathes"],
    "idiot": ["fool", "simpleton", "dolt"],
    "worst": ["least favorable", "poorest", "most lacking"],
    "disgusting": ["repulsive", "off-putting", "unpleasant", "distasteful"],
    "kill": ["stop", "end", "eliminate", "remove"],
    "ugly": ["unattractive", "unappealing", "homely"],
    "dumb": ["unwise", "foolish", "misguided"],
    "shut": ["close", "quiet", "silence"],
    "nobody": ["no one", "not a soul"],
    "wants": ["desires", "needs", "seeks"],
}


def _get_synonyms(word: str) -> List[str]:
    """Try WordNet via nltk first; fall back to the small built-in table."""
    lower = word.lower()
    try:
        from nltk.corpus import wordnet
        synonyms = set()
        for syn in wordnet.synsets(lower):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != lower:
                    synonyms.add(name)
        if synonyms:
            return list(synonyms)[:6]
    except Exception:
        pass
    return _FALLBACK_SYNONYMS.get(lower, [])


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def word_substitution_attack(
    text: str,
    score_fn: Callable[[str], float],
    similarity_fn: Callable[[str, str], float],
    max_queries: int = 40,
    min_similarity: float = 0.75,
) -> dict:
    """
    Greedy, importance-ranked word substitution:
      1. Score each word's "importance" by how much the classifier score
         drops when that word is deleted (a query-efficient proxy widely
         used in this literature, e.g. TextFooler's word importance ranking).
      2. Process words in descending importance order; for each, try its
         synonym candidates and keep the first one that reduces the running
         score without breaking the similarity floor.
      3. Stop when the query budget is exhausted or the text is no longer
         flagged.
    """
    orig_score = score_fn(text)
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    queries_used = 1

    # Step 1: importance ranking via leave-one-out deletion
    importances = []
    for i, tok in enumerate(tokens):
        if not tok.isalpha() or len(tok) < 3:
            continue
        if queries_used >= max_queries:
            break
        without = " ".join(tokens[:i] + tokens[i + 1:])
        s = score_fn(without)
        queries_used += 1
        importances.append((orig_score - s, i))
    importances.sort(reverse=True)  # most-important word first

    working_tokens = list(tokens)
    best_text = text
    best_score = orig_score

    # Step 2: greedy substitution in importance order
    for _, i in importances:
        if queries_used >= max_queries:
            break
        word = working_tokens[i]
        candidates = _get_synonyms(word)
        for cand in candidates:
            if queries_used >= max_queries:
                break
            cand_cased = _preserve_case(word, cand)
            trial_tokens = list(working_tokens)
            trial_tokens[i] = cand_cased
            trial_text = " ".join(trial_tokens)
            sim = similarity_fn(text, trial_text)
            if sim < min_similarity:
                continue
            score = score_fn(trial_text)
            queries_used += 1
            if score < best_score:
                working_tokens[i] = cand_cased
                best_text, best_score = trial_text, score
                break  # move to next word once this one improved things

        if best_score < 0.5:
            break

    final_sim = similarity_fn(text, best_text)
    return {
        "best_text": best_text,
        "best_score": best_score,
        "orig_score": orig_score,
        "queries_used": queries_used,
        "success": best_score < 0.5 and orig_score >= 0.5,
        "similarity": final_sim,
    }
