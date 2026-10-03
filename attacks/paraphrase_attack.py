"""
Paraphrase attack: generate several meaning-preserving rewrites of the input
and keep whichever one most reduces the surrogate classifier's score, subject
to a semantic-similarity floor.

Primary path uses a local HF paraphrasing model (e.g. a T5/Pegasus
paraphraser) so the whole pipeline stays offline/CPU-only and reproducible.
A rule-based synonym-substitution fallback is included so the pipeline can be
smoke-tested without any model downloads.
"""

import random
from typing import Callable, List

# A small built-in synonym table used only as an offline fallback when no
# paraphrase model is available. Not intended to replace a real paraphraser
# for actual experiments -- swap in a proper HF model (see
# `HFParaphraser` below) for paper-quality results.
_SYNONYMS = {
    "bad": ["poor", "awful", "terrible", "substandard"],
    "good": ["great", "fine", "positive", "solid"],
    "stupid": ["foolish", "unwise", "senseless"],
    "hate": ["dislike strongly", "despise", "resent"],
    "idiot": ["fool", "simpleton"],
    "worst": ["least favorable", "most lacking"],
    "disgusting": ["repulsive", "off-putting", "unpleasant"],
    "shut up": ["please stop talking", "be quiet"],
}


def rule_based_paraphrase(text: str, n_candidates: int = 5, rng: random.Random = None) -> List[str]:
    rng = rng or random
    words = text.split()
    candidates = []
    for _ in range(n_candidates):
        new_words = []
        for w in words:
            stripped = w.strip(".,!?").lower()
            if stripped in _SYNONYMS and rng.random() < 0.5:
                repl = rng.choice(_SYNONYMS[stripped])
                new_words.append(repl)
            else:
                new_words.append(w)
        candidates.append(" ".join(new_words))
    return list(set(candidates))


class HFParaphraser:
    """
    Wraps a HuggingFace seq2seq paraphrasing model, e.g.
    'Vamsi/T5_Paraphrase_Paws' or 'tuner007/pegasus_paraphrase'.

    Requires network access to huggingface.co to download weights on first
    use. Falls back to `rule_based_paraphrase` if the model can't be loaded
    (e.g. no network), so the rest of the pipeline still runs end-to-end.
    """

    def __init__(self, model_name: str = "Vamsi/T5_Paraphrase_Paws"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _lazy_load(self):
        if self._model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def paraphrase(self, text: str, n_candidates: int = 5) -> List[str]:
        try:
            self._lazy_load()
            import torch
            prompt = f"paraphrase: {text} </s>"
            inputs = self._tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True, max_length=128
            )
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=128,
                    num_return_sequences=n_candidates,
                    num_beams=max(n_candidates, 5),
                    temperature=1.5,
                    do_sample=True,
                    top_k=120,
                    top_p=0.95,
                )
            candidates = [
                self._tokenizer.decode(o, skip_special_tokens=True) for o in outputs
            ]
            return list(set(candidates))
        except Exception as e:
            print(
                f"[paraphrase_attack] HF paraphrase model unavailable "
                f"({e!r}); falling back to rule-based paraphrase."
            )
            return rule_based_paraphrase(text, n_candidates=n_candidates)


def paraphrase_attack(
    text: str,
    score_fn: Callable[[str], float],
    similarity_fn: Callable[[str, str], float],
    paraphraser=None,
    n_candidates: int = 5,
    min_similarity: float = 0.75,
) -> dict:
    """
    Generates paraphrase candidates and returns the one that most reduces
    score_fn's output while staying above min_similarity.
    """
    paraphraser = paraphraser or HFParaphraser()
    orig_score = score_fn(text)

    candidates = paraphraser.paraphrase(text, n_candidates=n_candidates)
    best_text, best_score, best_sim = text, orig_score, 1.0
    queries_used = 1

    for cand in candidates:
        if not cand.strip() or cand.strip() == text.strip():
            continue
        sim = similarity_fn(text, cand)
        if sim < min_similarity:
            continue
        score = score_fn(cand)
        queries_used += 1
        if score < best_score:
            best_text, best_score, best_sim = cand, score, sim

    return {
        "best_text": best_text,
        "best_score": best_score,
        "orig_score": orig_score,
        "queries_used": queries_used,
        "success": best_score < 0.5 and orig_score >= 0.5,
        "similarity": best_sim,
    }
