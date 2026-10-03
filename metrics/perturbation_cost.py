"""
Metrics for measuring how "expensive"/noticeable an adversarial perturbation
is, so attack success can be reported alongside how much the text had to
change -- a perturbation that mangles text into gibberish "succeeds" in a
much less interesting way than one that reads naturally.
"""

import config


def char_edit_distance(a: str, b: str) -> int:
    try:
        import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:
        # Fallback pure-python Levenshtein (O(n*m), fine for short strings)
        if len(a) < len(b):
            a, b = b, a
        if len(b) == 0:
            return len(a)
        previous_row = range(len(b) + 1)
        for i, ca in enumerate(a):
            current_row = [i + 1]
            for j, cb in enumerate(b):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (ca != cb)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return char_edit_distance(a, b) / max(len(a), len(b))


class SemanticSimilarity:
    """
    Cosine similarity between sentence embeddings. Lazily loads a
    sentence-transformers model; falls back to a crude bag-of-words Jaccard
    similarity if the model can't be downloaded (offline smoke-testing).
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)

    def __call__(self, a: str, b: str) -> float:
        try:
            self._lazy_load()
            import numpy as np
            emb = self._model.encode([a, b])
            num = (emb[0] * emb[1]).sum()
            denom = (
                (emb[0] ** 2).sum() ** 0.5 * (emb[1] ** 2).sum() ** 0.5
            )
            return float(num / denom) if denom else 0.0
        except Exception:
            return self._jaccard_fallback(a, b)

    @staticmethod
    def _jaccard_fallback(a: str, b: str) -> float:
        set_a, set_b = set(a.lower().split()), set(b.lower().split())
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)
