"""
Wrapper around a local, CPU-friendly moderation/toxicity classifier
(default: unitary/toxic-bert), used as the white/gray-box surrogate target.
"""

import config

# Labels that mean "benign" in the registry models' outputs. Their scores must
# be EXCLUDED when computing the toxicity score: e.g.
#   martin-ha/toxic-comment-model                    -> {non-toxic, toxic}
#   facebook/roberta-hate-speech-dynabench-r4-target -> {nothate, hate}
# Taking a plain max over ALL labels (as the original code did) returns the
# benign-class probability for clean text, which is always >= 0.5 for a
# 2-class softmax -- so every input would count as "flagged" and no attack
# could ever succeed on those two models.
BENIGN_LABELS = {
    "non-toxic", "non_toxic", "nontoxic", "not toxic", "not_toxic",
    "nothate", "not_hate", "not hate", "no_hate", "neutral", "normal",
    "label_0", "safe", "clean", "ok", "benign",
}


class LocalSurrogateClassifier:
    def __init__(self, model_name: str = None, cache: bool = True):
        self.model_name = model_name or config.LOCAL_SURROGATE_MODEL
        self._pipe = None
        # Attacks re-score the same strings a lot (original text, repeated
        # candidates, transfer matrix); caching avoids redundant forward passes.
        self._cache = {} if cache else None

    def _lazy_load(self):
        if self._pipe is not None:
            return
        from transformers import pipeline
        device = -1
        try:
            import torch
            if torch.cuda.is_available():
                device = 0
        except ImportError:
            pass
        self._pipe = pipeline(
            "text-classification",
            model=self.model_name,
            top_k=None,       # return all class scores
            truncation=True,
            max_length=512,
            device=device,
        )

    @staticmethod
    def _flatten(result):
        # pipeline output shape varies across transformers versions:
        # [[{label, score}, ...]] or [{label, score}, ...]
        while isinstance(result, list) and result and isinstance(result[0], list):
            result = result[0]
        return result

    @staticmethod
    def toxic_score_from_labels(label_scores) -> float:
        """Max score over the non-benign labels."""
        toxic = [r["score"] for r in label_scores
                 if r["label"].strip().lower() not in BENIGN_LABELS]
        if not toxic:  # model uses unknown label names; fall back to last class
            return label_scores[-1]["score"]
        return max(toxic)

    def score(self, text: str) -> float:
        """
        Returns a single scalar in [0, 1]: the probability the text is
        flagged as toxic/violating, taking the max across the toxicity-related
        (non-benign) labels the model outputs.
        """
        if self._cache is not None and text in self._cache:
            return self._cache[text]
        self._lazy_load()
        s = float(self.toxic_score_from_labels(self._flatten(self._pipe(text))))
        if self._cache is not None:
            self._cache[text] = s
        return s

    def is_flagged(self, text: str) -> bool:
        return self.score(text) >= config.LOCAL_SURROGATE_THRESHOLD


class MockClassifier:
    """
    Deterministic, dependency-free stand-in for LocalSurrogateClassifier.
    Used only for offline smoke-testing of the attack/evaluation pipeline
    when transformers/torch or network access to the HF hub isn't available.

    Heuristic: flags text containing any of a small fixed list of markers;
    score decays as the markers are perturbed/removed. This is NOT a real
    classifier and should never be used for actual paper results.
    """

    MARKERS = ["idiot", "hate", "stupid", "shut up", "worst", "disgusting", "hates"]

    def score(self, text: str) -> float:
        lowered = text.lower()
        hits = sum(1 for m in self.MARKERS if m in lowered)
        if hits == 0:
            return 0.05
        return min(0.5 + 0.25 * hits, 0.98)

    def is_flagged(self, text: str) -> bool:
        return self.score(text) >= 0.5
