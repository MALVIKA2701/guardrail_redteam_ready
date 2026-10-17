"""
Where the deferral band's BENIGN calibration and test texts come from.

The review budget is only meaningful relative to the benign traffic it is
calibrated on. Jigsaw's non-toxic comments are mostly mild, which makes a
low band cheap. Real traffic also contains benign text that is heated,
sarcastic or uses strong language without being abusive, and that text
scores higher. These sources let the study check whether the result
survives harder benign traffic:

  jigsaw       non-toxic comments from the same stratified sample as the
               attacked texts (the original setting)
  jigsaw_hard  non-toxic-LABELLED Jigsaw comments that contain at least one
               strong-language or insult-adjacent word (see HARD_TERMS).
               Selected lexically, never by a classifier's score, so the
               selection is not biased toward or against any model.
  hatexplain   posts the HateXplain annotators labelled "normal": social-
               media text from a different platform and time period.

In every case the attacked toxic texts stay the same; only the benign
traffic the band is calibrated and tested on changes.
"""

import random
import re
from typing import List, Sequence, Tuple

# Words that make a benign comment look heated without making it abusive.
# Matched as whole words, case-insensitive.
HARD_TERMS = [
    "damn", "hell", "crap", "shit", "fuck", "fucking", "bloody", "bullshit",
    "stupid", "idiot", "idiots", "idiotic", "dumb", "moron", "ridiculous",
    "hate", "hated", "sucks", "suck", "kill", "killed", "worst", "pathetic",
    "shut up", "nonsense", "garbage", "awful", "disgusting", "liar", "lies",
]

_HARD_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in HARD_TERMS) + r")\b", re.IGNORECASE)

SOURCES = ("jigsaw", "jigsaw_hard", "hatexplain")


def is_hard_benign(text: str) -> bool:
    return bool(_HARD_RE.search(text))


def _sample(texts: Sequence[str], n: int, seed: int, exclude: set) -> List[str]:
    pool = sorted(set(t for t in texts if t not in exclude))  # sorted: reproducible order
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def select_benign(source: str, sample_pairs: Sequence[Tuple[str, int]], n: int, seed: int,
                  full_pairs_loader=None) -> List[str]:
    """
    Returns up to n benign texts for calibration + testing.

    sample_pairs:      the (text, label) sample the attacks are drawn from.
    full_pairs_loader: zero-argument callable returning (text, label) pairs
                       for the source's full dataset (only needed for
                       jigsaw_hard and hatexplain). Texts that also appear
                       in sample_pairs are excluded, so no text is both
                       attacked and used for calibration.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown benign source {source!r}; choose from {SOURCES}")
    in_sample = {t for t, _ in sample_pairs}

    if source == "jigsaw":
        benign = [t for t, y in sample_pairs if y == 0]
        return _sample(benign, n, seed, exclude=set())

    if full_pairs_loader is None:
        raise ValueError(f"benign source {source!r} needs full_pairs_loader")
    pairs = full_pairs_loader()
    benign = [t for t, y in pairs if y == 0]
    if source == "jigsaw_hard":
        benign = [t for t in benign if is_hard_benign(t)]
    return _sample(benign, n, seed, exclude=in_sample)
