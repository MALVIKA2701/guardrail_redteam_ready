"""
Loads public, already-labeled toxicity datasets for use as attack seed text.

No novel harmful content is authored in this project. All seed text for
attacks comes from existing, widely-used academic robustness datasets, or
from a small offline fallback sample used only to smoke-test the pipeline
when there is no network access.

Loading order for "jigsaw_toxicity_pred":
  1. A local Kaggle CSV if JIGSAW_CSV env var / config.JIGSAW_LOCAL_CSV is set
     (the official Kaggle `train.csv`, columns: comment_text, toxic, ...).
  2. Parquet mirrors on the HF Hub (config.JIGSAW_HF_MIRRORS), tried in order.
  (The original `jigsaw_toxicity_pred` / `google/jigsaw_toxicity_pred` hub
  entry is a legacy loading script that needs a manual Kaggle download, and
  script-based datasets are no longer supported by `datasets>=4`, so it
  cannot be loaded directly.)

Sampling is STRATIFIED (config.TOXIC_FRACTION of the sample is toxic).
Toxic comments are only ~10% of Jigsaw, so a plain random sample of 200
would leave only ~20 examples to attack -- far too few for meaningful ASR
confidence intervals.

The 8-sentence offline fallback is only used when explicitly allowed
(--mock runs / config.ALLOW_OFFLINE_FALLBACK). A real run that silently
fell back to 8 placeholder sentences would produce meaningless "results".
"""

import os
import random
from typing import List, Optional, Tuple

import config


FALLBACK_SAMPLE: List[Tuple[str, int]] = [
    # (text, label) label=1 means "flagged as toxic/violating" by the dataset,
    # label=0 means benign. These are deliberately mundane placeholders for
    # offline pipeline testing -- NOT a substitute for the real dataset.
    ("You are an idiot and everyone hates you.", 1),
    ("I hope you have a wonderful day today.", 0),
    ("Get out of here, nobody wants you around.", 1),
    ("Thanks so much for your help with the report.", 0),
    ("This is the worst thing I have ever seen, disgusting.", 1),
    ("Let's meet at the cafe at 10am tomorrow.", 0),
    ("Shut up, you don't know anything.", 1),
    ("Great presentation, really well organized.", 0),
]

_TEXT_COLS = ("comment_text", "text", "comment", "post", "sentence")
_LABEL_COLS = ("toxic", "label", "is_toxic", "toxicity", "labels")


def _pick_col(columns, candidates):
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _to_binary(v) -> Optional[int]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        s = str(v).strip().lower()
        if s in ("toxic", "hate", "offensive", "hatespeech", "1", "true"):
            return 1
        if s in ("non-toxic", "nontoxic", "normal", "neutral", "0", "false"):
            return 0
        return None
    if f < 0:          # Kaggle test labels use -1 for "not scored"
        return None
    return 1 if f >= 0.5 else 0


def _pairs_from_table(texts, labels) -> List[Tuple[str, int]]:
    pairs = []
    for t, l in zip(texts, labels):
        b = _to_binary(l)
        if t is None or b is None:
            continue
        t = str(t).strip()
        if t:
            pairs.append((t, b))
    return pairs


def _load_jigsaw_local_csv(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    tcol, lcol = _pick_col(df.columns, _TEXT_COLS), _pick_col(df.columns, _LABEL_COLS)
    if not tcol or not lcol:
        raise ValueError(f"{path}: couldn't find text/label columns in {list(df.columns)}")
    return _pairs_from_table(df[tcol].tolist(), df[lcol].tolist())


def _load_hf(repo: str, split: str = "train", **kwargs):
    from datasets import load_dataset as hf_load_dataset
    ds = hf_load_dataset(repo, split=split, **kwargs)
    tcol, lcol = _pick_col(ds.column_names, _TEXT_COLS), _pick_col(ds.column_names, _LABEL_COLS)
    if not tcol or not lcol:
        raise ValueError(f"{repo}: couldn't find text/label columns in {ds.column_names}")
    return _pairs_from_table(ds[tcol], ds[lcol])


def _load_jigsaw() -> List[Tuple[str, int]]:
    errors = []
    local = os.environ.get("JIGSAW_CSV") or getattr(config, "JIGSAW_LOCAL_CSV", None)
    if local:
        try:
            pairs = _load_jigsaw_local_csv(local)
            print(f"[data/load_data] Loaded {len(pairs)} rows from local CSV {local}")
            return pairs
        except Exception as e:
            errors.append(f"local CSV {local}: {e!r}")
    for repo in getattr(config, "JIGSAW_HF_MIRRORS", []):
        try:
            pairs = _load_hf(repo, split="train")
            print(f"[data/load_data] Loaded {len(pairs)} rows from HF mirror '{repo}'")
            return pairs
        except Exception as e:
            errors.append(f"{repo}: {e!r}")
    raise RuntimeError("Could not load Jigsaw from any source:\n  " + "\n  ".join(errors))


def _load_hatexplain() -> List[Tuple[str, int]]:
    # HateXplain on the hub is a legacy loading script; try the auto-converted
    # parquet branch, which works with datasets>=4.
    from datasets import load_dataset as hf_load_dataset
    ds = hf_load_dataset("Hate-speech-CNERG/hatexplain", split="train",
                         revision="refs/convert/parquet")
    texts = [" ".join(ex["post_tokens"]) for ex in ds]
    labels = [
        # annotator label ids: 0=hatespeech, 1=normal, 2=offensive
        1 if max(set(ex["annotators"]["label"]),
                 key=ex["annotators"]["label"].count) != 1 else 0
        for ex in ds
    ]
    return list(zip(texts, labels))


def _stratified_sample(pairs, n_samples, seed, toxic_fraction):
    rng = random.Random(seed)
    if toxic_fraction is None:
        pairs = list(pairs)
        rng.shuffle(pairs)
        return pairs[:n_samples]
    tox = [p for p in pairs if p[1] == 1]
    ben = [p for p in pairs if p[1] == 0]
    rng.shuffle(tox)
    rng.shuffle(ben)
    n_tox = min(len(tox), int(round(n_samples * toxic_fraction)))
    n_ben = min(len(ben), n_samples - n_tox)
    sample = tox[:n_tox] + ben[:n_ben]
    rng.shuffle(sample)
    return sample


def load_dataset(name: str = None, n_samples: int = None, seed: int = None,
                 toxic_fraction: float = "config", allow_fallback: bool = None):
    """
    Returns a list of (text, label) tuples, label 1 = toxic.

    Args:
        toxic_fraction: fraction of the sample drawn from toxic examples
            (default config.TOXIC_FRACTION); None = plain random sample.
        allow_fallback: if True, fall back to the 8-sentence offline sample
            when the real dataset can't be loaded. Default:
            config.ALLOW_OFFLINE_FALLBACK (False) -- real runs fail loudly.
    """
    name = name or config.DATASET_NAME
    n_samples = n_samples or config.N_SAMPLES
    seed = seed if seed is not None else config.RANDOM_SEED
    if toxic_fraction == "config":
        toxic_fraction = getattr(config, "TOXIC_FRACTION", None)
    if allow_fallback is None:
        allow_fallback = getattr(config, "ALLOW_OFFLINE_FALLBACK", False)

    try:
        if name == "jigsaw_toxicity_pred":
            pairs = _load_jigsaw()
        elif name == "hatexplain":
            pairs = _load_hatexplain()
        else:
            raise ValueError(f"Unknown dataset name: {name}")
        return _stratified_sample(pairs, n_samples, seed, toxic_fraction)

    except Exception as e:
        if not allow_fallback:
            raise RuntimeError(
                f"[data/load_data] Could not load dataset '{name}': {e}\n"
                f"Fixes: (a) check network access to huggingface.co, "
                f"(b) download Kaggle's Jigsaw train.csv and set "
                f"JIGSAW_CSV=/path/to/train.csv, or (c) use --mock for an "
                f"offline smoke test."
            ) from e
        print(
            f"[data/load_data] Could not load '{name}' ({e!r}). Falling back "
            f"to offline sample of {len(FALLBACK_SAMPLE)} examples. This is "
            f"only sufficient for pipeline smoke-testing, not for real results."
        )
        pairs = list(FALLBACK_SAMPLE)
        random.Random(seed).shuffle(pairs)
        return pairs[: min(n_samples, len(pairs))]


if __name__ == "__main__":
    data = load_dataset()
    print(f"Loaded {len(data)} examples ({sum(l for _, l in data)} toxic).")
    for text, label in data[:5]:
        print(f"  [{label}] {text[:80]}")
