"""Offline tests for the scoring, sampling, and defense-data fixes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import config
from targets.local_classifier import LocalSurrogateClassifier
from data.load_data import _stratified_sample, _to_binary, load_dataset
from defense.adversarial_finetune import build_adversarial_training_set


def test_binary_model_benign_label_is_ignored():
    # martin-ha style output on CLEAN text: non-toxic dominates
    out = [{"label": "non-toxic", "score": 0.97}, {"label": "toxic", "score": 0.03}]
    assert LocalSurrogateClassifier.toxic_score_from_labels(out) == pytest.approx(0.03)
    # roberta dynabench style
    out = [{"label": "nothate", "score": 0.9}, {"label": "hate", "score": 0.1}]
    assert LocalSurrogateClassifier.toxic_score_from_labels(out) == pytest.approx(0.1)


def test_multilabel_model_takes_max_over_toxic_labels():
    out = [{"label": "toxic", "score": 0.2}, {"label": "insult", "score": 0.8},
           {"label": "threat", "score": 0.01}]
    assert LocalSurrogateClassifier.toxic_score_from_labels(out) == pytest.approx(0.8)


def test_flatten_handles_both_pipeline_output_shapes():
    inner = [{"label": "toxic", "score": 0.5}]
    assert LocalSurrogateClassifier._flatten([inner]) == inner
    assert LocalSurrogateClassifier._flatten(inner) == inner


def test_stratified_sample_hits_toxic_fraction():
    pairs = [(f"t{i}", 1) for i in range(100)] + [(f"b{i}", 0) for i in range(900)]
    s = _stratified_sample(pairs, 200, seed=0, toxic_fraction=0.5)
    assert len(s) == 200 and sum(l for _, l in s) == 100


def test_to_binary():
    assert _to_binary(1) == 1 and _to_binary(0.7) == 1 and _to_binary(0.2) == 0
    assert _to_binary(-1) is None  # unscored Kaggle test rows are dropped


def test_real_run_fails_loudly_without_data(monkeypatch):
    monkeypatch.setattr(config, "JIGSAW_HF_MIRRORS", [])
    monkeypatch.setattr(config, "JIGSAW_LOCAL_CSV", "/nonexistent.csv")
    with pytest.raises(RuntimeError):
        load_dataset(allow_fallback=False)
    assert len(load_dataset(allow_fallback=True)) > 0


def test_defense_training_set_is_deduped_and_has_benign():
    records = [
        {"orig_text": "a", "orig_label": 1, "best_text": "a'", "success": True, "attack": "char"},
        {"orig_text": "a", "orig_label": 1, "best_text": "a", "success": False, "attack": "word"},
    ]
    pool = [("x", 0), ("y", 0), ("z", 1)]
    data = build_adversarial_training_set(records, benign_pool=pool)
    texts = [t for t, _ in data]
    assert len(texts) == len(set(texts))
    assert sum(1 for _, l in data if l == 0) == 2   # 1 benign per toxic
    assert ("a'", 1) in data
