import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.perturbation_cost import (
    char_edit_distance, normalized_edit_distance, SemanticSimilarity,
)


def test_char_edit_distance_identical_strings():
    assert char_edit_distance("hello", "hello") == 0


def test_char_edit_distance_known_case():
    assert char_edit_distance("kitten", "sitting") == 3


def test_normalized_edit_distance_bounds():
    d = normalized_edit_distance("hello world", "hello wor1d")
    assert 0 <= d <= 1


def test_normalized_edit_distance_empty_strings():
    assert normalized_edit_distance("", "") == 0.0


def test_semantic_similarity_jaccard_fallback_identical():
    sim = SemanticSimilarity._jaccard_fallback("hello world", "hello world")
    assert sim == 1.0


def test_semantic_similarity_jaccard_fallback_disjoint():
    sim = SemanticSimilarity._jaccard_fallback("apple banana", "car truck")
    assert sim == 0.0


def test_semantic_similarity_jaccard_fallback_partial_overlap():
    sim = SemanticSimilarity._jaccard_fallback("the quick brown fox", "the quick red fox")
    assert 0 < sim < 1
