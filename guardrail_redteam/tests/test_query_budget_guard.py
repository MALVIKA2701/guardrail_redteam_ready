import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from blackbox_direct_attack_study import QueryBudgetGuard, QueryBudgetExceeded


def test_query_budget_guard_allows_calls_under_cap():
    guard = QueryBudgetGuard(lambda t: 0.5, cap=3)
    assert guard(" a") == 0.5
    assert guard("b") == 0.5
    assert guard("c") == 0.5
    assert guard.count == 3


def test_query_budget_guard_raises_when_exceeded():
    guard = QueryBudgetGuard(lambda t: 0.5, cap=2)
    guard("a")
    guard("b")
    with pytest.raises(QueryBudgetExceeded):
        guard("c")


def test_query_budget_guard_count_tracks_accurately():
    guard = QueryBudgetGuard(lambda t: 0.1, cap=100)
    for i in range(10):
        guard(f"text {i}")
    assert guard.count == 10
