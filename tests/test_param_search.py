"""Tests for EXP-1 — grid/random parameter search."""

from __future__ import annotations

from trading_research.experiments.grid_search import expand_grid
from trading_research.experiments.random_search import random_search

# Пример из плана (раздел 12) — MA Cross сетка.
PLAN_GRID = {"fast_ma": [10, 20, 50], "slow_ma": [100, 150, 200]}


def test_grid_size_and_content_deterministic() -> None:
    combos = expand_grid(PLAN_GRID)
    assert len(combos) == 9
    # Полный, детерминированный набор в порядке (fast_ma, slow_ma).
    assert combos == [{"fast_ma": f, "slow_ma": s} for f in (10, 20, 50) for s in (100, 150, 200)]


def test_grid_order_independent_of_key_order() -> None:
    a = expand_grid({"fast_ma": [10, 20], "slow_ma": [100, 200]})
    b = expand_grid({"slow_ma": [100, 200], "fast_ma": [10, 20]})
    assert a == b


def test_grid_predicate_filters_invalid_combos() -> None:
    combos = expand_grid(
        {"fast_ma": [10, 100], "slow_ma": [50, 200]},
        predicate=lambda c: c["fast_ma"] < c["slow_ma"],
    )
    assert {"fast_ma": 100, "slow_ma": 50} not in combos
    assert all(c["fast_ma"] < c["slow_ma"] for c in combos)


def test_empty_grid_yields_single_default_combo() -> None:
    assert expand_grid({}) == [{}]


def test_grid_rejects_empty_value_list() -> None:
    import pytest

    with pytest.raises(ValueError, match="empty"):
        expand_grid({"fast_ma": []})


def test_random_search_reproducible_with_seed() -> None:
    a = random_search(PLAN_GRID, n=4, seed=42)
    b = random_search(PLAN_GRID, n=4, seed=42)
    assert a == b
    assert len(a) == 4


def test_random_search_seed_changes_selection() -> None:
    a = random_search(PLAN_GRID, n=4, seed=1)
    b = random_search(PLAN_GRID, n=4, seed=2)
    assert a != b


def test_random_search_unique_no_duplicates() -> None:
    combos = random_search(PLAN_GRID, n=9, seed=7)
    seen = [tuple(sorted(c.items())) for c in combos]
    assert len(seen) == len(set(seen))
    assert len(combos) == 9


def test_random_search_clamps_to_grid_size() -> None:
    combos = random_search(PLAN_GRID, n=100, seed=0)
    assert len(combos) == 9


def test_random_search_respects_predicate() -> None:
    combos = random_search(PLAN_GRID, n=9, seed=3, predicate=lambda c: c["fast_ma"] < c["slow_ma"])
    assert all(c["fast_ma"] < c["slow_ma"] for c in combos)
