"""Tests for BT-1 — Strategy Protocol + MA Cross."""

from __future__ import annotations

import polars as pl
import pytest

from trading_research.strategies import available_strategies, get_strategy
from trading_research.strategies.base import LONG_SIGNAL, SHORT_SIGNAL, Strategy
from trading_research.strategies.ma_cross import MovingAverageCrossStrategy


def _frame(closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"close": closes})


def test_registry_resolves_ma_cross() -> None:
    strat = get_strategy("ma_cross")
    assert strat.name == "ma_cross"
    assert "ma_cross" in available_strategies()
    assert isinstance(strat, Strategy)


def test_warmup_equals_slow_period() -> None:
    strat = MovingAverageCrossStrategy()
    assert strat.warmup_bars({"fast_ma": 10, "slow_ma": 50}) == 50


def test_signals_are_clean_booleans() -> None:
    strat = MovingAverageCrossStrategy()
    df = strat.generate_signals(_frame([1, 2, 3, 4, 5, 6]), {"fast_ma": 2, "slow_ma": 3})
    for col in (LONG_SIGNAL, SHORT_SIGNAL):
        assert df.schema[col] == pl.Boolean
        assert df[col].null_count() == 0


def test_upward_series_triggers_long_not_short() -> None:
    strat = MovingAverageCrossStrategy()
    closes = [10, 10, 10, 10, 11, 12, 13, 14, 15]
    df = strat.generate_signals(_frame(closes), {"fast_ma": 2, "slow_ma": 4})
    assert df[LONG_SIGNAL].sum() >= 1
    assert df[SHORT_SIGNAL].sum() == 0


def test_downward_series_triggers_short_not_long() -> None:
    strat = MovingAverageCrossStrategy()
    closes = [15, 15, 15, 15, 14, 13, 12, 11, 10]
    df = strat.generate_signals(_frame(closes), {"fast_ma": 2, "slow_ma": 4})
    assert df[SHORT_SIGNAL].sum() >= 1
    assert df[LONG_SIGNAL].sum() == 0


def test_cross_up_then_down_orders_signals() -> None:
    strat = MovingAverageCrossStrategy()
    closes = [10, 10, 10, 10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 9]
    df = strat.generate_signals(_frame(closes), {"fast_ma": 2, "slow_ma": 4})
    long_idx = df.with_row_index().filter(pl.col(LONG_SIGNAL))["index"].to_list()
    short_idx = df.with_row_index().filter(pl.col(SHORT_SIGNAL))["index"].to_list()
    assert long_idx and short_idx
    assert min(long_idx) < min(short_idx)


def test_invalid_params_raise() -> None:
    strat = MovingAverageCrossStrategy()
    with pytest.raises(ValueError, match="fast_ma must be < slow_ma"):
        strat.generate_signals(_frame([1, 2, 3]), {"fast_ma": 50, "slow_ma": 10})
    with pytest.raises(KeyError):
        strat.warmup_bars({"fast_ma": 10})


def test_unknown_strategy_raises() -> None:
    with pytest.raises(KeyError, match="Unknown strategy"):
        get_strategy("does_not_exist")
