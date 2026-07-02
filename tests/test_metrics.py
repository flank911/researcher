"""Tests for BT-7 — MetricsCalculator."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import polars as pl
import pytest

from trading_research.backtest.metrics import (
    MS_PER_YEAR,
    compute_metrics,
    periods_per_year,
)
from trading_research.backtest.orders import OrderReason, Side
from trading_research.backtest.portfolio import Trade

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _equity(values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"equity": [float(v) for v in values]})


def _trade(pnl: float, pnl_pct: float) -> Trade:
    return Trade(
        entry_time=T0,
        exit_time=T0,
        side=Side.LONG,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        qty=1.0,
        pnl=pnl,
        pnl_pct=pnl_pct,
        fees=0.0,
        reason=OrderReason.EXIT,
    )


# --- annualization ----------------------------------------------------------


def test_periods_per_year_matches_timeframe() -> None:
    assert periods_per_year("1d") == pytest.approx(365.0)
    assert periods_per_year("1h") == pytest.approx(365 * 24)
    assert periods_per_year("1m") == pytest.approx(MS_PER_YEAR / 60_000)


def test_unsupported_timeframe_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        compute_metrics(_equity([100.0, 101.0]), [], "7h")


# --- reference values on a hand-computed fixture ----------------------------


def test_total_return_and_drawdown_reference() -> None:
    # peak 120 after bar1, trough 90 → dd = (120-90)/120 = 25%.
    m = compute_metrics(_equity([100.0, 120.0, 90.0, 108.0]), [], "1d")
    assert m.total_return_pct == pytest.approx(8.0)
    assert m.max_drawdown_pct == pytest.approx(25.0)
    # recovery = net profit (8) / max dd abs (30) = 0.2667.
    assert m.recovery_factor == pytest.approx(8.0 / 30.0)


def test_sharpe_reference() -> None:
    # returns: +10%, then -50% ( 110 -> 55 ), then +100% ( 55 -> 110 ).
    equity = [100.0, 110.0, 55.0, 110.0]
    returns = [0.10, -0.5, 1.0]
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    expected = mean / std * math.sqrt(periods_per_year("1d"))
    m = compute_metrics(_equity(equity), [], "1d")
    assert m.sharpe == pytest.approx(expected)


def test_sortino_uses_downside_only() -> None:
    equity = [100.0, 110.0, 55.0, 110.0]
    returns = [0.10, -0.5, 1.0]
    mean = sum(returns) / len(returns)
    downside = math.sqrt(sum(r**2 for r in returns if r < 0) / (len(returns) - 1))
    expected = mean / downside * math.sqrt(periods_per_year("1d"))
    m = compute_metrics(_equity(equity), [], "1d")
    assert m.sortino == pytest.approx(expected)


def test_trade_based_metrics_reference() -> None:
    trades = [_trade(30.0, 3.0), _trade(-10.0, -1.0), _trade(20.0, 2.0), _trade(-20.0, -2.0)]
    m = compute_metrics(_equity([100.0, 120.0]), trades, "1d", min_trades=1)
    assert m.trades_count == 4
    assert m.win_rate == pytest.approx(50.0)
    assert m.profit_factor == pytest.approx((30 + 20) / (10 + 20))
    assert m.avg_trade_pct == pytest.approx((3 - 1 + 2 - 2) / 4)
    assert m.expectancy == pytest.approx((30 - 10 + 20 - 20) / 4)


# --- cross-timeframe comparability ------------------------------------------


def test_sharpe_annualization_scales_with_timeframe() -> None:
    equity = _equity([100.0, 101.0, 100.5, 102.0, 101.0])
    sharpe_h = compute_metrics(equity, [], "1h").sharpe
    sharpe_d = compute_metrics(equity, [], "1d").sharpe
    assert sharpe_h is not None and sharpe_d is not None
    # Same return series → per-bar Sharpe identical; only annualization differs.
    ratio = math.sqrt(periods_per_year("1h") / periods_per_year("1d"))
    assert sharpe_h / sharpe_d == pytest.approx(ratio)


# --- reliability flag -------------------------------------------------------


def test_reliable_flag_respects_min_trades() -> None:
    trades = [_trade(1.0, 1.0) for _ in range(3)]
    below = compute_metrics(_equity([100.0, 103.0]), trades, "1d", min_trades=5)
    at = compute_metrics(_equity([100.0, 103.0]), trades, "1d", min_trades=3)
    assert below.reliable is False
    assert at.reliable is True
    # Values are still populated when unreliable (marked, not zeroed).
    assert below.win_rate == pytest.approx(100.0)


# --- edge cases -------------------------------------------------------------


def test_no_trades() -> None:
    m = compute_metrics(_equity([100.0, 100.0]), [], "1d")
    assert m.trades_count == 0
    assert m.profit_factor is None
    assert m.win_rate is None
    assert m.avg_trade_pct is None
    assert m.expectancy is None
    assert m.reliable is False


def test_flat_equity_has_no_drawdown_and_no_sharpe() -> None:
    m = compute_metrics(_equity([100.0, 100.0, 100.0]), [], "1d")
    assert m.total_return_pct == pytest.approx(0.0)
    assert m.max_drawdown_pct == pytest.approx(0.0)
    assert m.sharpe is None
    assert m.sortino is None
    assert m.recovery_factor is None


def test_no_losing_trades_profit_factor_undefined() -> None:
    trades = [_trade(10.0, 1.0), _trade(5.0, 0.5)]
    m = compute_metrics(_equity([100.0, 115.0]), trades, "1d", min_trades=1)
    assert m.profit_factor is None
    assert m.win_rate == pytest.approx(100.0)


def test_monotonic_up_has_positive_sortino_no_downside() -> None:
    # No negative returns → downside deviation 0 → sortino undefined.
    m = compute_metrics(_equity([100.0, 101.0, 103.0, 106.0]), [], "1d")
    assert m.max_drawdown_pct == pytest.approx(0.0)
    assert m.sortino is None
    assert m.sharpe is not None


def test_single_equity_point() -> None:
    m = compute_metrics(_equity([100.0]), [], "1d")
    assert m.total_return_pct == pytest.approx(0.0)
    assert m.max_drawdown_pct == pytest.approx(0.0)
    assert m.sharpe is None
