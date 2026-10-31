"""Tests for BT-7 — MetricsCalculator."""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, datetime
from fractions import Fraction

import polars as pl
import pytest

from trading_research.backtest.metrics import (
    MS_PER_YEAR,
    _downside_deviation,
    _mean,
    _stdev,
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


# --- INF-8: суммирование не зависит от версии интерпретатора ----------------
#
# Встроенный ``sum`` складывает слева направо на Python 3.11 и компенсированно
# (суммирование Нейтмайера) на 3.12+. Одна и та же кривая капитала давала из-за
# этого разные последние биты ``sharpe``, и побитовая сверка с эталоном в
# `test_walk_forward_performance` падала на 3.11, хотя считала то же самое.
# ``math.fsum`` округляет точную сумму один раз, то есть не зависит ни от версии,
# ни от сборки — это и проверяется ниже.
#
# Ряды подобраны так, чтобы наивное сложение теряло мелкие слагаемые под крупным:
# ULP у 1e16 равен 2, поэтому слагаемые порядка единицы, добавляемые по одному, не
# двигают накопитель вовсе, а точная сумма их удерживает. На реальной кривой
# расхождение сидит в 13-м знаке и на коротком ряде его не покажешь — поэтому ряды
# условные, и каждый тест отдельно проверяет, что ряд **различающий**: иначе
# «упрощение» фикстуры превратило бы тест в тавтологию, ничего не сломав.

# Прыжок в 1e16 раз, дальше множители — точные двоичные дроби, чтобы доходности
# восстанавливались из кривой бит в бит (это проверяет отдельный тест ниже).
_JUMP = 1e16
_FACTORS = (1.25, 2.0, 1.25, 1.25, 2.0, 1.25, 2.0, 1.25, 1.25, 1.25)


def _wide_equity() -> list[float]:
    curve = [1.0, _JUMP]
    for factor in _FACTORS:
        curve.append(curve[-1] * factor)
    return curve


_WIDE_EQUITY = _wide_equity()
_WIDE_RETURNS = [_JUMP, *(factor - 1.0 for factor in _FACTORS)]
_WIDE_NEGATIVES = [-1.0] + [-1e-9] * 200


def _naive_sum(values: Iterable[float]) -> float:
    """Сложение слева направо — ровно то, что делает встроенный ``sum`` на py3.11."""
    total = 0.0
    for value in values:
        total += value
    return total


def _exact_sum(values: Iterable[float]) -> float:
    """Точная сумма, округлённая один раз: эталон, не зависящий от реализации."""
    return float(sum((Fraction(v) for v in values), Fraction(0)))


def test_wide_equity_curve_gives_the_wide_returns() -> None:
    """Кривая и ряд доходностей согласованы — на этом держатся тесты ниже."""
    pairs = zip(_WIDE_EQUITY, _WIDE_EQUITY[1:], strict=False)
    returns = [cur / prev - 1.0 for prev, cur in pairs if prev > 0]
    assert returns == _WIDE_RETURNS


def test_mean_and_stdev_are_exactly_rounded_not_left_to_right() -> None:
    n = len(_WIDE_RETURNS)
    exact_mean = _exact_sum(_WIDE_RETURNS) / n
    exact_var = _exact_sum((v - exact_mean) ** 2 for v in _WIDE_RETURNS) / (n - 1)

    assert _mean(_WIDE_RETURNS) == exact_mean
    assert _stdev(_WIDE_RETURNS) == math.sqrt(exact_var)

    # Ряд различающий: при сложении слева направо числа были бы другими.
    naive_mean = _naive_sum(_WIDE_RETURNS) / n
    assert naive_mean != exact_mean
    assert math.sqrt(_naive_sum((v - naive_mean) ** 2 for v in _WIDE_RETURNS) / (n - 1)) != _stdev(
        _WIDE_RETURNS
    )


def test_downside_deviation_is_exactly_rounded_not_left_to_right() -> None:
    n = len(_WIDE_NEGATIVES)
    exact = math.sqrt(_exact_sum(v**2 for v in _WIDE_NEGATIVES) / (n - 1))

    assert _downside_deviation(_WIDE_NEGATIVES) == exact
    naive = math.sqrt(_naive_sum(v**2 for v in _WIDE_NEGATIVES) / (n - 1))
    assert naive != exact


def test_profit_factor_is_exactly_rounded_not_left_to_right() -> None:
    pnls = [*_WIDE_RETURNS, -2.0]
    trades = [_trade(pnl, pnl / 100.0) for pnl in pnls]
    exact = _exact_sum(_WIDE_RETURNS) / 2.0

    m = compute_metrics(_equity([100.0, 101.0]), trades, "1d", min_trades=1)
    assert m.profit_factor == exact
    assert _naive_sum(_WIDE_RETURNS) / 2.0 != exact


def test_sharpe_does_not_depend_on_summation_order() -> None:
    """Метрика, которую видит пользователь, — точно округлённая, а не «как сложилось».

    Это и есть INF-8: тот же прогон на py3.11 давал другой последний знак sharpe
    при той же кривой капитала, и побитовая сверка RUN-8 проверяла интерпретатор, а
    не то, ради чего написана.
    """
    n = len(_WIDE_RETURNS)
    ppy = periods_per_year("1d")
    exact_mean = _exact_sum(_WIDE_RETURNS) / n
    exact_var = _exact_sum((v - exact_mean) ** 2 for v in _WIDE_RETURNS) / (n - 1)
    expected = exact_mean / math.sqrt(exact_var) * math.sqrt(ppy)

    assert compute_metrics(_equity(_WIDE_EQUITY), [], "1d").sharpe == expected

    naive_mean = _naive_sum(_WIDE_RETURNS) / n
    naive_var = _naive_sum((v - naive_mean) ** 2 for v in _WIDE_RETURNS) / (n - 1)
    assert naive_mean / math.sqrt(naive_var) * math.sqrt(ppy) != expected
