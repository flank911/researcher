"""Tests for BT-6 — funding payments in PnL."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from trading_research.backtest.engine import BacktestEngine
from trading_research.backtest.funding import bucket_funding
from trading_research.backtest.orders import Side
from trading_research.backtest.portfolio import Portfolio
from trading_research.domain import ExecutionModel, FillOn, PositionSizeType
from trading_research.strategies.base import LONG_SIGNAL, SHORT_SIGNAL

H = timedelta(hours=1)
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _exec(**overrides: object) -> ExecutionModel:
    base: dict[str, object] = {
        "initial_balance": 10_000.0,
        "commission_rate": 0.0,
        "slippage_pct": 0.0,
        "leverage": 1,
        "position_size_type": PositionSizeType.PERCENT_BALANCE,
        "position_size_value": 100.0,
        "allow_short": True,
        "close_on_reverse_signal": True,
        "signal_lag": 0,
        "fill_on": FillOn.CURRENT_CLOSE,
    }
    base.update(overrides)
    return ExecutionModel(**base)  # type: ignore[arg-type]


def _signals(rows: list[tuple[float, bool, bool]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open_time": [T0 + i * H for i in range(len(rows))],
            "close": [r[0] for r in rows],
            LONG_SIGNAL: [r[1] for r in rows],
            SHORT_SIGNAL: [r[2] for r in rows],
        }
    )


def _funding(entries: list[tuple[datetime, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open_time": [e[0] for e in entries],
            "funding_rate": [e[1] for e in entries],
        }
    )


# --- unit: bucket_funding ---------------------------------------------------


def test_bucket_none_and_empty_return_zeros() -> None:
    times = [T0 + i * H for i in range(3)]
    assert bucket_funding(times, None) == [0.0, 0.0, 0.0]
    assert bucket_funding(times, _funding([])) == [0.0, 0.0, 0.0]
    assert bucket_funding([], _funding([(T0, 0.01)])) == []


def test_bucket_aligns_to_bar() -> None:
    times = [T0 + i * H for i in range(4)]
    per_bar = bucket_funding(times, _funding([(T0 + 1 * H, 0.01), (T0 + 3 * H, -0.02)]))
    assert per_bar == [0.0, 0.01, 0.0, -0.02]


def test_bucket_sums_multiple_within_one_bar() -> None:
    # Дневные бары, funding каждые 8ч → 3 ставки в первый бар.
    times = [T0, T0 + timedelta(days=1)]
    funding = _funding(
        [
            (T0, 0.01),
            (T0 + timedelta(hours=8), 0.01),
            (T0 + timedelta(hours=16), 0.01),
            (T0 + timedelta(days=1), 0.02),
        ]
    )
    assert bucket_funding(times, funding) == pytest.approx([0.03, 0.02])


def test_bucket_out_of_range_ignored() -> None:
    times = [T0 + i * H for i in range(3)]
    per_bar = bucket_funding(
        times, _funding([(T0 - 5 * H, 0.5), (T0 + 100 * H, 0.5), (T0 + 1 * H, 0.01)])
    )
    assert per_bar == [0.0, 0.01, 0.0]


def test_bucket_missing_columns_raise() -> None:
    bad = pl.DataFrame({"open_time": [T0], "rate": [0.01]})
    with pytest.raises(ValueError, match="funding frame missing"):
        bucket_funding([T0], bad)


# --- unit: Portfolio.apply_funding ------------------------------------------


def test_apply_funding_long_pays_when_rate_positive() -> None:
    p = Portfolio(10_000.0)
    p.open(Side.LONG, qty=100.0, price=100.0, time=T0)
    cash = p.apply_funding(0.01, mark_price=100.0)
    assert cash == pytest.approx(-100.0)  # long платит
    assert p.balance == pytest.approx(9_900.0)
    assert p.funding_total == pytest.approx(-100.0)


def test_apply_funding_short_receives_when_rate_positive() -> None:
    p = Portfolio(10_000.0)
    p.open(Side.SHORT, qty=100.0, price=100.0, time=T0)
    cash = p.apply_funding(0.01, mark_price=100.0)
    assert cash == pytest.approx(100.0)  # short получает
    assert p.balance == pytest.approx(10_100.0)


def test_apply_funding_noop_without_position() -> None:
    p = Portfolio(10_000.0)
    assert p.apply_funding(0.01, mark_price=100.0) == 0.0
    assert p.balance == pytest.approx(10_000.0)


# --- integration: engine ----------------------------------------------------


def _long_run() -> pl.DataFrame:
    # enter long@100 (qty=100), hold, final close@110 → gross pnl +1000
    return _signals([(100.0, True, False), (100.0, False, False), (110.0, False, False)])


def test_funding_reduces_long_pnl_by_expected_amount() -> None:
    signals = _long_run()
    funding = _funding([(T0 + 1 * H, 0.01)])  # notional=100*100=10000 → -100

    base = BacktestEngine().run(signals, _exec())
    with_f = BacktestEngine().run(signals, _exec(), funding)

    base_eq = base.equity_curve["equity"].to_list()[-1]
    f_eq = with_f.equity_curve["equity"].to_list()[-1]
    assert base_eq == pytest.approx(11_000.0)
    assert f_eq == pytest.approx(10_900.0)
    assert f_eq - base_eq == pytest.approx(-100.0)
    assert with_f.funding_total == pytest.approx(-100.0)
    assert with_f.trades[0].funding == pytest.approx(-100.0)
    assert with_f.trades[0].pnl == pytest.approx(1_000.0)  # gross PnL без funding


def test_funding_sign_opposite_for_long_and_short() -> None:
    funding = _funding([(T0 + 1 * H, 0.01)])
    long_sig = _long_run()
    short_sig = _signals([(100.0, False, True), (100.0, False, False), (110.0, False, False)])

    long_res = BacktestEngine().run(long_sig, _exec(), funding)
    short_res = BacktestEngine().run(short_sig, _exec(), funding)

    assert long_res.funding_total == pytest.approx(-100.0)
    assert short_res.funding_total == pytest.approx(100.0)


def test_missing_funding_is_graceful() -> None:
    signals = _long_run()
    no_arg = BacktestEngine().run(signals, _exec())
    none_arg = BacktestEngine().run(signals, _exec(), None)
    empty_arg = BacktestEngine().run(signals, _exec(), _funding([]))

    for res in (no_arg, none_arg, empty_arg):
        assert res.funding_total == 0.0
        assert res.equity_curve["equity"].to_list()[-1] == pytest.approx(11_000.0)


def test_funding_not_applied_while_flat() -> None:
    # Сигнала нет до бара 2 → на баре 0 (funding) позиция ещё не открыта.
    signals = _signals([(100.0, False, False), (100.0, True, False), (110.0, False, False)])
    funding = _funding([(T0, 0.01)])  # на баре 0, когда мы во флэте
    res = BacktestEngine().run(signals, _exec(), funding)
    assert res.funding_total == 0.0
