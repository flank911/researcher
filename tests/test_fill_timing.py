"""Tests for BT-3 — fill timing / anti-look-ahead."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from trading_research.backtest.engine import BacktestEngine
from trading_research.backtest.orders import OrderReason, Side
from trading_research.domain import ExecutionModel, FillOn, PositionSizeType
from trading_research.strategies.base import LONG_SIGNAL, SHORT_SIGNAL


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
        "signal_lag": 1,
        "fill_on": FillOn.NEXT_OPEN,
    }
    base.update(overrides)
    return ExecutionModel(**base)  # type: ignore[arg-type]


def _signals(rows: list[tuple[float, float, bool, bool]]) -> pl.DataFrame:
    """rows = list of (open, close, long_signal, short_signal)."""
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "open_time": [t0 + timedelta(hours=i) for i in range(len(rows))],
            "open": [r[0] for r in rows],
            "close": [r[1] for r in rows],
            LONG_SIGNAL: [r[2] for r in rows],
            SHORT_SIGNAL: [r[3] for r in rows],
        }
    )


def test_entry_fills_at_next_open_not_signal_close() -> None:
    # signal on bar0 (close=100); gap up -> bar1 open=105. Fill must be 105, not 100.
    signals = _signals(
        [
            (999.0, 100.0, True, False),
            (105.0, 110.0, False, False),
        ]
    )
    result = BacktestEngine().run(signals, _exec())

    assert len(result.trades) == 1
    tr = result.trades[0]
    assert tr.side is Side.LONG
    assert tr.entry_price == 105.0  # next open, not the 100 signal close (no look-ahead)
    assert tr.exit_price == 110.0
    assert tr.reason is OrderReason.FINAL


def test_signal_on_last_bar_never_executes() -> None:
    # long signal on the last bar -> no future bar to fill -> no trade
    signals = _signals(
        [
            (100.0, 100.0, False, False),
            (100.0, 100.0, True, False),
        ]
    )
    result = BacktestEngine().run(signals, _exec())
    assert result.trades == []
    assert result.equity_curve["equity"].to_list() == [10_000.0, 10_000.0]


def test_signal_lag_two_executes_two_bars_later() -> None:
    signals = _signals(
        [
            (1.0, 100.0, True, False),  # signal here
            (200.0, 100.0, False, False),  # lag=1 would fill here (200) — must NOT
            (150.0, 160.0, False, False),  # lag=2 fills here at open=150
        ]
    )
    result = BacktestEngine().run(signals, _exec(signal_lag=2))
    assert len(result.trades) == 1
    assert result.trades[0].entry_price == 150.0


def test_next_open_requires_open_column() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    no_open = pl.DataFrame(
        {
            "open_time": [t0],
            "close": [100.0],
            LONG_SIGNAL: [True],
            SHORT_SIGNAL: [False],
        }
    )
    with pytest.raises(ValueError, match="open"):
        BacktestEngine().run(no_open, _exec())


def test_current_close_mode_fills_at_close() -> None:
    signals = _signals(
        [
            (999.0, 100.0, True, False),
            (105.0, 110.0, False, False),
        ]
    )
    result = BacktestEngine().run(signals, _exec(signal_lag=0, fill_on=FillOn.CURRENT_CLOSE))
    assert len(result.trades) == 1
    assert result.trades[0].entry_price == 100.0  # signal-bar close


def test_reverse_uses_next_open() -> None:
    # long @ bar1 open=100 (qty=100); short signal bar2 -> reverse fills bar3 open=90
    signals = _signals(
        [
            (1.0, 100.0, True, False),  # long signal
            (100.0, 120.0, False, False),  # fill long @100
            (120.0, 100.0, False, True),  # short signal
            (90.0, 95.0, False, False),  # reverse fills @ open=90
        ]
    )
    result = BacktestEngine().run(signals, _exec())
    assert len(result.trades) == 2
    long_tr, short_tr = result.trades
    assert long_tr.entry_price == 100.0
    assert long_tr.exit_price == 90.0  # closed at reverse fill (next open)
    assert long_tr.pnl == pytest.approx(100.0 * (90.0 - 100.0))
    assert short_tr.side is Side.SHORT
    assert short_tr.entry_price == 90.0
    assert short_tr.reason is OrderReason.FINAL
