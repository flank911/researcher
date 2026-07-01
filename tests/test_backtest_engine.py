"""Tests for BT-2 — event-driven backtest engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from trading_research.backtest.engine import BacktestEngine
from trading_research.backtest.orders import OrderReason, Side
from trading_research.domain import ExecutionModel, FillOn, PositionSizeType
from trading_research.strategies.base import LONG_SIGNAL, SHORT_SIGNAL


def _exec(**overrides: object) -> ExecutionModel:
    # BT-2 mechanics tested with immediate fill (close, no lag); fill timing — BT-3.
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
    """rows = list of (close, long_signal, short_signal)."""
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "open_time": [t0 + timedelta(hours=i) for i in range(len(rows))],
            "close": [r[0] for r in rows],
            LONG_SIGNAL: [r[1] for r in rows],
            SHORT_SIGNAL: [r[2] for r in rows],
        }
    )


def test_single_long_trade_hand_calc() -> None:
    # enter long @100 (qty=100), forced close @110 -> pnl = +1000
    signals = _signals([(100.0, True, False), (110.0, False, False)])
    result = BacktestEngine().run(signals, _exec())

    assert len(result.trades) == 1
    tr = result.trades[0]
    assert tr.side is Side.LONG
    assert tr.entry_price == 100.0
    assert tr.exit_price == 110.0
    assert tr.qty == pytest.approx(100.0)
    assert tr.pnl == pytest.approx(1000.0)
    assert tr.pnl_pct == pytest.approx(10.0)
    assert tr.reason is OrderReason.FINAL
    assert result.equity_curve["equity"].to_list() == pytest.approx([10_000.0, 11_000.0])


def test_reverse_long_to_short_hand_calc() -> None:
    # bar0 long@100 (qty=100); bar1 no signal (mark @110);
    # bar2 short signal -> close long@90 (pnl=-1000, bal=9000), open short qty=9000/90=100;
    # bar3 final close short@95 (pnl=100*(90-95)=-500, bal=8500)
    signals = _signals(
        [
            (100.0, True, False),
            (110.0, False, False),
            (90.0, False, True),
            (95.0, False, False),
        ]
    )
    result = BacktestEngine().run(signals, _exec())

    assert len(result.trades) == 2
    long_tr, short_tr = result.trades
    assert long_tr.side is Side.LONG
    assert long_tr.pnl == pytest.approx(-1000.0)
    assert long_tr.reason is OrderReason.REVERSE

    assert short_tr.side is Side.SHORT
    assert short_tr.entry_price == 90.0
    assert short_tr.qty == pytest.approx(100.0)
    assert short_tr.pnl == pytest.approx(-500.0)
    assert short_tr.reason is OrderReason.FINAL

    equity = result.equity_curve["equity"].to_list()
    assert equity == pytest.approx([10_000.0, 11_000.0, 9_000.0, 8_500.0])


def test_short_disabled_closes_to_flat() -> None:
    # long@100, short signal closes to flat (no short opened), stays flat afterwards
    signals = _signals(
        [
            (100.0, True, False),
            (120.0, False, True),
            (130.0, False, False),
        ]
    )
    result = BacktestEngine().run(signals, _exec(allow_short=False))

    assert len(result.trades) == 1
    tr = result.trades[0]
    assert tr.side is Side.LONG
    assert tr.exit_price == 120.0
    assert tr.pnl == pytest.approx(2000.0)
    assert tr.reason is OrderReason.REVERSE
    # после выхода во флэт equity не меняется на последнем баре
    assert result.equity_curve["equity"].to_list() == pytest.approx([10_000.0, 12_000.0, 12_000.0])


def test_no_pyramiding_on_repeated_signal() -> None:
    signals = _signals(
        [
            (100.0, True, False),
            (105.0, True, False),  # повторный long — игнор
            (110.0, False, False),
        ]
    )
    result = BacktestEngine().run(signals, _exec())
    assert len(result.trades) == 1
    assert result.trades[0].entry_price == 100.0  # вход только по первому сигналу


def test_no_reverse_when_flag_disabled() -> None:
    signals = _signals(
        [
            (100.0, True, False),
            (90.0, False, True),  # reverse выключен -> игнор, остаёмся long
            (120.0, False, False),
        ]
    )
    result = BacktestEngine().run(signals, _exec(close_on_reverse_signal=False))
    assert len(result.trades) == 1
    tr = result.trades[0]
    assert tr.side is Side.LONG
    assert tr.exit_price == 120.0
    assert tr.reason is OrderReason.FINAL


def test_leverage_scales_pnl() -> None:
    signals = _signals([(100.0, True, False), (110.0, False, False)])
    result = BacktestEngine().run(signals, _exec(leverage=3))
    assert result.trades[0].pnl == pytest.approx(3000.0)


def test_missing_columns_raise() -> None:
    bad = pl.DataFrame({"open_time": [datetime(2024, 1, 1, tzinfo=UTC)], "close": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        BacktestEngine().run(bad, _exec())


def test_risk_based_without_stop_loss_raises() -> None:
    signals = _signals([(100.0, True, False), (110.0, False, False)])
    with pytest.raises(ValueError, match="stop_loss_pct"):
        BacktestEngine().run(signals, _exec(position_size_type=PositionSizeType.RISK_BASED))
