"""Tests for BT-4 — fees, slippage, sizing."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from trading_research.backtest.engine import BacktestEngine
from trading_research.backtest.fees import FeeModel
from trading_research.backtest.sizing import target_qty
from trading_research.backtest.slippage import SlippageModel
from trading_research.domain import ExecutionModel, FillOn, PositionSizeType
from trading_research.strategies.base import LONG_SIGNAL, SHORT_SIGNAL


def _exec(**overrides: object) -> ExecutionModel:
    # immediate fill so prices are exactly the bar close (isolate fees/slippage/sizing)
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
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "open_time": [t0 + timedelta(hours=i) for i in range(len(rows))],
            "close": [r[0] for r in rows],
            LONG_SIGNAL: [r[1] for r in rows],
            SHORT_SIGNAL: [r[2] for r in rows],
        }
    )


# --- unit: models -----------------------------------------------------------


def test_fee_model_uses_notional() -> None:
    assert FeeModel(0.001).fee(100.0, 10.0) == pytest.approx(1.0)


def test_slippage_buy_costs_more_sell_costs_less() -> None:
    sm = SlippageModel(0.01)
    assert sm.fill_price(100.0, is_buy=True) == pytest.approx(101.0)
    assert sm.fill_price(100.0, is_buy=False) == pytest.approx(99.0)


def test_sizing_percent_and_fixed() -> None:
    pct = _exec(position_size_type=PositionSizeType.PERCENT_BALANCE, position_size_value=20.0)
    assert target_qty(pct, balance=10_000.0, price=100.0) == pytest.approx(20.0)

    fixed = _exec(position_size_type=PositionSizeType.FIXED, position_size_value=5_000.0)
    assert target_qty(fixed, balance=10_000.0, price=100.0) == pytest.approx(50.0)


def test_sizing_applies_leverage() -> None:
    pct = _exec(position_size_value=100.0, leverage=3)
    assert target_qty(pct, balance=10_000.0, price=100.0) == pytest.approx(300.0)


def test_risk_based_requires_stop_loss() -> None:
    rb = _exec(position_size_type=PositionSizeType.RISK_BASED, position_size_value=1.0)
    with pytest.raises(ValueError, match="stop_loss_pct"):
        target_qty(rb, balance=10_000.0, price=100.0)


def test_risk_based_sizes_to_risk_budget() -> None:
    # риск 1% от 10_000 = 100; стоп 2% от 100 = 2 на единицу → qty = 100/2 = 50
    rb = _exec(
        position_size_type=PositionSizeType.RISK_BASED,
        position_size_value=1.0,
        stop_loss_pct=0.02,
        leverage=10,  # экспозиция 50*100=5000 < 10*10000, кап не бьёт
    )
    assert target_qty(rb, balance=10_000.0, price=100.0) == pytest.approx(50.0)


def test_risk_based_capped_by_leverage() -> None:
    # риск 5% → qty = 500/(100*0.001)=... большой; кап: 1*10000/100 = 100 единиц
    rb = _exec(
        position_size_type=PositionSizeType.RISK_BASED,
        position_size_value=5.0,
        stop_loss_pct=0.001,
        leverage=1,
    )
    assert target_qty(rb, balance=10_000.0, price=100.0) == pytest.approx(100.0)


# --- integration: engine ----------------------------------------------------


def test_commission_reduces_balance_and_records_fees() -> None:
    signals = _signals([(100.0, True, False), (110.0, False, False)])
    result = BacktestEngine().run(signals, _exec(commission_rate=0.001))

    tr = result.trades[0]
    # qty = 10000/100 = 100; entry fee = 100*100*0.001 = 10; exit fee = 110*100*0.001 = 11
    assert tr.qty == pytest.approx(100.0)
    assert tr.pnl == pytest.approx(1000.0)  # gross
    assert tr.fees == pytest.approx(21.0)
    # balance = 10000 - 10 + 1000 - 11 = 10979
    assert result.equity_curve["equity"].to_list()[-1] == pytest.approx(10_979.0)


def test_slippage_worsens_entry_and_exit() -> None:
    signals = _signals([(100.0, True, False), (110.0, False, False)])
    result = BacktestEngine().run(signals, _exec(slippage_pct=0.01))

    tr = result.trades[0]
    assert tr.entry_price == pytest.approx(101.0)  # buy slips up
    assert tr.exit_price == pytest.approx(108.9)  # sell slips down (110 * 0.99)


def test_fixed_sizing_end_to_end() -> None:
    signals = _signals([(100.0, True, False), (110.0, False, False)])
    result = BacktestEngine().run(
        signals, _exec(position_size_type=PositionSizeType.FIXED, position_size_value=5_000.0)
    )
    tr = result.trades[0]
    assert tr.qty == pytest.approx(50.0)
    assert tr.pnl == pytest.approx(500.0)


def test_zero_costs_match_no_fees() -> None:
    signals = _signals([(100.0, True, False), (110.0, False, False)])
    result = BacktestEngine().run(signals, _exec())
    tr = result.trades[0]
    assert tr.fees == 0.0
    assert result.equity_curve["equity"].to_list()[-1] == pytest.approx(11_000.0)
