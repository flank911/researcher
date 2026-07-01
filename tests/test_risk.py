"""Tests for BT-5 — risk layer: margin/liquidation and intrabar TP/SL/trailing."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from trading_research.backtest.engine import BacktestEngine
from trading_research.backtest.orders import OrderReason, Side
from trading_research.backtest.position import Position
from trading_research.backtest.risk import RiskManager
from trading_research.domain import ExecutionModel, FillOn, PositionSizeType
from trading_research.strategies.base import LONG_SIGNAL, SHORT_SIGNAL


def _exec(**overrides: object) -> ExecutionModel:
    # Вход по open следующего бара (lag=1): high/low того же бара валидны intrabar.
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


def _signals(rows: list[tuple[float, float, float, float, bool, bool]]) -> pl.DataFrame:
    """rows = list of (open, high, low, close, long_signal, short_signal)."""
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "open_time": [t0 + timedelta(hours=i) for i in range(len(rows))],
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            LONG_SIGNAL: [r[4] for r in rows],
            SHORT_SIGNAL: [r[5] for r in rows],
        }
    )


def _long(entry: float = 100.0, qty: float = 100.0) -> Position:
    return Position(side=Side.LONG, qty=qty, entry_price=entry, entry_time=datetime(2024, 1, 1))


def _short(entry: float = 100.0, qty: float = 100.0) -> Position:
    return Position(side=Side.SHORT, qty=qty, entry_price=entry, entry_time=datetime(2024, 1, 1))


# --- unit: RiskManager ------------------------------------------------------


def test_liquidation_price_long_hand_calc() -> None:
    # qty=1000, entry=100, balance=10000, mm=0.005
    # P = (1000*100 - 10000) / (1000*0.995) = 90000/995
    rm = RiskManager(_exec(maintenance_margin_rate=0.005))
    assert rm.liquidation_price(_long(qty=1000.0), balance=10_000.0) == pytest.approx(90000 / 995)


def test_liquidation_price_short_hand_calc() -> None:
    # P = (10000 + 1000*100) / (1000*1.005) = 110000/1005
    rm = RiskManager(_exec(maintenance_margin_rate=0.005))
    assert rm.liquidation_price(_short(qty=1000.0), balance=10_000.0) == pytest.approx(
        110000 / 1005
    )


def test_evaluate_tp_only() -> None:
    rm = RiskManager(_exec(take_profit_pct=0.10))
    event, _ = rm.evaluate(_long(), 10_000.0, 100.0, high=112.0, low=99.0, bar_open=100.0)
    assert event is not None
    assert event.reason is OrderReason.TAKE_PROFIT
    assert event.price == pytest.approx(110.0)


def test_evaluate_sl_only() -> None:
    rm = RiskManager(_exec(stop_loss_pct=0.05))
    event, _ = rm.evaluate(_long(), 10_000.0, 100.0, high=101.0, low=94.0, bar_open=100.0)
    assert event is not None
    assert event.reason is OrderReason.STOP_LOSS
    assert event.price == pytest.approx(95.0)


def test_evaluate_tp_and_sl_same_bar_prefers_sl() -> None:
    # Оба уровня задеты на одном баре → пессимистично срабатывает SL.
    rm = RiskManager(_exec(stop_loss_pct=0.05, take_profit_pct=0.05))
    event, _ = rm.evaluate(_long(), 10_000.0, 100.0, high=106.0, low=94.0, bar_open=100.0)
    assert event is not None
    assert event.reason is OrderReason.STOP_LOSS
    assert event.price == pytest.approx(95.0)


def test_evaluate_gap_through_stop_fills_at_open() -> None:
    # Бар открылся ниже SL (гэп) → исполнение по open, а не по уровню стопа.
    rm = RiskManager(_exec(stop_loss_pct=0.05))
    event, _ = rm.evaluate(_long(), 10_000.0, 100.0, high=94.0, low=90.0, bar_open=93.0)
    assert event is not None
    assert event.reason is OrderReason.STOP_LOSS
    assert event.price == pytest.approx(93.0)


def test_evaluate_trailing_uses_prebar_extreme() -> None:
    # extreme=120 → trail=108; текущий high (200) не защищает от текущего low.
    rm = RiskManager(_exec(trailing_stop_pct=0.10))
    event, _ = rm.evaluate(_long(), 10_000.0, 120.0, high=200.0, low=107.0, bar_open=118.0)
    assert event is not None
    assert event.reason is OrderReason.TRAILING_STOP
    assert event.price == pytest.approx(108.0)


def test_evaluate_updates_extreme_when_no_exit() -> None:
    rm = RiskManager(_exec(trailing_stop_pct=0.50))
    event, new_extreme = rm.evaluate(_long(), 10_000.0, 100.0, high=130.0, low=99.0, bar_open=100.0)
    assert event is None
    assert new_extreme == pytest.approx(130.0)


# --- integration: engine ----------------------------------------------------


def test_intrabar_take_profit() -> None:
    signals = _signals(
        [
            (100.0, 100.0, 100.0, 100.0, True, False),  # long signal
            (100.0, 112.0, 99.0, 105.0, False, False),  # entry@open=100, TP@110
        ]
    )
    result = BacktestEngine().run(signals, _exec(take_profit_pct=0.10))
    assert len(result.trades) == 1
    tr = result.trades[0]
    assert tr.reason is OrderReason.TAKE_PROFIT
    assert tr.exit_price == pytest.approx(110.0)
    assert tr.pnl == pytest.approx(1000.0)  # qty=100 * (110-100)


def test_intrabar_stop_loss() -> None:
    signals = _signals(
        [
            (100.0, 100.0, 100.0, 100.0, True, False),
            (100.0, 101.0, 94.0, 96.0, False, False),  # SL@95 breached (low=94)
        ]
    )
    result = BacktestEngine().run(signals, _exec(stop_loss_pct=0.05))
    tr = result.trades[0]
    assert tr.reason is OrderReason.STOP_LOSS
    assert tr.exit_price == pytest.approx(95.0)
    assert tr.pnl == pytest.approx(-500.0)


def test_intrabar_tp_and_sl_prefers_sl() -> None:
    # DoD: одновременное задевание TP и SL → срабатывает SL (пессимистично).
    signals = _signals(
        [
            (100.0, 100.0, 100.0, 100.0, True, False),
            (100.0, 106.0, 94.0, 100.0, False, False),  # TP@105 и SL@95 оба задеты
        ]
    )
    result = BacktestEngine().run(signals, _exec(stop_loss_pct=0.05, take_profit_pct=0.05))
    tr = result.trades[0]
    assert tr.reason is OrderReason.STOP_LOSS
    assert tr.exit_price == pytest.approx(95.0)


def test_trailing_stop_tracks_path_extremum() -> None:
    # DoD: trailing следует за экстремумом пути (120), а не за ценой входа (100).
    signals = _signals(
        [
            (100.0, 100.0, 100.0, 100.0, True, False),  # long signal
            (100.0, 120.0, 100.0, 118.0, False, False),  # entry@100; extreme→120; trail@90 ok
            (118.0, 130.0, 107.0, 110.0, False, False),  # trail@120*0.9=108; low=107 → стоп@108
        ]
    )
    result = BacktestEngine().run(signals, _exec(trailing_stop_pct=0.10))
    assert len(result.trades) == 1
    tr = result.trades[0]
    assert tr.reason is OrderReason.TRAILING_STOP
    assert tr.exit_price == pytest.approx(108.0)
    assert tr.pnl == pytest.approx(800.0)  # 100 * (108-100)


def test_liquidation_caps_loss_and_keeps_equity_positive() -> None:
    # DoD: ликвидация при пробое маржи; equity не уходит в нереалистичный минус.
    signals = _signals(
        [
            (100.0, 100.0, 100.0, 100.0, True, False),
            (100.0, 101.0, 90.0, 95.0, False, False),  # low=90 пробивает liq≈90.45
        ]
    )
    result = BacktestEngine().run(signals, _exec(leverage=10))
    tr = result.trades[0]
    assert tr.reason is OrderReason.LIQUIDATION
    # liq_price = (1000*100 - 10000)/(1000*0.995) = 90000/995
    assert tr.exit_price == pytest.approx(90000 / 995)
    final_equity = result.equity_curve["equity"].to_list()[-1]
    # Без ликвидации закрытие по low=90 дало бы убыток -10000 (баланс 0).
    assert final_equity > 0.0
    assert final_equity == pytest.approx(10_000.0 + 1000.0 * (90000 / 995 - 100.0))


def test_risk_based_sizing_end_to_end() -> None:
    # риск 1% (=100) при стопе 2% → qty=50; выход по SL@98 → pnl=-100 (ровно 1%).
    signals = _signals(
        [
            (100.0, 100.0, 100.0, 100.0, True, False),
            (100.0, 101.0, 97.0, 99.0, False, False),  # SL@98, low=97
        ]
    )
    result = BacktestEngine().run(
        signals,
        _exec(
            position_size_type=PositionSizeType.RISK_BASED,
            position_size_value=1.0,
            stop_loss_pct=0.02,
        ),
    )
    tr = result.trades[0]
    assert tr.qty == pytest.approx(50.0)
    assert tr.reason is OrderReason.STOP_LOSS
    assert tr.pnl == pytest.approx(-100.0)


def test_intrabar_exit_applies_slippage() -> None:
    # Вход по open=100 проскальзывает вверх → entry=101; TP=101*1.10=111.1;
    # выход — рыночная продажа, слиппедж вниз: 111.1*0.99=109.989.
    signals = _signals(
        [
            (100.0, 100.0, 100.0, 100.0, True, False),
            (100.0, 112.0, 99.0, 105.0, False, False),
        ]
    )
    result = BacktestEngine().run(signals, _exec(take_profit_pct=0.10, slippage_pct=0.01))
    tr = result.trades[0]
    assert tr.reason is OrderReason.TAKE_PROFIT
    assert tr.entry_price == pytest.approx(101.0)
    assert tr.exit_price == pytest.approx(109.989)  # 111.1 * 0.99 — слиппедж на выходе


def test_short_intrabar_take_profit() -> None:
    signals = _signals(
        [
            (100.0, 100.0, 100.0, 100.0, False, True),  # short signal
            (100.0, 101.0, 94.0, 96.0, False, False),  # TP short @95 (low=94)
        ]
    )
    result = BacktestEngine().run(signals, _exec(take_profit_pct=0.05))
    tr = result.trades[0]
    assert tr.side is Side.SHORT
    assert tr.reason is OrderReason.TAKE_PROFIT
    assert tr.exit_price == pytest.approx(95.0)
    assert tr.pnl == pytest.approx(500.0)  # -1 * 100 * (95-100)


def test_short_intrabar_stop_loss() -> None:
    signals = _signals(
        [
            (100.0, 100.0, 100.0, 100.0, False, True),
            (100.0, 106.0, 99.0, 102.0, False, False),  # SL short @105 (high=106)
        ]
    )
    result = BacktestEngine().run(signals, _exec(stop_loss_pct=0.05))
    tr = result.trades[0]
    assert tr.side is Side.SHORT
    assert tr.reason is OrderReason.STOP_LOSS
    assert tr.exit_price == pytest.approx(105.0)
    assert tr.pnl == pytest.approx(-500.0)


def test_risk_stops_require_high_low_columns() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    no_hl = pl.DataFrame(
        {
            "open_time": [t0],
            "open": [100.0],
            "close": [100.0],
            LONG_SIGNAL: [True],
            SHORT_SIGNAL: [False],
        }
    )
    with pytest.raises(ValueError, match="high|low"):
        BacktestEngine().run(no_hl, _exec(stop_loss_pct=0.05))
