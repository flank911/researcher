"""Tests for WF-2 — walk-forward оркестратор (out-of-sample)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
import pytest

from trading_research.domain import ExecutionModel, FillOn, PositionSizeType
from trading_research.experiments.time_splitters import generate_walk_forward_folds
from trading_research.experiments.walk_forward import run_walk_forward
from trading_research.strategies.base import LONG_SIGNAL, SHORT_SIGNAL

START = datetime(2024, 1, 1, tzinfo=UTC)
HOUR = timedelta(hours=1)
DAY = timedelta(days=1)


class _HoldStrategy:
    """Вход на первом баре окна и удержание: param ``side`` = long/short.

    PnL за окно = направление * доходность цены окна (buy/sell-and-hold).
    """

    name = "hold"

    def warmup_bars(self, params: Mapping[str, Any]) -> int:
        return 0

    def generate_signals(self, data: pl.DataFrame, params: Mapping[str, Any]) -> pl.DataFrame:
        n = data.height
        is_long = params["side"] == "long"
        longs = [i == 0 and is_long for i in range(n)]
        shorts = [i == 0 and not is_long for i in range(n)]
        return data.with_columns(pl.Series(LONG_SIGNAL, longs), pl.Series(SHORT_SIGNAL, shorts))


def _exec(**kw: object) -> ExecutionModel:
    base: dict[str, object] = {
        "initial_balance": 10_000.0,
        "commission_rate": 0.0,
        "slippage_pct": 0.0,
        "leverage": 1,
        "position_size_type": PositionSizeType.PERCENT_BALANCE,
        "position_size_value": 100.0,
        "allow_short": True,
        "signal_lag": 0,
        "fill_on": FillOn.CURRENT_CLOSE,
    }
    base.update(kw)
    return ExecutionModel(**base)  # type: ignore[arg-type]


def _data(closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open_time": [START + i * HOUR for i in range(len(closes))],
            "close": closes,
        }
    )


SIDES = [{"side": "long"}, {"side": "short"}]


# --- no leakage -------------------------------------------------------------


def test_selection_uses_only_train_no_leakage() -> None:
    # Train (бары 0..71) растёт → long лучший; test (72..119) падает → short лучший.
    train_up = [100.0 + i for i in range(72)]  # 100 → 171
    test_down = [171.0 - (i + 1) for i in range(48)]  # 170 → 123
    data = _data(train_up + test_down)

    folds = generate_walk_forward_folds(
        START, START + 5 * DAY, train=3 * DAY, test=2 * DAY, step=10 * DAY
    )
    assert len(folds) == 1

    result = run_walk_forward(data, _HoldStrategy(), SIDES, folds, _exec(), "1h", min_trades=0)

    assert len(result.selections) == 1
    sel = result.selections[0]
    # На train выбран long (растущий), НЕ short — хотя на test выгоднее short.
    assert sel.params == {"side": "long"}
    assert sel.train_score > 0
    # Торговали long на падающем test → OOS в минусе. Будь утечка — был бы плюс.
    assert result.oos_metrics.total_return_pct < 0
    assert result.oos_equity.height == 48  # бары test-окна


# --- continuous OOS with balance carry --------------------------------------


def test_continuous_oos_equity_with_carry() -> None:
    # Монотонно растущая цена → long выигрывает на каждом фолде.
    closes = [100.0 + i for i in range(120)]  # 5 дней по 24 бара
    data = _data(closes)
    folds = generate_walk_forward_folds(
        START, START + 5 * DAY, train=2 * DAY, test=1 * DAY, step=1 * DAY
    )
    assert len(folds) == 3

    result = run_walk_forward(data, _HoldStrategy(), SIDES, folds, _exec(), "1h", min_trades=0)

    assert len(result.selections) == 3
    assert all(s.params == {"side": "long"} for s in result.selections)
    # Непрерывная OOS-кривая = склейка 3 test-окон по 24 бара.
    assert result.oos_equity.height == 72

    eq = result.oos_equity["equity"].to_list()
    # Перенос баланса: стартует с initial_balance и монотонно растёт (long на росте).
    assert eq[0] == pytest.approx(10_000.0, rel=1e-6)
    assert eq[-1] > eq[0]
    # На стыке фолдов капитал непрерывен (новый фолд стартует с накопленного).
    assert eq[24] == pytest.approx(eq[23], rel=0.02)
    assert result.oos_metrics.total_return_pct > 0


# --- validation -------------------------------------------------------------


def test_invalid_args_raise() -> None:
    data = _data([100.0 + i for i in range(120)])
    folds = generate_walk_forward_folds(
        START, START + 5 * DAY, train=2 * DAY, test=1 * DAY, step=1 * DAY
    )
    with pytest.raises(ValueError, match="select_by"):
        run_walk_forward(data, _HoldStrategy(), SIDES, folds, _exec(), "1h", select_by="reliable")
    with pytest.raises(ValueError, match="param_combos"):
        run_walk_forward(data, _HoldStrategy(), [], folds, _exec(), "1h")


def test_empty_folds_give_flat_oos() -> None:
    data = _data([100.0 + i for i in range(48)])
    result = run_walk_forward(data, _HoldStrategy(), SIDES, [], _exec(), "1h", min_trades=0)
    assert result.selections == []
    assert result.oos_equity.height == 0
    assert result.oos_metrics.total_return_pct == pytest.approx(0.0)
