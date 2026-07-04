"""BT-8 — golden test: сверка нашего движка с ``backtesting.py``.

Прогоняем MA Cross на одном синтетическом датасете в нашем движке и в
``backtesting.py`` и сверяем сделки: сторону, времена входа/выхода, цены и
per-trade доходность. Абсолютный PnL не сверяется — ``backtesting.py`` торгует
целыми юнитами, наш движок допускает дробный объём; per-trade доходность (%) от
размера не зависит и потому инвариантна.

Задокументированные расхождения и допуски — см. ``tickets/BT-8-reconciliation.md``.
Тест **пропускается**, если ``backtesting`` не установлен (внешняя зависимость;
ставится через ``pip install -e ".[golden]"``).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

backtesting = pytest.importorskip("backtesting")

import pandas as pd  # noqa: E402  (доступен вместе с backtesting)
from backtesting import Backtest, Strategy  # noqa: E402
from backtesting.lib import crossover  # noqa: E402

from trading_research.backtest.engine import BacktestEngine  # noqa: E402
from trading_research.domain import ExecutionModel, FillOn, PositionSizeType  # noqa: E402
from trading_research.strategies.ma_cross import MovingAverageCrossStrategy  # noqa: E402

N_BARS = 40
FAST = 3
SLOW = 7
INITIAL_CASH = 100_000.0

# Допуски (см. отчёт о сверке).
PRICE_RTOL = 1e-6
PNL_TOL_PCT = 1e-3  # процентных пунктов на сделку для полностью совпадающих сделок
FINAL_PNL_TOL_PCT = 0.5  # финальная сделка: разная цена принудительного закрытия


def _ohlc() -> tuple[list[datetime], list[float], list[float], list[float], list[float]]:
    """Детерминированный осциллирующий ряд с несколькими пересечениями MA."""
    close = [100.0 + 8.0 * math.sin(i / 3.0) for i in range(N_BARS)]
    open_ = [c - 0.1 for c in close]
    high = [max(o, c) + 0.5 for o, c in zip(open_, close, strict=True)]
    low = [min(o, c) - 0.5 for o, c in zip(open_, close, strict=True)]
    times = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(N_BARS)]
    return times, open_, high, low, close


class _Trade:
    __slots__ = ("side", "entry_time", "exit_time", "entry_price", "exit_price", "pnl_pct")

    def __init__(
        self,
        side: str,
        entry_time: datetime,
        exit_time: datetime,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
    ) -> None:
        self.side = side
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.pnl_pct = pnl_pct


def _run_ours(
    times: list[datetime],
    open_: list[float],
    high: list[float],
    low: list[float],
    close: list[float],
) -> list[_Trade]:
    data = pl.DataFrame(
        {
            "open_time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1.0] * len(times),
        }
    )
    signals = MovingAverageCrossStrategy().generate_signals(
        data, {"fast_ma": FAST, "slow_ma": SLOW}
    )
    execution = ExecutionModel(
        initial_balance=INITIAL_CASH,
        commission_rate=0.0,
        slippage_pct=0.0,
        leverage=1,
        position_size_type=PositionSizeType.PERCENT_BALANCE,
        position_size_value=100.0,
        allow_short=True,
        close_on_reverse_signal=True,
        signal_lag=1,
        fill_on=FillOn.NEXT_OPEN,
    )
    result = BacktestEngine().run(signals, execution)
    return [
        _Trade(t.side.value, t.entry_time, t.exit_time, t.entry_price, t.exit_price, t.pnl_pct)
        for t in result.trades
    ]


def _sma(values: object, window: int) -> object:
    return pd.Series(values).rolling(window).mean().values


class _MaCrossStrategy(Strategy):
    def init(self) -> None:
        self.fast = self.I(_sma, self.data.Close, FAST)
        self.slow = self.I(_sma, self.data.Close, SLOW)

    def next(self) -> None:
        # exclusive_orders=True → новый ордер закрывает встречную позицию (reverse).
        if crossover(self.fast, self.slow):
            self.buy()
        elif crossover(self.slow, self.fast):
            self.sell()


def _run_backtesting(
    times: list[datetime],
    open_: list[float],
    high: list[float],
    low: list[float],
    close: list[float],
) -> list[_Trade]:
    frame = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close},
        index=pd.DatetimeIndex(times),
    )
    stats = Backtest(
        frame,
        _MaCrossStrategy,
        cash=INITIAL_CASH,
        commission=0.0,
        trade_on_close=False,  # исполнение по open следующего бара (как наш NEXT_OPEN)
        exclusive_orders=True,  # stop-and-reverse
        finalize_trades=True,  # закрыть открытую позицию в конце данных
    ).run()
    trades: list[_Trade] = []
    for row in stats["_trades"].itertuples():
        side = "long" if row.Size > 0 else "short"
        trades.append(
            _Trade(
                side,
                row.EntryTime.to_pydatetime(),
                row.ExitTime.to_pydatetime(),
                float(row.EntryPrice),
                float(row.ExitPrice),
                float(row.ReturnPct) * 100.0,
            )
        )
    return trades


@pytest.fixture(scope="module")
def golden() -> tuple[list[_Trade], list[_Trade], list[float], list[float]]:
    times, open_, high, low, close = _ohlc()
    ours = _run_ours(times, open_, high, low, close)
    theirs = _run_backtesting(times, open_, high, low, close)
    return ours, theirs, open_, close


def test_same_number_of_trades(
    golden: tuple[list[_Trade], list[_Trade], list[float], list[float]],
) -> None:
    ours, theirs, _, _ = golden
    assert len(ours) == len(theirs) > 0


def test_trade_structure_matches(
    golden: tuple[list[_Trade], list[_Trade], list[float], list[float]],
) -> None:
    ours, theirs, _, _ = golden
    for a, b in zip(ours, theirs, strict=True):
        assert a.side == b.side
        assert a.entry_time == b.entry_time
        assert a.exit_time == b.exit_time
        # Вход по open бара исполнения — цена совпадает в обоих движках.
        assert a.entry_price == pytest.approx(b.entry_price, rel=PRICE_RTOL)


def test_non_final_trades_match_exactly(
    golden: tuple[list[_Trade], list[_Trade], list[float], list[float]],
) -> None:
    ours, theirs, _, _ = golden
    for a, b in zip(ours[:-1], theirs[:-1], strict=True):
        assert a.exit_price == pytest.approx(b.exit_price, rel=PRICE_RTOL)
        assert a.pnl_pct == pytest.approx(b.pnl_pct, abs=PNL_TOL_PCT)


def test_final_trade_divergence_is_forced_close_convention(
    golden: tuple[list[_Trade], list[_Trade], list[float], list[float]],
) -> None:
    ours, theirs, open_, close = golden
    a, b = ours[-1], theirs[-1]
    # Единственное расхождение — цена принудительного закрытия в конце данных:
    # наш движок закрывает по close последнего бара, backtesting.py — по его open.
    assert a.exit_price == pytest.approx(close[-1], rel=PRICE_RTOL)
    assert b.exit_price == pytest.approx(open_[-1], rel=PRICE_RTOL)
    assert a.pnl_pct == pytest.approx(b.pnl_pct, abs=FINAL_PNL_TOL_PCT)


def test_total_return_within_tolerance(
    golden: tuple[list[_Trade], list[_Trade], list[float], list[float]],
) -> None:
    ours, theirs, _, _ = golden
    ours_sum = sum(t.pnl_pct for t in ours)
    theirs_sum = sum(t.pnl_pct for t in theirs)
    # Совокупное расхождение ограничено единственной финальной сделкой.
    assert ours_sum == pytest.approx(theirs_sum, abs=FINAL_PNL_TOL_PCT)
