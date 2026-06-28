"""BT-2/BT-3 — событийный (bar-by-bar) бэктест-движок.

Потребляет DataFrame сигналов (выход стратегии: ``open_time``, ``open``,
``close``, ``long_signal``, ``short_signal``) и ``ExecutionModel``, возвращает
сделки и кривую капитала.

Fill timing (BT-3): сигнал, рассчитанный по ``close`` бара ``i``, исполняется на
баре ``i + signal_lag``. ``fill_on=NEXT_OPEN`` — по цене ``open`` бара исполнения
(анти-look-ahead, дефолт); ``fill_on=CURRENT_CLOSE`` — по ``close`` (отладка).

Границы тикета (что будет добавлено позже):
- BT-4: комиссии, слиппедж и модели размера позиции (сейчас только percent_balance);
- BT-5: маржа/ликвидация и intrabar TP/SL;
- BT-7: расчёт метрик и drawdown поверх equity_curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from trading_research.backtest.orders import OrderReason, Side
from trading_research.backtest.portfolio import Portfolio, Trade
from trading_research.domain import ExecutionModel, FillOn, PositionSizeType
from trading_research.strategies.base import LONG_SIGNAL, SHORT_SIGNAL

REQUIRED_COLUMNS = ("open_time", "close", LONG_SIGNAL, SHORT_SIGNAL)


@dataclass(frozen=True)
class BacktestResult:
    trades: list[Trade]
    equity_curve: pl.DataFrame  # columns: open_time, equity


class BacktestEngine:
    """Минимальный stop-and-reverse движок по барам."""

    def run(self, signals: pl.DataFrame, execution: ExecutionModel) -> BacktestResult:
        missing = [c for c in REQUIRED_COLUMNS if c not in signals.columns]
        if execution.fill_on is FillOn.NEXT_OPEN and "open" not in signals.columns:
            missing.append("open")
        if missing:
            raise ValueError(f"signals missing required columns: {missing}")

        times: list[datetime] = signals["open_time"].to_list()
        closes: list[float] = [float(x) for x in signals["close"].to_list()]
        opens: list[float] = (
            [float(x) for x in signals["open"].to_list()]
            if "open" in signals.columns
            else closes
        )
        longs: list[bool] = [bool(x) for x in signals[LONG_SIGNAL].to_list()]
        shorts: list[bool] = [bool(x) for x in signals[SHORT_SIGNAL].to_list()]
        n = len(times)

        wants = [self._desired_side(longs[i], shorts[i]) for i in range(n)]
        lag = execution.signal_lag

        portfolio = Portfolio(execution.initial_balance)
        equities: list[float] = []

        for j in range(n):
            # Исполнение сигнала, поданного lag баров назад, на текущем баре.
            source = j - lag
            want = wants[source] if source >= 0 else None
            if want is not None:
                fill_price = opens[j] if execution.fill_on is FillOn.NEXT_OPEN else closes[j]
                self._apply_signal(portfolio, want, fill_price, times[j], execution)
            equities.append(portfolio.equity(closes[j]))

        # Принудительно закрыть открытую позицию в конце данных.
        if portfolio.position is not None and n > 0:
            portfolio.close(closes[-1], times[-1], OrderReason.FINAL)
            equities[-1] = portfolio.balance

        equity_curve = pl.DataFrame({"open_time": times, "equity": equities})
        return BacktestResult(trades=list(portfolio.trades), equity_curve=equity_curve)

    @staticmethod
    def _desired_side(long_sig: bool, short_sig: bool) -> Side | None:
        if long_sig and not short_sig:
            return Side.LONG
        if short_sig and not long_sig:
            return Side.SHORT
        return None

    def _apply_signal(
        self,
        portfolio: Portfolio,
        want: Side,
        price: float,
        time: datetime,
        execution: ExecutionModel,
    ) -> None:
        pos = portfolio.position
        if pos is None:
            self._try_open(portfolio, want, price, time, execution)
            return
        if pos.side is want:
            return  # без пирамидинга
        if not execution.close_on_reverse_signal:
            return
        portfolio.close(price, time, OrderReason.REVERSE)
        self._try_open(portfolio, want, price, time, execution)

    def _try_open(
        self,
        portfolio: Portfolio,
        side: Side,
        price: float,
        time: datetime,
        execution: ExecutionModel,
    ) -> None:
        if side is Side.SHORT and not execution.allow_short:
            return
        qty = self._target_qty(execution, portfolio.balance, price)
        if qty <= 0:
            return
        portfolio.open(side, qty, price, time)

    @staticmethod
    def _target_qty(execution: ExecutionModel, equity: float, price: float) -> float:
        if equity <= 0 or price <= 0:
            return 0.0
        if execution.position_size_type is PositionSizeType.PERCENT_BALANCE:
            notional = equity * (execution.position_size_value / 100.0) * execution.leverage
            return notional / price
        raise NotImplementedError(
            f"position_size_type {execution.position_size_type!r} is implemented in BT-4"
        )
