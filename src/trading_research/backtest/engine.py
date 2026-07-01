"""BT-2/BT-3 — событийный (bar-by-bar) бэктест-движок.

Потребляет DataFrame сигналов (выход стратегии: ``open_time``, ``open``,
``close``, ``long_signal``, ``short_signal``) и ``ExecutionModel``, возвращает
сделки и кривую капитала.

Fill timing (BT-3): сигнал, рассчитанный по ``close`` бара ``i``, исполняется на
баре ``i + signal_lag``. ``fill_on=NEXT_OPEN`` — по цене ``open`` бара исполнения
(анти-look-ahead, дефолт); ``fill_on=CURRENT_CLOSE`` — по ``close`` (отладка).

Risk (BT-5): защитные выходы (TP/SL/trailing) и ликвидация срабатывают внутри бара
по ``high``/``low`` — после исполнения сигнала на ``open`` и до отметки капитала на
``close``. Пессимистичные допущения (порядок хуже→лучше, SL раньше TP, гэп сквозь
уровень) описаны в ``backtest/risk.py``. Внутрибарная оценка требует колонок
``high``/``low``; при заданных TP/SL/trailing их отсутствие — ошибка.

Границы тикета (что будет добавлено позже):
- BT-6: funding-платежи в PnL для futures;
- BT-7: расчёт метрик и drawdown поверх equity_curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from trading_research.backtest.fees import FeeModel
from trading_research.backtest.orders import OrderReason, Side
from trading_research.backtest.portfolio import Portfolio, Trade
from trading_research.backtest.risk import RiskManager
from trading_research.backtest.sizing import target_qty
from trading_research.backtest.slippage import SlippageModel
from trading_research.domain import ExecutionModel, FillOn
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

        risk = RiskManager(execution)
        has_hl = "high" in signals.columns and "low" in signals.columns
        if risk.enabled and not has_hl:
            # Внутрибарные TP/SL/trailing невозможны без экстремумов бара.
            missing.extend(c for c in ("high", "low") if c not in signals.columns)
        if missing:
            raise ValueError(f"signals missing required columns: {missing}")

        times: list[datetime] = signals["open_time"].to_list()
        closes: list[float] = [float(x) for x in signals["close"].to_list()]
        opens: list[float] = (
            [float(x) for x in signals["open"].to_list()]
            if "open" in signals.columns
            else closes
        )
        highs: list[float] = (
            [float(x) for x in signals["high"].to_list()] if has_hl else closes
        )
        lows: list[float] = (
            [float(x) for x in signals["low"].to_list()] if has_hl else closes
        )
        longs: list[bool] = [bool(x) for x in signals[LONG_SIGNAL].to_list()]
        shorts: list[bool] = [bool(x) for x in signals[SHORT_SIGNAL].to_list()]
        n = len(times)

        wants = [self._desired_side(longs[i], shorts[i]) for i in range(n)]
        lag = execution.signal_lag
        fee_model = FeeModel(execution.commission_rate)
        slip_model = SlippageModel(execution.slippage_pct)

        portfolio = Portfolio(execution.initial_balance)
        equities: list[float] = []
        tracked = portfolio.position  # для инициализации экстремума при новой позиции
        extreme = 0.0

        for j in range(n):
            # 1) Исполнение сигнала (на open/close бара) — хронологически первым.
            source = j - lag
            want = wants[source] if source >= 0 else None
            if want is not None:
                raw_price = opens[j] if execution.fill_on is FillOn.NEXT_OPEN else closes[j]
                self._apply_signal(
                    portfolio, want, raw_price, times[j], execution, fee_model, slip_model
                )

            # Новая позиция → переинициализировать экстремум пути для trailing.
            if portfolio.position is not tracked:
                tracked = portfolio.position
                if tracked is not None:
                    extreme = risk.initial_extreme(tracked)

            # 2) Внутрибарные защитные выходы и ликвидация (после open, до close).
            if has_hl and portfolio.position is not None:
                event, extreme = risk.evaluate(
                    portfolio.position,
                    portfolio.balance,
                    extreme,
                    high=highs[j],
                    low=lows[j],
                    bar_open=opens[j],
                )
                if event is not None:
                    self._close(
                        portfolio, event.price, times[j], event.reason, fee_model, slip_model
                    )
                    tracked = portfolio.position

            equities.append(portfolio.equity(closes[j]))

        # Принудительно закрыть открытую позицию в конце данных.
        if portfolio.position is not None and n > 0:
            self._close(portfolio, closes[-1], times[-1], OrderReason.FINAL, fee_model, slip_model)
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
        raw_price: float,
        time: datetime,
        execution: ExecutionModel,
        fee_model: FeeModel,
        slip_model: SlippageModel,
    ) -> None:
        pos = portfolio.position
        if pos is None:
            self._try_open(portfolio, want, raw_price, time, execution, fee_model, slip_model)
            return
        if pos.side is want:
            return  # без пирамидинга
        if not execution.close_on_reverse_signal:
            return
        self._close(portfolio, raw_price, time, OrderReason.REVERSE, fee_model, slip_model)
        self._try_open(portfolio, want, raw_price, time, execution, fee_model, slip_model)

    @staticmethod
    def _try_open(
        portfolio: Portfolio,
        side: Side,
        raw_price: float,
        time: datetime,
        execution: ExecutionModel,
        fee_model: FeeModel,
        slip_model: SlippageModel,
    ) -> None:
        if side is Side.SHORT and not execution.allow_short:
            return
        # Открытие long — покупка, short — продажа.
        eff_price = slip_model.fill_price(raw_price, is_buy=side is Side.LONG)
        qty = target_qty(execution, portfolio.balance, eff_price)
        if qty <= 0:
            return
        fee = fee_model.fee(eff_price, qty)
        portfolio.open(side, qty, eff_price, time, fee)

    @staticmethod
    def _close(
        portfolio: Portfolio,
        raw_price: float,
        time: datetime,
        reason: OrderReason,
        fee_model: FeeModel,
        slip_model: SlippageModel,
    ) -> None:
        pos = portfolio.position
        if pos is None:
            return
        # Закрытие long — продажа, short — покупка.
        eff_price = slip_model.fill_price(raw_price, is_buy=pos.side is Side.SHORT)
        fee = fee_model.fee(eff_price, pos.qty)
        portfolio.close(eff_price, time, reason, fee)
