"""BT-5 — риск-слой: маржа/ликвидация и внутрибарное исполнение TP/SL/trailing.

Движок исполняет сигналы по ``open``/``close`` (BT-3), но защитные выходы
(take-profit, stop-loss, trailing-stop) и ликвидация срабатывают *внутри* бара,
где известны только экстремумы ``high``/``low`` — без тиковой траектории.

Пессимистичные допущения (документированы намеренно):

- **Порядок движения цены неизвестен**, поэтому предполагается худший для
  трейдера сценарий: сначала цена идёт в неблагоприятную сторону, затем в
  благоприятную. Как следствие, если на одном баре задеты и TP, и SL —
  **срабатывает SL** (убыток проверяется раньше прибыли).
- **Trailing-стоп** на текущем баре считается от экстремума пути,
  зафиксированного к *началу* бара; ``high``/``low`` самого бара обновляют
  экстремум только *после* проверки стопа. То есть внутрибарный рост цены не
  защищает от внутрибарного падения на том же баре.
- **Гэп сквозь уровень**: если бар открылся уже за стоп-уровнем, исполнение
  происходит по цене открытия (реалистично хуже уровня), а не по самому уровню.

Модель ликвидации — кросс-маржинальная: весь баланс счёта служит обеспечением
позиции. Ликвидация наступает, когда капитал падает до поддерживающей маржи
``maintenance_margin_rate * notional``. Цена ликвидации:

- long:  ``P = (qty*entry - balance) / (qty * (1 - mm))``
- short: ``P = (balance + qty*entry) / (qty * (1 + mm))``

При исполнении по цене ликвидации капитал равен поддерживающей марже (малый плюс),
поэтому equity не уходит в нереалистичный минус (кроме гэпа сквозь уровень —
реалистичный остаточный убыток).
"""

from __future__ import annotations

from dataclasses import dataclass

from trading_research.backtest.orders import OrderReason
from trading_research.backtest.position import Position
from trading_research.domain import ExecutionModel


@dataclass(frozen=True)
class ExitEvent:
    """Внутрибарный выход: цена-триггер и причина закрытия."""

    price: float
    reason: OrderReason


class RiskManager:
    """Оценивает внутрибарные выходы и цену ликвидации позиции."""

    def __init__(self, execution: ExecutionModel) -> None:
        self.execution = execution

    @property
    def enabled(self) -> bool:
        """Заданы ли пользователем защитные стопы (TP/SL/trailing)."""
        ex = self.execution
        return (
            ex.stop_loss_pct is not None
            or ex.take_profit_pct is not None
            or ex.trailing_stop_pct is not None
        )

    def initial_extreme(self, position: Position) -> float:
        """Стартовое значение экстремума пути для trailing-стопа."""
        return position.entry_price

    def liquidation_price(self, position: Position, balance: float) -> float:
        """Цена, при которой капитал падает до поддерживающей маржи."""
        mm = self.execution.maintenance_margin_rate
        qty = position.qty
        entry = position.entry_price
        if position.is_long:
            return (qty * entry - balance) / (qty * (1.0 - mm))
        return (balance + qty * entry) / (qty * (1.0 + mm))

    def evaluate(
        self,
        position: Position,
        balance: float,
        extreme: float,
        *,
        high: float,
        low: float,
        bar_open: float,
    ) -> tuple[ExitEvent | None, float]:
        """Проверить внутрибарные выходы для ``position`` на баре.

        ``extreme`` — экстремум пути к началу бара (max ``high`` для long,
        min ``low`` для short). Возвращает событие выхода (или ``None``) и
        обновлённый экстремум пути.
        """
        if position.is_long:
            return self._evaluate_long(position, balance, extreme, high, low, bar_open)
        return self._evaluate_short(position, balance, extreme, high, low, bar_open)

    def _evaluate_long(
        self,
        position: Position,
        balance: float,
        extreme: float,
        high: float,
        low: float,
        bar_open: float,
    ) -> tuple[ExitEvent | None, float]:
        ex = self.execution
        entry = position.entry_price

        # Неблагоприятные уровни (ниже входа) — проверяются первыми (пессимизм).
        levels: list[tuple[float, OrderReason]] = [
            (self.liquidation_price(position, balance), OrderReason.LIQUIDATION)
        ]
        if ex.stop_loss_pct is not None:
            levels.append((entry * (1.0 - ex.stop_loss_pct), OrderReason.STOP_LOSS))
        if ex.trailing_stop_pct is not None:
            levels.append((extreme * (1.0 - ex.trailing_stop_pct), OrderReason.TRAILING_STOP))

        breached = [(p, r) for p, r in levels if low <= p]
        if breached:
            # На пути вниз первым задевается самый высокий пробитый уровень.
            trigger, reason = max(breached, key=lambda pr: pr[0])
            fill = min(bar_open, trigger)  # гэп вниз → хуже уровня
            return ExitEvent(fill, reason), extreme

        # Благоприятный TP — проверяется последним.
        if ex.take_profit_pct is not None:
            tp = entry * (1.0 + ex.take_profit_pct)
            if high >= tp:
                return ExitEvent(max(bar_open, tp), OrderReason.TAKE_PROFIT), extreme

        return None, max(extreme, high)

    def _evaluate_short(
        self,
        position: Position,
        balance: float,
        extreme: float,
        high: float,
        low: float,
        bar_open: float,
    ) -> tuple[ExitEvent | None, float]:
        ex = self.execution
        entry = position.entry_price

        # Неблагоприятные уровни (выше входа) — проверяются первыми.
        levels: list[tuple[float, OrderReason]] = [
            (self.liquidation_price(position, balance), OrderReason.LIQUIDATION)
        ]
        if ex.stop_loss_pct is not None:
            levels.append((entry * (1.0 + ex.stop_loss_pct), OrderReason.STOP_LOSS))
        if ex.trailing_stop_pct is not None:
            levels.append((extreme * (1.0 + ex.trailing_stop_pct), OrderReason.TRAILING_STOP))

        breached = [(p, r) for p, r in levels if high >= p]
        if breached:
            # На пути вверх первым задевается самый низкий пробитый уровень.
            trigger, reason = min(breached, key=lambda pr: pr[0])
            fill = max(bar_open, trigger)  # гэп вверх → хуже уровня
            return ExitEvent(fill, reason), extreme

        if ex.take_profit_pct is not None:
            tp = entry * (1.0 - ex.take_profit_pct)
            if low <= tp:
                return ExitEvent(min(bar_open, tp), OrderReason.TAKE_PROFIT), extreme

        return None, min(extreme, low)


__all__ = ["ExitEvent", "RiskManager"]
