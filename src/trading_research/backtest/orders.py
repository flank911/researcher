"""BT-2 — ордера и базовые перечисления исполнения."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        """+1 для long, -1 для short (направление PnL)."""
        return 1 if self is Side.LONG else -1

    def opposite(self) -> Side:
        return Side.SHORT if self is Side.LONG else Side.LONG


class OrderReason(StrEnum):
    ENTRY = "entry"
    EXIT = "exit"
    REVERSE = "reverse"
    FINAL = "final"  # принудительное закрытие в конце данных
    STOP_LOSS = "stop_loss"  # BT-5: intrabar SL
    TAKE_PROFIT = "take_profit"  # BT-5: intrabar TP
    TRAILING_STOP = "trailing_stop"  # BT-5: intrabar trailing stop
    LIQUIDATION = "liquidation"  # BT-5: пробой поддерживающей маржи


@dataclass(frozen=True)
class Order:
    """Намерение исполнить сделку на конкретном баре."""

    time: datetime
    side: Side
    qty: float
    price: float
    reason: OrderReason
