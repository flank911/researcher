"""BT-2 — открытая позиция."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading_research.backtest.orders import Side


@dataclass(frozen=True)
class Position:
    side: Side
    qty: float
    entry_price: float
    entry_time: datetime
    entry_fee: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.side is Side.LONG

    @property
    def notional(self) -> float:
        return self.qty * self.entry_price

    def unrealized_pnl(self, price: float) -> float:
        return self.side.sign * self.qty * (price - self.entry_price)
