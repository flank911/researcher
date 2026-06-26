"""BT-2 — портфель: баланс, открытая позиция, реализованные сделки.

Учитывает только механику позиций и реализованный PnL. Комиссии/слиппедж
(поле ``fees`` зарезервировано) добавляются в BT-4.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading_research.backtest.orders import OrderReason, Side
from trading_research.backtest.position import Position


@dataclass(frozen=True)
class Trade:
    entry_time: datetime
    exit_time: datetime
    side: Side
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    fees: float
    reason: OrderReason


class Portfolio:
    """Состояние счёта в ходе бэктеста."""

    def __init__(self, initial_balance: float) -> None:
        if initial_balance <= 0:
            raise ValueError(f"initial_balance must be > 0, got {initial_balance}")
        self.balance: float = initial_balance
        self.position: Position | None = None
        self.trades: list[Trade] = []

    def equity(self, price: float) -> float:
        """Маркированный по рынку капитал при текущей цене."""
        if self.position is None:
            return self.balance
        return self.balance + self.position.unrealized_pnl(price)

    def open(self, side: Side, qty: float, price: float, time: datetime) -> None:
        if self.position is not None:
            raise RuntimeError("cannot open: position already open")
        if qty <= 0:
            raise ValueError(f"qty must be > 0, got {qty}")
        self.position = Position(side=side, qty=qty, entry_price=price, entry_time=time)

    def close(self, price: float, time: datetime, reason: OrderReason) -> Trade:
        if self.position is None:
            raise RuntimeError("cannot close: no open position")
        pos = self.position
        pnl = pos.unrealized_pnl(price)
        pnl_pct = (price / pos.entry_price - 1.0) * 100.0 * pos.side.sign
        self.balance += pnl
        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=time,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=price,
            qty=pos.qty,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fees=0.0,
            reason=reason,
        )
        self.trades.append(trade)
        self.position = None
        return trade
