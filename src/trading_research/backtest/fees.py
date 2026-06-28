"""BT-4 — модель комиссий.

Текущие исполнения — рыночные (taker), поэтому используется единая
``commission_rate``. Разделение maker/taker появится вместе с поддержкой
лимитных ордеров (за рамками BT-4).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeeModel:
    commission_rate: float

    def fee(self, price: float, qty: float) -> float:
        """Комиссия за исполнение на ``price`` объёмом ``qty`` (по нотионалу)."""
        return abs(price * qty) * self.commission_rate
