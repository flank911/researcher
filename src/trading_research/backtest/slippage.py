"""BT-4 — модель проскальзывания.

Проскальзывание всегда против трейдера: покупка исполняется чуть дороже, продажа —
чуть дешевле.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlippageModel:
    slippage_pct: float

    def fill_price(self, price: float, *, is_buy: bool) -> float:
        """Эффективная цена исполнения с учётом проскальзывания."""
        direction = 1.0 if is_buy else -1.0
        return price * (1.0 + direction * self.slippage_pct)
