"""REG-2 — фичи funding на слайс."""

from __future__ import annotations

import polars as pl


def funding_avg(funding: pl.DataFrame | None) -> float | None:
    """Средний funding-rate за слайс (``None``, если данных нет)."""
    if funding is None or funding.is_empty() or "funding_rate" not in funding.columns:
        return None
    return float(funding["funding_rate"].mean())  # type: ignore[arg-type]


__all__ = ["funding_avg"]
