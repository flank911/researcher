"""REG-2 — фичи long/short ratio на слайс."""

from __future__ import annotations

import polars as pl


def long_short_ratio_avg(ls: pl.DataFrame | None) -> float | None:
    """Средний long/short account ratio за слайс (``None``, если данных нет)."""
    if ls is None or ls.is_empty() or "long_short_ratio" not in ls.columns:
        return None
    return float(ls["long_short_ratio"].mean())  # type: ignore[arg-type]


__all__ = ["long_short_ratio_avg"]
