"""REG-2 — фичи open interest на слайс."""

from __future__ import annotations

import polars as pl


def oi_change_pct(oi: pl.DataFrame | None) -> float | None:
    """Изменение open interest за слайс в процентах: ``(last/first - 1) * 100``.

    ``None``, если данных нет или первое значение нулевое.
    """
    if oi is None or oi.is_empty() or "open_interest" not in oi.columns:
        return None
    ordered = oi.sort("open_time")["open_interest"]
    first, last = float(ordered[0]), float(ordered[-1])
    if first == 0:
        return None
    return (last / first - 1.0) * 100.0


__all__ = ["oi_change_pct"]
