"""BT-6 — funding-платежи для futures.

Раскладывает ряд funding-ставок по барам прогона: каждая ставка относится к бару,
в интервал которого ``[open_time, open_time + step)`` попадает её момент расчёта.
Для таймфреймов крупнее интервала funding (8h) в один бар может попасть несколько
ставок — они суммируются.

Знак применяется в :meth:`Portfolio.apply_funding`: при ``rate > 0`` long платит,
short получает. Отсутствие funding-данных (``None``/пусто) даёт нулевой ряд —
прогон не ломается (graceful fallback).
"""

from __future__ import annotations

import bisect
from datetime import UTC, datetime

import polars as pl

FUNDING_TIME_COL = "open_time"
FUNDING_RATE_COL = "funding_rate"


def bucket_funding(times: list[datetime], funding: pl.DataFrame | None) -> list[float]:
    """Суммарная funding-ставка на каждый бар ``times`` (по индексам баров)."""
    n = len(times)
    per_bar = [0.0] * n
    if funding is None or n == 0 or funding.is_empty():
        return per_bar

    missing = [c for c in (FUNDING_TIME_COL, FUNDING_RATE_COL) if c not in funding.columns]
    if missing:
        raise ValueError(f"funding frame missing required columns: {missing}")

    step = times[1] - times[0] if n >= 2 else None
    ftimes = funding[FUNDING_TIME_COL].to_list()
    frates = funding[FUNDING_RATE_COL].to_list()
    for raw_t, raw_r in zip(ftimes, frates, strict=False):
        if raw_t is None or raw_r is None:
            continue
        t = _to_utc(raw_t)
        idx = bisect.bisect_right(times, t) - 1
        if idx < 0:
            continue  # funding до начала ряда
        if idx == n - 1 and step is not None and t >= times[-1] + step:
            continue  # funding за пределами последнего бара
        per_bar[idx] += float(raw_r)
    return per_bar


def _to_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


__all__ = ["FUNDING_RATE_COL", "FUNDING_TIME_COL", "bucket_funding"]
