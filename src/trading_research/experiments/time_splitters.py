"""DATA-6 — генератор TimeSlice.

Нарезает диапазон ``[start, end)`` на временные окна по календарю
(year / quarter / month / week) либо в один custom-слайс. Каждое окно несёт
``warmup_bars`` (для прогрева индикаторов) и может фильтроваться по ``min_bars``.

Граничные (неполные) календарные окна клипуются к ``[start, end)`` и могут быть
отброшены фильтром ``min_bars``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trading_research.data.schema import timeframe_to_ms
from trading_research.domain import TimeSlice, TimeSliceKind, TimeSliceRole


def _ensure_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _month_floor(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _add_month(dt: datetime) -> datetime:
    year, month = dt.year + (dt.month // 12), (dt.month % 12) + 1
    return dt.replace(year=year, month=month, day=1)


def _quarter_floor(dt: datetime) -> datetime:
    start_month = 3 * ((dt.month - 1) // 3) + 1
    return dt.replace(month=start_month, day=1, hour=0, minute=0, second=0, microsecond=0)


def _add_quarter(dt: datetime) -> datetime:
    result = dt
    for _ in range(3):
        result = _add_month(result)
    return result


def _year_floor(dt: datetime) -> datetime:
    return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


def _add_year(dt: datetime) -> datetime:
    return dt.replace(year=dt.year + 1, month=1, day=1)


def _week_floor(dt: datetime) -> datetime:
    monday = dt - timedelta(days=dt.weekday())
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)


def _add_week(dt: datetime) -> datetime:
    return dt + timedelta(days=7)


def _label(kind: TimeSliceKind, period_start: datetime) -> str:
    if kind is TimeSliceKind.YEAR:
        return f"{period_start.year}"
    if kind is TimeSliceKind.QUARTER:
        quarter = (period_start.month - 1) // 3 + 1
        return f"{period_start.year}-Q{quarter}"
    if kind is TimeSliceKind.MONTH:
        return f"{period_start.year}-{period_start.month:02d}"
    if kind is TimeSliceKind.WEEK:
        iso = period_start.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return "custom"


_FLOOR_AND_STEP = {
    TimeSliceKind.YEAR: (_year_floor, _add_year),
    TimeSliceKind.QUARTER: (_quarter_floor, _add_quarter),
    TimeSliceKind.MONTH: (_month_floor, _add_month),
    TimeSliceKind.WEEK: (_week_floor, _add_week),
}


def generate_time_slices(
    kind: TimeSliceKind,
    start: datetime,
    end: datetime,
    *,
    timeframe: str | None = None,
    warmup_bars: int = 0,
    min_bars: int = 0,
    role: TimeSliceRole = TimeSliceRole.RESEARCH,
) -> list[TimeSlice]:
    """Сгенерировать список временных окон.

    Args:
        kind: тип окна (year/quarter/month/week/custom).
        start, end: общий диапазон (полуинтервал ``[start, end)``).
        timeframe: нужен для расчёта числа баров при ``min_bars`` фильтрации.
        warmup_bars: сколько баров прогрева приписать каждому окну.
        min_bars: отбрасывать окна короче этого числа баров (требует ``timeframe``).
        role: роль окна (research/train/test/holdout).
    """
    start, end = _ensure_utc(start), _ensure_utc(end)
    if end <= start:
        raise ValueError(f"end must be after start: {start} >= {end}")
    if min_bars > 0 and timeframe is None:
        raise ValueError("min_bars filtering requires timeframe")

    if kind is TimeSliceKind.CUSTOM:
        periods = [(start, end)]
    else:
        floor, step = _FLOOR_AND_STEP[kind]
        periods = []
        cursor = floor(start)
        while cursor < end:
            nxt = step(cursor)
            periods.append((cursor, nxt))
            cursor = nxt

    interval_ms = timeframe_to_ms(timeframe) if timeframe else None
    slices: list[TimeSlice] = []
    for period_start, period_end in periods:
        sl_start = max(period_start, start)
        sl_end = min(period_end, end)
        if sl_start >= sl_end:
            continue
        if interval_ms is not None and min_bars > 0:
            span_ms = int((sl_end - sl_start).total_seconds() * 1000)
            if span_ms // interval_ms < min_bars:
                continue
        label = _label(kind, period_start)
        slices.append(
            TimeSlice(
                id=label,
                start=sl_start,
                end=sl_end,
                label=label,
                kind=kind,
                role=role,
                warmup_bars=warmup_bars,
                min_bars=min_bars,
            )
        )
    return slices
