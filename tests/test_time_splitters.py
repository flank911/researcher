"""Tests for DATA-6 — TimeSlice generator."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trading_research.domain import TimeSliceKind, TimeSliceRole
from trading_research.experiments.time_splitters import generate_time_slices


def _dt(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=UTC)


def test_monthly_full_year() -> None:
    slices = generate_time_slices(TimeSliceKind.MONTH, _dt(2024, 1, 1), _dt(2025, 1, 1))
    assert len(slices) == 12
    assert slices[0].label == "2024-01"
    assert slices[-1].label == "2024-12"
    assert slices[0].start == _dt(2024, 1, 1)
    assert slices[0].end == _dt(2024, 2, 1)
    assert slices[-1].end == _dt(2025, 1, 1)


def test_monthly_spans_year_boundary() -> None:
    slices = generate_time_slices(TimeSliceKind.MONTH, _dt(2024, 11, 1), _dt(2025, 2, 1))
    assert [s.label for s in slices] == ["2024-11", "2024-12", "2025-01"]


def test_partial_edges_are_clipped() -> None:
    slices = generate_time_slices(TimeSliceKind.MONTH, _dt(2024, 1, 15), _dt(2024, 3, 10))
    assert slices[0].label == "2024-01"
    assert slices[0].start == _dt(2024, 1, 15)  # клип к началу диапазона
    assert slices[-1].label == "2024-03"
    assert slices[-1].end == _dt(2024, 3, 10)  # клип к концу диапазона


def test_quarterly_and_yearly() -> None:
    q = generate_time_slices(TimeSliceKind.QUARTER, _dt(2024, 1, 1), _dt(2025, 1, 1))
    assert [s.label for s in q] == ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4"]
    y = generate_time_slices(TimeSliceKind.YEAR, _dt(2021, 1, 1), _dt(2024, 1, 1))
    assert [s.label for s in y] == ["2021", "2022", "2023"]


def test_weekly_starts_on_monday() -> None:
    slices = generate_time_slices(TimeSliceKind.WEEK, _dt(2024, 1, 1), _dt(2024, 1, 22))
    # 2024-01-01 is a Monday
    assert slices[0].start == _dt(2024, 1, 1)
    assert slices[0].start.weekday() == 0
    assert slices[1].start == _dt(2024, 1, 8)


def test_custom_single_slice() -> None:
    slices = generate_time_slices(TimeSliceKind.CUSTOM, _dt(2021, 3, 5), _dt(2022, 6, 1))
    assert len(slices) == 1
    assert slices[0].start == _dt(2021, 3, 5)
    assert slices[0].end == _dt(2022, 6, 1)


def test_warmup_and_role_propagated() -> None:
    slices = generate_time_slices(
        TimeSliceKind.MONTH,
        _dt(2024, 1, 1),
        _dt(2024, 3, 1),
        warmup_bars=200,
        role=TimeSliceRole.TRAIN,
    )
    assert all(s.warmup_bars == 200 for s in slices)
    assert all(s.role == TimeSliceRole.TRAIN for s in slices)


def test_min_bars_filters_short_windows() -> None:
    # январь имеет 31*24=744 бара на 1h, неполный край — меньше
    slices = generate_time_slices(
        TimeSliceKind.MONTH,
        _dt(2024, 1, 28),  # неполный январь: ~3 дня
        _dt(2024, 3, 1),
        timeframe="1h",
        min_bars=300,
    )
    labels = [s.label for s in slices]
    assert "2024-01" not in labels  # отброшен (короткий)
    assert "2024-02" in labels


def test_min_bars_requires_timeframe() -> None:
    with pytest.raises(ValueError, match="requires timeframe"):
        generate_time_slices(TimeSliceKind.MONTH, _dt(2024, 1, 1), _dt(2024, 2, 1), min_bars=100)


def test_bad_range_raises() -> None:
    with pytest.raises(ValueError, match="end must be after start"):
        generate_time_slices(TimeSliceKind.MONTH, _dt(2024, 2, 1), _dt(2024, 1, 1))
