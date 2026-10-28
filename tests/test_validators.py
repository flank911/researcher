"""Tests for DATA-3 — валидация качества свечей."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from trading_research.data.schema import UTC_MS
from trading_research.data.validators import (
    ValidationReport,
    save_report,
    validate_candles,
)

H = timedelta(hours=1)
T0 = datetime(2024, 1, 1, tzinfo=UTC)

# Поля строки: (open_time, open, high, low, close, volume).
Row = tuple[datetime, float, float, float, float, float]


def _df(rows: list[Row]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open_time": [r[0] for r in rows],
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [r[5] for r in rows],
        },
        schema_overrides={"open_time": UTC_MS},
    )


def _ohlc(t: datetime, price: float = 100.0, vol: float = 10.0) -> Row:
    """Валидный бар с ценой ``price``."""
    return (t, price, price + 1.0, price - 1.0, price, vol)


def _run(hours: list[int]) -> list[Row]:
    """Бары на указанных часовых смещениях от T0."""
    return [_ohlc(T0 + h * H) for h in hours]


# --- clean baseline ---------------------------------------------------------


def test_clean_dataset_is_clean() -> None:
    report = validate_candles(_df(_run([0, 1, 2, 3, 4])), timeframe="1h")
    assert report.total_bars == 5
    assert report.step_ms == 3_600_000
    assert report.is_clean
    assert report.gaps == []
    assert report.is_monotonic


def test_empty_dataset() -> None:
    empty = _df([])
    report = validate_candles(empty, timeframe="1h")
    assert report.total_bars == 0
    assert report.is_clean


def test_missing_columns_raise() -> None:
    bad = pl.DataFrame({"open_time": [T0], "close": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_candles(bad, timeframe="1h")


# --- gaps -------------------------------------------------------------------


def test_holey_dataset_finds_all_gaps() -> None:
    # Пропущены часы 3,4 (между 2 и 5) и час 7 (между 6 и 8).
    report = validate_candles(_df(_run([0, 1, 2, 5, 6, 8])), timeframe="1h")
    assert len(report.gaps) == 2
    assert [g.missing_bars for g in report.gaps] == [2, 1]
    assert report.missing_bars_total == 3
    assert report.gaps[0].start == T0 + 2 * H
    assert report.gaps[0].end == T0 + 5 * H
    assert not report.is_clean


def test_leading_and_trailing_gaps_with_window() -> None:
    rows = _run([2, 3, 4])  # данные с часа 2 по 4
    report = validate_candles(
        _df(rows),
        timeframe="1h",
        expected_start=T0,  # окно с часа 0 → ведущий пропуск 2 бара
        expected_end=T0 + 6 * H,  # до часа 6 → замыкающий пропуск 2 бара
    )
    assert len(report.gaps) == 2
    assert report.gaps[0].missing_bars == 2  # часы 0,1
    assert report.gaps[0].start == T0
    assert report.gaps[-1].missing_bars == 2  # часы 5,6
    assert report.gaps[-1].end == T0 + 6 * H


# --- duplicates / monotonicity ----------------------------------------------


def test_detects_duplicates() -> None:
    rows = [_ohlc(T0), _ohlc(T0 + H), _ohlc(T0 + H), _ohlc(T0 + 2 * H)]
    report = validate_candles(_df(rows), timeframe="1h")
    assert report.duplicate_times == [T0 + H]
    assert not report.is_monotonic  # равные соседние времена рвут строгий рост


def test_detects_non_monotonic_without_duplicates() -> None:
    rows = [_ohlc(T0), _ohlc(T0 + 2 * H), _ohlc(T0 + H), _ohlc(T0 + 3 * H)]
    report = validate_candles(_df(rows), timeframe="1h")
    assert not report.is_monotonic
    assert report.duplicate_times == []


# --- price / volume defects -------------------------------------------------


def test_detects_non_positive_prices() -> None:
    rows = [_ohlc(T0), (T0 + H, 100.0, 101.0, 0.0, 100.0, 10.0)]  # low = 0
    report = validate_candles(_df(rows), timeframe="1h")
    assert report.non_positive_prices == 1


def test_detects_negative_volume() -> None:
    rows = [_ohlc(T0), (T0 + H, 100.0, 101.0, 99.0, 100.0, -5.0)]
    report = validate_candles(_df(rows), timeframe="1h")
    assert report.negative_volume == 1


def test_detects_ohlc_inconsistency() -> None:
    # high ниже close — нарушение инварианта «high — максимум бара».
    rows = [_ohlc(T0), (T0 + H, 100.0, 100.5, 99.0, 101.0, 10.0)]
    report = validate_candles(_df(rows), timeframe="1h")
    assert report.ohlc_inconsistencies == [T0 + H]
    assert not report.is_clean


def test_detects_price_outlier() -> None:
    # close 100 → 200 (+100%) при пороге 50%.
    rows = [_ohlc(T0, price=100.0), _ohlc(T0 + H, price=200.0)]
    report = validate_candles(_df(rows), timeframe="1h", max_return=0.5)
    assert report.price_outliers == [T0 + H]


# --- listing ----------------------------------------------------------------


def test_pre_listing_period_not_flagged() -> None:
    rows = _run([0, 1, 2])  # данные с T0
    window_start = T0 - 5 * H  # окно начинается за 5ч до первого бара

    # Без листинга — ведущий пропуск есть.
    no_listing = validate_candles(_df(rows), timeframe="1h", expected_start=window_start)
    assert no_listing.missing_bars_total == 5

    # С листингом = T0 (данных до него быть не должно) — пропуска нет.
    with_listing = validate_candles(
        _df(rows), timeframe="1h", expected_start=window_start, listing_at=T0
    )
    assert with_listing.gaps == []
    assert with_listing.listing_at == T0


def test_partial_pre_listing_counts_only_after_listing() -> None:
    rows = _run([0, 1, 2])
    window_start = T0 - 5 * H
    listing = T0 - 2 * H  # листинг за 2ч до первого бара
    report = validate_candles(
        _df(rows), timeframe="1h", expected_start=window_start, listing_at=listing
    )
    assert report.missing_bars_total == 2  # только часы [-2, -1]
    assert report.gaps[0].start == listing


# --- report serialization ---------------------------------------------------


def test_report_json_roundtrip(tmp_path: Path) -> None:
    report = validate_candles(_df(_run([0, 1, 2, 5])), timeframe="1h")
    path = tmp_path / "report.json"
    save_report(report, path)
    restored = ValidationReport.model_validate_json(path.read_text(encoding="utf-8"))
    assert restored == report
    assert restored.gaps[0].missing_bars == 2
