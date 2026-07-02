"""Tests for DATA-2 — partitioned Parquet store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from trading_research.data.schema import CANDLE_SCHEMA, UTC_MS
from trading_research.data.storage.parquet_store import FUNDING, ParquetStore

REF = {"exchange": "binance", "market": "futures", "symbol": "BTCUSDT", "timeframe": "1h"}


def _candles(times: list[datetime], base: float = 100.0) -> pl.DataFrame:
    n = len(times)
    return pl.DataFrame(
        {
            "open_time": times,
            "open": [base + i for i in range(n)],
            "high": [base + i + 1 for i in range(n)],
            "low": [base + i - 1 for i in range(n)],
            "close": [base + i + 0.5 for i in range(n)],
            "volume": [10.0 + i for i in range(n)],
            "close_time": [t + timedelta(hours=1) for t in times],
            "quote_volume": [1000.0 + i for i in range(n)],
            "trades": [5 + i for i in range(n)],
            "taker_buy_base": [4.0 + i for i in range(n)],
            "taker_buy_quote": [400.0 + i for i in range(n)],
        },
        schema=CANDLE_SCHEMA,
    )


def _hours(start: datetime, n: int) -> list[datetime]:
    return [start + timedelta(hours=i) for i in range(n)]


def test_roundtrip_preserves_types_and_utc(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    df = _candles(_hours(datetime(2024, 1, 1, tzinfo=UTC), 5))
    store.write_candles(df, **REF)
    out = store.read_candles(**REF)

    assert out.schema["open_time"] == UTC_MS
    assert out.schema == df.schema
    assert out.equals(df)


def test_idempotent_rewrite_no_duplicates(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    df = _candles(_hours(datetime(2024, 1, 1, tzinfo=UTC), 24))
    store.write_candles(df, **REF)
    store.write_candles(df, **REF)  # повторная запись того же окна
    out = store.read_candles(**REF)

    assert out.height == 24
    assert out["open_time"].n_unique() == 24
    assert out.equals(df)


def test_upsert_overwrites_colliding_rows_and_appends_new(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    times = _hours(datetime(2024, 1, 1, tzinfo=UTC), 3)
    store.write_candles(_candles(times, base=100.0), **REF)

    # Перекрывающееся окно с новыми значениями + один новый бар.
    overlap = _candles([times[2], times[2] + timedelta(hours=1)], base=999.0)
    store.write_candles(overlap, **REF)

    out = store.read_candles(**REF)
    assert out.height == 4  # 3 старых + 1 новый, без дублей
    # Значение на общем баре times[2] заменено новой записью.
    collided = out.filter(pl.col("open_time") == times[2])
    assert collided["open"].item() == pytest.approx(999.0)


def test_read_range_across_multiple_partitions(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    # Данные тянутся через три месяца → три партиции.
    times = [datetime(2024, 1, 15, tzinfo=UTC) + timedelta(days=d) for d in range(0, 75, 5)]
    df = _candles(times)
    store.write_candles(df, **REF)

    # Диапазон, пересекающий границу янв/фев.
    start = datetime(2024, 1, 25, tzinfo=UTC)
    end = datetime(2024, 2, 10, tzinfo=UTC)
    out = store.read_candles(start=start, end=end, **REF)

    assert out.height > 0
    assert out["open_time"].min() >= start
    assert out["open_time"].max() <= end
    # Склейка отсортирована по времени.
    assert out["open_time"].to_list() == sorted(out["open_time"].to_list())
    expected = df.filter((pl.col("open_time") >= start) & (pl.col("open_time") <= end))
    assert out.equals(expected)


def test_partition_layout_on_disk(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    times = [datetime(2024, 1, 31, 23, tzinfo=UTC), datetime(2024, 2, 1, 0, tzinfo=UTC)]
    store.write_candles(_candles(times), **REF)

    jan = (
        tmp_path
        / "candles/exchange=binance/market=futures/symbol=BTCUSDT/timeframe=1h"
        / "year=2024/month=01/candles.parquet"
    )
    feb = jan.parent.parent / "month=02/candles.parquet"
    assert jan.exists()
    assert feb.exists()


def test_read_missing_returns_empty_with_schema(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    out = store.read_candles(**REF)
    assert out.height == 0
    assert out.schema == CANDLE_SCHEMA


def test_write_empty_is_noop(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    store.write_candles(pl.DataFrame(schema=CANDLE_SCHEMA), **REF)
    assert store.read_candles(**REF).height == 0


def test_naive_datetime_bounds_are_treated_as_utc(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    times = _hours(datetime(2024, 3, 1, tzinfo=UTC), 10)
    store.write_candles(_candles(times), **REF)
    out = store.read_candles(start=datetime(2024, 3, 1, 2), end=datetime(2024, 3, 1, 5), **REF)
    assert out.height == 4  # часы 2,3,4,5


def test_series_placeholder_roundtrip(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    times = _hours(datetime(2024, 1, 1, tzinfo=UTC), 3)
    funding = pl.DataFrame(
        {"open_time": times, "funding_rate": [0.0001, -0.0002, 0.0003]}
    ).with_columns(pl.col("open_time").cast(UTC_MS))
    store.write_series(FUNDING, funding, **REF)
    out = store.read_series(FUNDING, **REF)
    assert out.equals(funding)


def test_missing_time_column_raises(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    with pytest.raises(ValueError, match="time column"):
        store.write_series(FUNDING, pl.DataFrame({"x": [1, 2]}), **REF)
