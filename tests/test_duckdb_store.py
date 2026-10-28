"""Tests for DATA-5 — DuckDB/Polars access layer (+warmup)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from trading_research.data.schema import CANDLE_SCHEMA
from trading_research.data.storage.duckdb_store import WARMUP_COL, DuckDBStore
from trading_research.data.storage.parquet_store import ParquetStore

KEYS = {"exchange": "binance", "market": "futures", "symbol": "BTCUSDT", "timeframe": "1h"}
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _candles(n: int, start: datetime = T0) -> pl.DataFrame:
    times = [start + timedelta(hours=i) for i in range(n)]
    return pl.DataFrame(
        {
            "open_time": times,
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [10.0 + i for i in range(n)],
            "close_time": [t + timedelta(hours=1) for t in times],
            "quote_volume": [1000.0 + i for i in range(n)],
            "trades": [5 + i for i in range(n)],
            "taker_buy_base": [4.0 + i for i in range(n)],
            "taker_buy_quote": [400.0 + i for i in range(n)],
        },
        schema=CANDLE_SCHEMA,
    )


def _store(tmp_path: Path, n: int) -> DuckDBStore:
    ParquetStore(tmp_path).write_candles(_candles(n), **KEYS)
    return DuckDBStore(tmp_path)


# --- scan -------------------------------------------------------------------


def test_scan_returns_lazyframe(tmp_path: Path) -> None:
    store = _store(tmp_path, 24)
    lf = store.scan_candles(**KEYS)
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect().height == 24


def test_scan_empty_when_no_data(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path)
    out = store.scan_candles(**KEYS).collect()
    assert out.height == 0
    assert set(out.columns) == set(CANDLE_SCHEMA)


# --- warmup window ----------------------------------------------------------


def test_read_window_marks_warmup(tmp_path: Path) -> None:
    store = _store(tmp_path, 100)
    out = store.read_window(
        **KEYS, start=T0 + 40 * timedelta(hours=1), end=T0 + 60 * timedelta(hours=1), warmup_bars=10
    )
    # Прогрев: бары 30..39 (10 шт), окно: бары 40..60 (21 шт).
    assert out.height == 31
    assert out[WARMUP_COL].sum() == 10
    assert out[WARMUP_COL].to_list()[:10] == [True] * 10
    assert out.filter(~pl.col(WARMUP_COL))["open_time"].min() == T0 + 40 * timedelta(hours=1)
    assert out["open_time"].min() == T0 + 30 * timedelta(hours=1)


def test_read_window_no_warmup(tmp_path: Path) -> None:
    store = _store(tmp_path, 50)
    out = store.read_window(
        **KEYS, start=T0 + 10 * timedelta(hours=1), end=T0 + 20 * timedelta(hours=1)
    )
    assert out.height == 11
    assert out[WARMUP_COL].sum() == 0
    assert out["open_time"].min() == T0 + 10 * timedelta(hours=1)


def test_warmup_best_effort_at_series_start(tmp_path: Path) -> None:
    # Запрошено 20 баров прогрева, но до start доступно лишь 5 → берём 5.
    store = _store(tmp_path, 50)
    out = store.read_window(**KEYS, start=T0 + 5 * timedelta(hours=1), warmup_bars=20)
    assert out[WARMUP_COL].sum() == 5
    assert out["open_time"].min() == T0


def test_read_window_large_fixture_returns_only_window(tmp_path: Path) -> None:
    # Большой фикстур во множестве партиций; окно читается выборочно (не весь файл).
    store = _store(tmp_path, 50_000)
    window_start = T0 + 40_000 * timedelta(hours=1)
    window_end = window_start + 23 * timedelta(hours=1)
    out = store.read_window(**KEYS, start=window_start, end=window_end, warmup_bars=10)
    assert out.height == 34  # 10 warmup + 24 bars
    assert out[WARMUP_COL].sum() == 10


# --- duckdb sql -------------------------------------------------------------


def test_query_sql_aggregates(tmp_path: Path) -> None:
    store = _store(tmp_path, 24)
    out = store.query("SELECT count(*) AS n, max(high) AS hi FROM candles", **KEYS)
    assert out["n"][0] == 24
    assert out["hi"][0] == 101.0 + 23  # high = 101 + i


def test_query_sql_with_filter(tmp_path: Path) -> None:
    store = _store(tmp_path, 24)
    out = store.query("SELECT count(*) AS n FROM candles WHERE close > 110", **KEYS)
    # close = 100.5 + i > 110 → i >= 10 → 14 баров (i=10..23)
    assert out["n"][0] == 14


def test_query_empty_source(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path)
    out = store.query("SELECT count(*) AS n FROM candles", **KEYS)
    assert out["n"][0] == 0
