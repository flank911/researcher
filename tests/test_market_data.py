"""Tests for REG-1 — funding / open interest / long-short ratio (no network)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from trading_research.data.ingestion.binance_client import HttpResponse
from trading_research.data.ingestion.market_data_client import BinanceMarketDataClient
from trading_research.data.schema import (
    FUNDING_SCHEMA,
    LONG_SHORT_RATIO_SCHEMA,
    OPEN_INTEREST_SCHEMA,
    UTC_MS,
)
from trading_research.data.storage.parquet_store import FUNDING, ParquetStore
from trading_research.data.validators import validate_series

T0 = datetime(2024, 1, 1, tzinfo=UTC)
H = timedelta(hours=1)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


class _FakeTransport:
    """Мок HTTP: по подстроке пути возвращает заранее заданный JSON (200)."""

    def __init__(self, by_path: dict[str, list[dict[str, object]]]) -> None:
        self.by_path = by_path
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, params: Mapping[str, str]) -> HttpResponse:
        self.calls.append((url, dict(params)))
        for path, rows in self.by_path.items():
            if path in url:
                return HttpResponse(200, {}, json.dumps(rows).encode())
        return HttpResponse(404, {}, b"not found")


def _funding_rows() -> list[dict[str, object]]:
    return [
        {
            "symbol": "BTCUSDT",
            "fundingTime": _ms(T0 + i * 8 * H),
            "fundingRate": f"{0.0001 * (i + 1)}",
        }
        for i in range(3)
    ]


def _oi_rows() -> list[dict[str, object]]:
    return [
        {
            "symbol": "BTCUSDT",
            "sumOpenInterest": f"{1000.0 + i}",
            "sumOpenInterestValue": f"{5_000_000.0 + i}",
            "timestamp": _ms(T0 + i * H),
        }
        for i in range(3)
    ]


def _ls_rows() -> list[dict[str, object]]:
    return [
        {
            "symbol": "BTCUSDT",
            "longShortRatio": f"{1.5 + i}",
            "longAccount": "0.6",
            "shortAccount": "0.4",
            "timestamp": _ms(T0 + i * H),
        }
        for i in range(3)
    ]


def _client(rows: dict[str, list[dict[str, object]]]) -> BinanceMarketDataClient:
    return BinanceMarketDataClient(transport=_FakeTransport(rows), sleep_fn=lambda _s: None)


# --- fetchers ---------------------------------------------------------------


def test_fetch_funding_rate() -> None:
    client = _client({"fundingRate": _funding_rows()})
    df = client.fetch_funding_rate("BTCUSDT", T0, T0 + 1 * timedelta(days=1))
    assert list(df.columns) == ["open_time", "funding_rate"]
    assert df.height == 3
    assert df.schema["open_time"] == UTC_MS
    assert df["funding_rate"].to_list() == pytest.approx([0.0001, 0.0002, 0.0003])
    assert df["open_time"][0] == T0


def test_fetch_open_interest() -> None:
    client = _client({"openInterestHist": _oi_rows()})
    df = client.fetch_open_interest("BTCUSDT", T0, T0 + 1 * timedelta(days=1), period="1h")
    assert dict(df.schema) == OPEN_INTEREST_SCHEMA
    assert df.height == 3
    assert df["open_interest"][0] == pytest.approx(1000.0)


def test_fetch_long_short_ratio() -> None:
    client = _client({"globalLongShortAccountRatio": _ls_rows()})
    df = client.fetch_long_short_ratio("BTCUSDT", T0, T0 + 1 * timedelta(days=1), period="1h")
    assert dict(df.schema) == LONG_SHORT_RATIO_SCHEMA
    assert df["long_short_ratio"].to_list() == pytest.approx([1.5, 2.5, 3.5])


def test_fetch_empty_returns_schema() -> None:
    client = _client({"fundingRate": []})
    df = client.fetch_funding_rate("BTCUSDT", T0, T0 + 1 * timedelta(days=1))
    assert df.height == 0
    assert dict(df.schema) == FUNDING_SCHEMA


# --- retry ------------------------------------------------------------------


class _FlakyTransport:
    """429 на первый запрос, затем 200 — проверка retry/backoff."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.n = 0

    def get(self, url: str, params: Mapping[str, str]) -> HttpResponse:
        self.n += 1
        if self.n == 1:
            return HttpResponse(429, {"retry-after": "0"}, b"rate limited")
        return HttpResponse(200, {}, json.dumps(self.rows).encode())


def test_retry_on_429_then_success() -> None:
    transport = _FlakyTransport(_funding_rows())
    client = BinanceMarketDataClient(transport=transport, sleep_fn=lambda _s: None)
    df = client.fetch_funding_rate("BTCUSDT", T0, T0 + 1 * timedelta(days=1))
    assert df.height == 3
    assert transport.n == 2  # первый 429, второй — успех


# --- parquet round-trip -----------------------------------------------------


def test_parquet_store_round_trip(tmp_path: Path) -> None:
    client = _client({"fundingRate": _funding_rows()})
    df = client.fetch_funding_rate("BTCUSDT", T0, T0 + 1 * timedelta(days=1))
    store = ParquetStore(tmp_path)
    store.write_series(
        FUNDING, df, exchange="binance", market="futures", symbol="BTCUSDT", timeframe="8h"
    )
    back = store.read_series(
        FUNDING, exchange="binance", market="futures", symbol="BTCUSDT", timeframe="8h"
    )
    assert back.sort("open_time").equals(df)


# --- validation (DATA-3 approach) -------------------------------------------


def _funding_df(times: list[datetime], rates: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"open_time": times, "funding_rate": rates}, schema_overrides={"open_time": UTC_MS}
    )


def test_validate_series_finds_gap_and_duplicate() -> None:
    # Шаг 8h; есть пропуск (нет +16h) и дубликат (+8h дважды).
    times = [T0, T0 + 8 * H, T0 + 8 * H, T0 + 24 * H]
    report = validate_series(
        _funding_df(times, [0.0001] * 4), timeframe="8h", value_columns=("funding_rate",)
    )
    assert report.duplicate_times == [T0 + 8 * H]
    assert not report.is_monotonic
    assert report.missing_total == 1  # отсутствует +16h
    assert not report.is_clean


def test_validate_series_clean() -> None:
    times = [T0 + i * 8 * H for i in range(4)]
    report = validate_series(
        _funding_df(times, [0.0001] * 4), timeframe="8h", value_columns=("funding_rate",)
    )
    assert report.is_clean


def test_validate_series_negative_and_non_finite() -> None:
    df = pl.DataFrame(
        {
            "open_time": [T0 + i * H for i in range(3)],
            "open_interest": [1000.0, -5.0, float("nan")],
        },
        schema_overrides={"open_time": UTC_MS},
    )
    report = validate_series(
        df,
        timeframe="1h",
        value_columns=("open_interest",),
        non_negative_columns=("open_interest",),
    )
    assert report.non_finite == 1  # NaN
    assert report.negative == 1  # -5.0


def test_validate_series_missing_open_time_raises() -> None:
    with pytest.raises(ValueError, match="open_time"):
        validate_series(pl.DataFrame({"funding_rate": [0.1]}), timeframe="8h")
