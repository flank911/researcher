"""Tests for DATA-1 — Binance futures OHLCV client (no network)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest

from trading_research.data.ingestion.binance_client import (
    BinanceAPIError,
    BinanceFuturesClient,
    HttpResponse,
)
from trading_research.data.schema import CANDLE_COLUMNS

HOUR_MS = 60 * 60_000


def _make_klines(start_ms: int, count: int, interval_ms: int = HOUR_MS) -> list[list[object]]:
    out: list[list[object]] = []
    for i in range(count):
        ot = start_ms + i * interval_ms
        price = 100.0 + i
        out.append(
            [
                ot,  # open_time
                f"{price:.2f}",  # open
                f"{price + 1:.2f}",  # high
                f"{price - 1:.2f}",  # low
                f"{price + 0.5:.2f}",  # close
                f"{10 + i:.3f}",  # volume
                ot + interval_ms - 1,  # close_time
                "1000.0",  # quote_volume
                5 + i,  # trades
                "4.0",  # taker_buy_base
                "400.0",  # taker_buy_quote
                "0",  # ignore
            ]
        )
    return out


class FakeBinanceTransport:
    """Эмулирует /fapi/v1/klines поверх заранее заданного набора свечей."""

    def __init__(self, klines: list[list[object]], *, page_limit: int = 1000) -> None:
        self._klines = sorted(klines, key=lambda r: int(r[0]))
        self._page_limit = page_limit
        self.calls = 0

    def get(self, url: str, params: Mapping[str, str]) -> HttpResponse:
        self.calls += 1
        start = int(params["startTime"])
        end = int(params["endTime"])
        limit = min(int(params["limit"]), self._page_limit)
        rows = [r for r in self._klines if start <= int(r[0]) <= end][:limit]
        return HttpResponse(200, {}, json.dumps(rows).encode())


class SequenceTransport:
    """Возвращает заранее заданную последовательность ответов."""

    def __init__(self, responses: list[HttpResponse]) -> None:
        self._responses = responses
        self.calls = 0

    def get(self, url: str, params: Mapping[str, str]) -> HttpResponse:
        resp = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return resp


def _dt(y: int, m: int, d: int, h: int = 0) -> datetime:
    return datetime(y, m, d, h, tzinfo=UTC)


def test_fetch_basic_schema_and_values() -> None:
    start_ms = int(_dt(2024, 1, 1).timestamp() * 1000)
    transport = FakeBinanceTransport(_make_klines(start_ms, 5))
    client = BinanceFuturesClient(transport=transport)

    df = client.fetch_klines("btcusdt", "1h", _dt(2024, 1, 1), _dt(2024, 1, 1, 5))

    assert list(df.columns) == list(CANDLE_COLUMNS)
    assert df.height == 5
    assert df["open"][0] == 100.0
    assert str(df.schema["open_time"]) == "Datetime(time_unit='ms', time_zone='UTC')"
    assert df["open_time"].is_sorted()


def test_pagination_across_batches() -> None:
    start_ms = int(_dt(2024, 1, 1).timestamp() * 1000)
    # 2500 свечей, страница по 1000 -> 3 запроса
    transport = FakeBinanceTransport(_make_klines(start_ms, 2500), page_limit=1000)
    client = BinanceFuturesClient(transport=transport)

    df = client.fetch_klines(
        "BTCUSDT", "1h", _dt(2024, 1, 1), _dt(2024, 4, 20), limit=1000
    )

    assert df.height == 2500
    assert transport.calls >= 3
    assert df["open_time"].n_unique() == 2500


def test_retry_on_429_then_success() -> None:
    start_ms = int(_dt(2024, 1, 1).timestamp() * 1000)
    ok_body = json.dumps(_make_klines(start_ms, 2)).encode()
    transport = SequenceTransport(
        [
            HttpResponse(429, {"retry-after": "0"}, b"rate limited"),
            HttpResponse(429, {}, b"rate limited"),
            HttpResponse(200, {}, ok_body),
        ]
    )
    sleeps: list[float] = []
    client = BinanceFuturesClient(transport=transport, sleep_fn=sleeps.append)

    df = client.fetch_klines("BTCUSDT", "1h", _dt(2024, 1, 1), _dt(2024, 1, 1, 2))

    assert df.height == 2
    assert transport.calls == 3
    assert len(sleeps) == 2  # два backoff перед успехом


def test_empty_before_listing_returns_empty_frame() -> None:
    transport = FakeBinanceTransport([])
    client = BinanceFuturesClient(transport=transport)

    df = client.fetch_klines("NEWCOIN", "1h", _dt(2017, 1, 1), _dt(2017, 1, 2))

    assert df.height == 0
    assert list(df.columns) == list(CANDLE_COLUMNS)


def test_non_retryable_error_raises() -> None:
    transport = SequenceTransport([HttpResponse(400, {}, b'{"code":-1121}')])
    client = BinanceFuturesClient(transport=transport)

    with pytest.raises(BinanceAPIError) as exc:
        client.fetch_klines("BADSYMBOL", "1h", _dt(2024, 1, 1), _dt(2024, 1, 2))
    assert exc.value.status == 400


def test_exhausted_retries_raise() -> None:
    transport = SequenceTransport([HttpResponse(503, {}, b"unavailable")])
    client = BinanceFuturesClient(transport=transport, max_retries=2, sleep_fn=lambda _: None)

    with pytest.raises(BinanceAPIError) as exc:
        client.fetch_klines("BTCUSDT", "1h", _dt(2024, 1, 1), _dt(2024, 1, 2))
    assert exc.value.status == 503
    assert transport.calls == 3  # 1 + 2 retries


def test_invalid_timeframe_raises() -> None:
    client = BinanceFuturesClient(transport=FakeBinanceTransport([]))
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        client.fetch_klines("BTCUSDT", "7m", _dt(2024, 1, 1), _dt(2024, 1, 2))
