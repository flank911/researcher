"""DATA-1 — клиент загрузки OHLCV с Binance USDT-M futures.

HTTP-транспорт инжектируется (Protocol ``Transport``), поэтому клиент тестируется
без сети. По умолчанию используется ``UrllibTransport`` поверх стандартной
библиотеки — без дополнительных зависимостей.

Особенности:
- пагинация по времени (Binance futures отдаёт до 1500 свечей за запрос);
- retry с экспоненциальным backoff на 418/429/5xx (с учётом ``Retry-After``);
- нормализация в единую схему (UTC, миллисекунды, ключ бара ``open_time``).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import polars as pl

from trading_research.data.schema import CANDLE_COLUMNS, empty_candles, timeframe_to_ms

DEFAULT_BASE_URL = "https://fapi.binance.com"
KLINES_PATH = "/fapi/v1/klines"
MAX_LIMIT = 1500


class BinanceAPIError(RuntimeError):
    """Невосстановимая ошибка ответа Binance API."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Binance API error {status}: {body[:200]}")
        self.status = status
        self.body = body


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class Transport(Protocol):
    """Абстракция HTTP GET (для инъекции в тестах)."""

    def get(self, url: str, params: Mapping[str, str]) -> HttpResponse: ...


class UrllibTransport:
    """Транспорт по умолчанию на базе ``urllib`` (stdlib)."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    def get(self, url: str, params: Mapping[str, str]) -> HttpResponse:
        query = urllib.parse.urlencode(params)
        full_url = f"{url}?{query}" if query else url
        req = urllib.request.Request(full_url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                return HttpResponse(
                    status=resp.status,
                    headers={k.lower(): v for k, v in resp.headers.items()},
                    body=resp.read(),
                )
        except urllib.error.HTTPError as exc:
            return HttpResponse(
                status=exc.code,
                headers={k.lower(): v for k, v in (exc.headers or {}).items()},
                body=exc.read() if hasattr(exc, "read") else b"",
            )


def _to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


class BinanceFuturesClient:
    """Загрузчик исторических свечей Binance USDT-M futures."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        transport: Transport | None = None,
        max_retries: int = 5,
        backoff_base: float = 0.5,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._transport = transport or UrllibTransport()
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._sleep = sleep_fn

    def fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        limit: int = MAX_LIMIT,
    ) -> pl.DataFrame:
        """Загрузить свечи ``[start, end)`` в каноническую схему.

        Возвращает отсортированный по ``open_time`` DataFrame без дублей.
        Для периода до листинга символа вернёт пустой DataFrame со схемой.
        """
        if end <= start:
            raise ValueError(f"end must be after start: {start} >= {end}")
        interval_ms = timeframe_to_ms(timeframe)
        start_ms, end_ms = _to_ms(start), _to_ms(end)
        limit = min(limit, MAX_LIMIT)

        rows: list[list[Any]] = []
        cursor = start_ms
        while cursor < end_ms:
            batch = self._request_klines(symbol, timeframe, cursor, end_ms, limit)
            if not batch:
                break
            rows.extend(batch)
            last_open = int(batch[-1][0])
            next_cursor = last_open + interval_ms
            if next_cursor <= cursor:  # защита от зацикливания
                break
            cursor = next_cursor
            if len(batch) < limit:
                break

        return self._to_frame(rows, start_ms, end_ms)

    def _request_klines(
        self, symbol: str, timeframe: str, start_ms: int, end_ms: int, limit: int
    ) -> list[list[Any]]:
        params = {
            "symbol": symbol.upper(),
            "interval": timeframe,
            "startTime": str(start_ms),
            "endTime": str(end_ms),
            "limit": str(limit),
        }
        url = f"{self._base_url}{KLINES_PATH}"
        payload = self._request_with_retry(url, params)
        if not isinstance(payload, list):
            raise BinanceAPIError(200, f"unexpected payload type: {type(payload).__name__}")
        return payload

    def _request_with_retry(self, url: str, params: Mapping[str, str]) -> Any:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            resp = self._transport.get(url, params)
            if resp.status == 200:
                return json.loads(resp.body.decode("utf-8"))
            if resp.status in (418, 429) or resp.status >= 500:
                last_error = BinanceAPIError(resp.status, resp.body.decode("utf-8", "replace"))
                if attempt < self._max_retries:
                    self._sleep(self._retry_delay(resp, attempt))
                    continue
            else:
                raise BinanceAPIError(resp.status, resp.body.decode("utf-8", "replace"))
        assert last_error is not None
        raise last_error

    def _retry_delay(self, resp: HttpResponse, attempt: int) -> float:
        retry_after = resp.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return float(self._backoff_base * (2**attempt))

    @staticmethod
    def _to_frame(rows: list[list[Any]], start_ms: int, end_ms: int) -> pl.DataFrame:
        if not rows:
            return empty_candles()
        # Binance kline: [open_time, open, high, low, close, volume, close_time,
        #   quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
        raw = pl.DataFrame(
            {
                "open_time": [int(r[0]) for r in rows],
                "open": [float(r[1]) for r in rows],
                "high": [float(r[2]) for r in rows],
                "low": [float(r[3]) for r in rows],
                "close": [float(r[4]) for r in rows],
                "volume": [float(r[5]) for r in rows],
                "close_time": [int(r[6]) for r in rows],
                "quote_volume": [float(r[7]) for r in rows],
                "trades": [int(r[8]) for r in rows],
                "taker_buy_base": [float(r[9]) for r in rows],
                "taker_buy_quote": [float(r[10]) for r in rows],
            }
        )
        return (
            raw.filter((pl.col("open_time") >= start_ms) & (pl.col("open_time") < end_ms))
            .unique(subset=["open_time"], keep="first")
            .sort("open_time")
            .with_columns(
                pl.col("open_time").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
                pl.col("close_time").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
            )
            .select(CANDLE_COLUMNS)
        )
