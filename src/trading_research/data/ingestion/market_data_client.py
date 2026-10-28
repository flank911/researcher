"""REG-1 — загрузчики внешних рыночных рядов Binance USDT-M futures.

Funding rate, open interest и long/short account ratio — по образцу DATA-1:
инжектируемый ``Transport`` (тестируется без сети), retry с backoff на 418/429/5xx,
нормализация в единую схему (UTC, миллисекунды, ключ ``open_time``).

Эндпоинты:
- funding:       ``/fapi/v1/fundingRate``
- open interest: ``/futures/data/openInterestHist`` (нужен ``period``)
- long/short:    ``/futures/data/globalLongShortAccountRatio`` (нужен ``period``)

Пагинация — по времени (курсор сдвигается за последний полученный таймстамп).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import polars as pl

from trading_research.data.ingestion.binance_client import (
    BinanceAPIError,
    HttpResponse,
    Transport,
    UrllibTransport,
)
from trading_research.data.schema import (
    FUNDING_SCHEMA,
    LONG_SHORT_RATIO_SCHEMA,
    OPEN_INTEREST_SCHEMA,
)

DEFAULT_BASE_URL = "https://fapi.binance.com"
FUNDING_PATH = "/fapi/v1/fundingRate"
OPEN_INTEREST_PATH = "/futures/data/openInterestHist"
LONG_SHORT_PATH = "/futures/data/globalLongShortAccountRatio"
MAX_LIMIT_FUNDING = 1000
MAX_LIMIT_DATA = 500


def _to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


class BinanceMarketDataClient:
    """Загрузчик funding / open interest / long-short ratio (USDT-M futures)."""

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

    # --- public fetchers ----------------------------------------------------

    def fetch_funding_rate(
        self, symbol: str, start: datetime, end: datetime, *, limit: int = MAX_LIMIT_FUNDING
    ) -> pl.DataFrame:
        """Funding rate за ``[start, end)`` → ``open_time``, ``funding_rate``."""
        rows = self._paginate(
            FUNDING_PATH, {"symbol": symbol.upper()}, "fundingTime", start, end, limit
        )
        if not rows:
            return pl.DataFrame(schema=FUNDING_SCHEMA)
        frame = pl.DataFrame(
            {
                "open_time": [int(r["fundingTime"]) for r in rows],
                "funding_rate": [float(r["fundingRate"]) for r in rows],
            }
        )
        return self._finalize(frame, FUNDING_SCHEMA)

    def fetch_open_interest(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        period: str = "1h",
        limit: int = MAX_LIMIT_DATA,
    ) -> pl.DataFrame:
        """Open interest за ``[start, end)`` → ``open_time``, ``open_interest(_value)``."""
        rows = self._paginate(
            OPEN_INTEREST_PATH,
            {"symbol": symbol.upper(), "period": period},
            "timestamp",
            start,
            end,
            limit,
        )
        if not rows:
            return pl.DataFrame(schema=OPEN_INTEREST_SCHEMA)
        frame = pl.DataFrame(
            {
                "open_time": [int(r["timestamp"]) for r in rows],
                "open_interest": [float(r["sumOpenInterest"]) for r in rows],
                "open_interest_value": [float(r["sumOpenInterestValue"]) for r in rows],
            }
        )
        return self._finalize(frame, OPEN_INTEREST_SCHEMA)

    def fetch_long_short_ratio(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        period: str = "1h",
        limit: int = MAX_LIMIT_DATA,
    ) -> pl.DataFrame:
        """Long/short account ratio за ``[start, end)``."""
        rows = self._paginate(
            LONG_SHORT_PATH,
            {"symbol": symbol.upper(), "period": period},
            "timestamp",
            start,
            end,
            limit,
        )
        if not rows:
            return pl.DataFrame(schema=LONG_SHORT_RATIO_SCHEMA)
        frame = pl.DataFrame(
            {
                "open_time": [int(r["timestamp"]) for r in rows],
                "long_short_ratio": [float(r["longShortRatio"]) for r in rows],
                "long_account": [float(r["longAccount"]) for r in rows],
                "short_account": [float(r["shortAccount"]) for r in rows],
            }
        )
        return self._finalize(frame, LONG_SHORT_RATIO_SCHEMA)

    # --- internals ----------------------------------------------------------

    def _paginate(
        self,
        path: str,
        base_params: Mapping[str, str],
        time_key: str,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[dict[str, Any]]:
        if end <= start:
            raise ValueError(f"end must be after start: {start} >= {end}")
        start_ms, end_ms = _to_ms(start), _to_ms(end)
        url = f"{self._base_url}{path}"

        rows: list[dict[str, Any]] = []
        cursor = start_ms
        while cursor < end_ms:
            params = {
                **base_params,
                "startTime": str(cursor),
                "endTime": str(end_ms),
                "limit": str(limit),
            }
            batch = self._request_with_retry(url, params)
            if not isinstance(batch, list):
                raise BinanceAPIError(200, f"unexpected payload type: {type(batch).__name__}")
            if not batch:
                break
            rows.extend(batch)
            last_time = int(batch[-1][time_key])
            next_cursor = last_time + 1  # сдвиг за последний таймстамп
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < limit:
                break
        return rows

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
        assert last_error is not None  # noqa: S101 — инвариант цикла retry
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
    def _finalize(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
        return (
            frame.unique(subset=["open_time"], keep="first")
            .sort("open_time")
            .with_columns(pl.col("open_time").cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"))
            .select(list(schema))
            .cast(schema)  # type: ignore[arg-type]
        )


__all__ = ["BinanceMarketDataClient"]
