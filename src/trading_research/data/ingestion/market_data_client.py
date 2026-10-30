"""REG-1 — загрузчики внешних рыночных рядов Binance USDT-M futures.

Funding rate, open interest и long/short account ratio — по образцу DATA-1:
инжектируемый ``Transport`` (тестируется без сети), retry с backoff на 418/429/5xx,
нормализация в единую схему (UTC, миллисекунды, ключ ``open_time``).

Эндпоинты:
- funding:       ``/fapi/v1/fundingRate``
- open interest: ``/futures/data/openInterestHist`` (нужен ``period``)
- long/short:    ``/futures/data/globalLongShortAccountRatio`` (нужен ``period``)

Пагинация зависит от эндпоинта, и это не деталь реализации, а REG-5:

- ``/fapi/v1/fundingRate`` отдаёт точки **от** ``startTime`` вперёд, поэтому для
  него работает курсор (сдвиг за последний полученный таймстамп);
- ``/futures/data/*`` отдаёт **хвост** запрошенного окна — последние ``limit``
  точек перед ``endTime``. На таком ответе курсор после первого батча оказывается
  за ``endTime``, цикл выходит, и начало диапазона просто не загружается. Поэтому
  эти ряды грузятся нарезкой диапазона на окна по ``limit`` периодов: в окне ровно
  ``limit`` точек, значит «хвост» окна и есть всё окно — от направления ответа
  такая загрузка не зависит.

Границы окна всюду полуинтервал ``[start, end)`` — как в ``fetch-data``,
``DuckDBStore.read_window`` и конфиге эксперимента. Binance трактует ``endTime``
включительно, поэтому в запрос уходит ``end - 1 ms``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
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

# Интервалы агрегации, которые Binance принимает в ``period`` для /futures/data/*,
# и их длительность. Длительность нужна не для валидации, а для нарезки диапазона
# на окна: ширина окна — ``limit × period``. У funding своего периода нет
# (расписание выплат задаёт биржа), поэтому его здесь и нет.
PERIOD_MS: dict[str, int] = {
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}

# Окно доступности /futures/data/* (OI и LS). Замерено на живом API 2026-10-30
# для обоих эндпоинтов: startTime за 25, 28 и 29 дней назад отвечает, за 30, 31,
# 32, 35 и 40 — `-1130 parameter 'startTime' is invalid`. У funding окна нет,
# его история доступна целиком.
RETENTION = timedelta(days=30)
# Кламп идёт с суточным запасом: без него запрос у самой границы окна падал бы
# случайным образом — из-за расхождения часов и времени самого запроса.
SAFE_RETENTION = timedelta(days=29)

# Код Binance для startTime за пределами окна retention. Это не пустой ответ и не
# общая ошибка параметра — по нему и опознаётся выход за окно.
RETENTION_ERROR_CODE = -1130


def _to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


class RetentionWindowError(BinanceAPIError):
    """``startTime`` за пределами окна доступности ``/futures/data/*``.

    Отдельный тип, а не сырой ``BinanceAPIError``: по тексту ``parameter
    'startTime' is invalid`` невозможно догадаться, что речь про 30-дневное окно
    и что история этих рядов добывается только регулярным сбором. Вызывающая
    сторона при этом может отличить выход за окно от прочих ошибок API.
    """

    def __init__(self, status: int, body: str, *, series: str) -> None:
        # В обход ``BinanceAPIError.__init__``: он обрезает тело до 200 символов,
        # то есть срезал бы как раз объяснение, ради которого этот тип и нужен.
        RuntimeError.__init__(
            self,
            f"{series}: Binance отдаёт этот ряд только за последние "
            f"{RETENTION.days} дней, а запрошенное начало окна за его пределами "
            f"(код {RETENTION_ERROR_CODE}). Запрашивайте окно в пределах "
            f"{SAFE_RETENTION.days} дней; историю за весь диапазон даёт только "
            f"регулярный сбор — команда update-market-data. "
            f"Для funding ограничения нет: --series funding. Ответ API: {body[:200]}",
        )
        self.status = status
        self.body = body


def _is_retention_error(body: str) -> bool:
    """Отличить выход за окно retention от прочих ошибок 400."""
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return False
    return isinstance(payload, dict) and payload.get("code") == RETENTION_ERROR_CODE


def _period_ms(period: str) -> int:
    """Длительность интервала агрегации в миллисекундах.

    Raises:
        ValueError: период не из набора, который поддерживает Binance — по нему
            не посчитать ширину окна, и запрос уйдёт в сеть заведомо неверным.
    """
    try:
        return PERIOD_MS[period]
    except KeyError:
        raise ValueError(
            f"unsupported period {period!r}; supported: {', '.join(PERIOD_MS)}"
        ) from None


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
        rows = self._paginate_forward(
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
        rows = self._paginate_windows(
            OPEN_INTEREST_PATH,
            {"symbol": symbol.upper(), "period": period},
            start,
            end,
            limit,
            period,
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
        rows = self._paginate_windows(
            LONG_SHORT_PATH,
            {"symbol": symbol.upper(), "period": period},
            start,
            end,
            limit,
            period,
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

    def _paginate_forward(
        self,
        path: str,
        base_params: Mapping[str, str],
        time_key: str,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Курсорная пагинация для эндпоинтов, отдающих точки **от** ``startTime``.

        Применима только к ``/fapi/v1/fundingRate``: на хвостовом ответе
        ``/futures/data/*`` курсор перескакивает за ``endTime`` после первого же
        батча и теряет начало диапазона (REG-5) — для них есть
        ``_paginate_windows``.
        """
        start_ms, end_ms = self._window_ms(start, end)
        url = f"{self._base_url}{path}"

        rows: list[dict[str, Any]] = []
        cursor = start_ms
        while cursor < end_ms:
            batch = self._fetch_batch(url, base_params, cursor, end_ms, limit)
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1][time_key]) + 1  # сдвиг за последний таймстамп
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < limit:
                break
        return rows

    def _paginate_windows(
        self,
        path: str,
        base_params: Mapping[str, str],
        start: datetime,
        end: datetime,
        limit: int,
        period: str,
    ) -> list[dict[str, Any]]:
        """Нарезка диапазона на окна по ``limit`` периодов — для ``/futures/data/*``.

        Эти эндпоинты отдают хвост запрошенного окна, поэтому направление ответа
        важно только до тех пор, пока окно шире ``limit`` точек. Делаем окна ровно
        в ``limit`` периодов — и хвост окна совпадает со всем окном.

        Пустой ответ цикл **не** прерывает: дыра в середине диапазона (данных за
        это окно у биржи нет) не должна отменять загрузку остальных окон.
        """
        start_ms, end_ms = self._window_ms(start, end)
        span_ms = _period_ms(period) * limit
        url = f"{self._base_url}{path}"

        rows: list[dict[str, Any]] = []
        window_start = start_ms
        while window_start < end_ms:
            window_end = min(window_start + span_ms, end_ms)
            try:
                batch = self._fetch_batch(url, base_params, window_start, window_end, limit)
            except BinanceAPIError as error:
                if not _is_retention_error(error.body):
                    raise
                raise RetentionWindowError(
                    error.status, error.body, series=path.rsplit("/", 1)[-1]
                ) from error
            rows.extend(batch)
            window_start = window_end
        return rows

    @staticmethod
    def _window_ms(start: datetime, end: datetime) -> tuple[int, int]:
        if end <= start:
            raise ValueError(f"end must be after start: {start} >= {end}")
        return _to_ms(start), _to_ms(end)

    def _fetch_batch(
        self,
        url: str,
        base_params: Mapping[str, str],
        start_ms: int,
        end_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Один запрос за окно ``[start_ms, end_ms)``.

        ``endTime`` у Binance включителен, поэтому в запрос уходит ``end_ms - 1``:
        иначе точка ровно на правой границе попадает и в это окно, и в следующее,
        а контракт метода — полуинтервал.
        """
        params = {
            **base_params,
            "startTime": str(start_ms),
            "endTime": str(end_ms - 1),
            "limit": str(limit),
        }
        batch = self._request_with_retry(url, params)
        if not isinstance(batch, list):
            raise BinanceAPIError(200, f"unexpected payload type: {type(batch).__name__}")
        return batch

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


__all__ = [
    "PERIOD_MS",
    "RETENTION",
    "RETENTION_ERROR_CODE",
    "SAFE_RETENTION",
    "BinanceMarketDataClient",
    "RetentionWindowError",
]
