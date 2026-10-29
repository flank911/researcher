"""RUN-2 — бизнес-логика команды ``fetch-market-data``.

Связка «загрузчик рыночных рядов (REG-1) → Parquet-store (DATA-2)». До этого
тикета `fetch_funding_rate` / `fetch_open_interest` / `fetch_long_short_ratio` и
парные им `write_series` / `read_series` существовали, но между ними не было ничего:
загрузить funding/OI/LS можно было только из Python.

Следствие было тише, чем кажется: BT-6 умеет учитывать funding в PnL, но на
реальном прогоне движок получал ``funding=None`` и **молча** считал PnL без
funding-платежей — просто потому, что данные нечем было положить на диск.

Layout повторяет свечи, чтобы RUN-3 читал ряды теми же ключами:

```
{data_root}/{dataset}/exchange=…/market=…/symbol=…/timeframe=…/year=…/month=…
```

``timeframe`` здесь — ключ партиции **и** интервал агрегации Binance для OI/LS
(у funding своей периодичности нет, его расписание задаёт биржа). Поэтому для OI/LS
допустимы только интервалы, которые поддерживает Binance, — см. ``SERIES_PERIODS``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import polars as pl

from trading_research.data.ingestion.market_data_client import BinanceMarketDataClient
from trading_research.data.storage.parquet_store import (
    FUNDING,
    LONG_SHORT_RATIO,
    OPEN_INTEREST,
    ParquetStore,
)


class SeriesKind(StrEnum):
    """Что загружаем. ``ALL`` разворачивается в три остальных."""

    FUNDING = FUNDING
    OPEN_INTEREST = OPEN_INTEREST
    LONG_SHORT_RATIO = LONG_SHORT_RATIO
    ALL = "all"


# Интервалы, которые Binance принимает в ``period`` для /futures/data/* (OI и LS).
# У funding интервала нет — расписание выплат задаёт биржа.
SERIES_PERIODS: tuple[str, ...] = ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d")

_PERIODIC = (SeriesKind.OPEN_INTEREST, SeriesKind.LONG_SHORT_RATIO)


class MarketDataFetcher(Protocol):
    """Источник рыночных рядов (реальный клиент REG-1 или подставной в тестах)."""

    def fetch_funding_rate(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame: ...

    def fetch_open_interest(
        self, symbol: str, start: datetime, end: datetime, *, period: str
    ) -> pl.DataFrame: ...

    def fetch_long_short_ratio(
        self, symbol: str, start: datetime, end: datetime, *, period: str
    ) -> pl.DataFrame: ...


def expand_series(series: SeriesKind) -> list[SeriesKind]:
    """``ALL`` → три конкретных ряда; остальное — сам ряд."""
    if series is SeriesKind.ALL:
        return [SeriesKind.FUNDING, SeriesKind.OPEN_INTEREST, SeriesKind.LONG_SHORT_RATIO]
    return [series]


def fetch_market_data(
    *,
    exchange: str,
    market: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    series: SeriesKind = SeriesKind.ALL,
    data_root: str | Path,
    fetcher: MarketDataFetcher | None = None,
) -> dict[str, int]:
    """Загрузить выбранные ряды за ``[start, end)`` и записать в Parquet-store.

    Возвращает число записанных строк по каждому ряду. Запись идемпотентна
    (``write_series`` делает upsert по ``open_time``, как и для свечей).
    ``fetcher=None`` → реальный клиент Binance USDT-M futures.

    Raises:
        ValueError: ``timeframe`` не поддерживается Binance для OI/LS.
    """
    wanted = expand_series(series)
    _validate_period(wanted, timeframe)

    fetcher = fetcher or BinanceMarketDataClient()
    store = ParquetStore(data_root)
    written: dict[str, int] = {}

    for kind in wanted:
        frame = _fetch_one(fetcher, kind, symbol=symbol, start=start, end=end, period=timeframe)
        store.write_series(
            kind.value,
            frame,
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
        )
        written[kind.value] = frame.height

    return written


def _validate_period(wanted: list[SeriesKind], timeframe: str) -> None:
    """OI и LS агрегируются Binance только по фиксированному набору интервалов.

    Без этой проверки неподдерживаемый интервал возвращает пустой список — то есть
    «успешную» загрузку нуля строк, что выглядит как отсутствие данных, а не как
    ошибка запроса.
    """
    periodic = [k.value for k in wanted if k in _PERIODIC]
    if periodic and timeframe not in SERIES_PERIODS:
        raise ValueError(
            f"timeframe {timeframe!r} is not a valid Binance period for {', '.join(periodic)}; "
            f"supported: {', '.join(SERIES_PERIODS)}"
        )


def _fetch_one(
    fetcher: MarketDataFetcher,
    kind: SeriesKind,
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    period: str,
) -> pl.DataFrame:
    if kind is SeriesKind.FUNDING:
        return fetcher.fetch_funding_rate(symbol, start, end)
    if kind is SeriesKind.OPEN_INTEREST:
        return fetcher.fetch_open_interest(symbol, start, end, period=period)
    if kind is SeriesKind.LONG_SHORT_RATIO:
        return fetcher.fetch_long_short_ratio(symbol, start, end, period=period)
    raise ValueError(f"cannot fetch aggregate series kind {kind!r}")


__all__ = [
    "SERIES_PERIODS",
    "MarketDataFetcher",
    "SeriesKind",
    "expand_series",
    "fetch_market_data",
]
