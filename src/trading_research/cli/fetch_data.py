"""EXP-6 — бизнес-логика команды ``fetch-data``.

Тонкая связка «загрузчик свечей → Parquet-store». Вынесена из Typer-обёртки, чтобы
тестироваться без CLI и без сети (через подставной ``KlineFetcher``).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol

import polars as pl

from trading_research.data.ingestion.binance_client import BinanceFuturesClient
from trading_research.data.storage.parquet_store import ParquetStore


class KlineFetcher(Protocol):
    """Источник свечей (реальный клиент Binance или подставной в тестах)."""

    def fetch_klines(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pl.DataFrame: ...


def fetch_data(
    *,
    exchange: str,
    market: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    data_root: str | Path,
    fetcher: KlineFetcher | None = None,
) -> int:
    """Загрузить свечи ``[start, end)`` и идемпотентно записать в Parquet-store.

    Возвращает число записанных баров. ``fetcher=None`` → реальный клиент Binance
    USDT-M futures.
    """
    fetcher = fetcher or BinanceFuturesClient()
    candles = fetcher.fetch_klines(symbol, timeframe, start, end)
    ParquetStore(data_root).write_candles(
        candles, exchange=exchange, market=market, symbol=symbol, timeframe=timeframe
    )
    return candles.height


__all__ = ["KlineFetcher", "fetch_data"]
