"""Канонические схемы данных (общие для ingestion и storage).

Единая схема OHLCV: ключ бара — ``open_time`` (UTC, миллисекунды).
"""

from __future__ import annotations

import polars as pl

UTC_MS = pl.Datetime("ms", "UTC")

CANDLE_SCHEMA: dict[str, pl.DataType] = {
    "open_time": UTC_MS,
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "close_time": UTC_MS,
    "quote_volume": pl.Float64(),
    "trades": pl.Int64(),
    "taker_buy_base": pl.Float64(),
    "taker_buy_quote": pl.Float64(),
}

CANDLE_COLUMNS: tuple[str, ...] = tuple(CANDLE_SCHEMA.keys())


def empty_candles() -> pl.DataFrame:
    """Пустой DataFrame со схемой свечей."""
    return pl.DataFrame(schema=CANDLE_SCHEMA)


# Поддерживаемые таймфреймы -> длительность бара в миллисекундах.
TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
}


def timeframe_to_ms(timeframe: str) -> int:
    """Длительность одного бара в миллисекундах."""
    try:
        return TIMEFRAME_MS[timeframe]
    except KeyError:
        raise ValueError(
            f"Unsupported timeframe: {timeframe!r}. Supported: {', '.join(TIMEFRAME_MS)}"
        ) from None
