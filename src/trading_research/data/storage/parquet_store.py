"""DATA-2 — партиционированный Parquet-store для OHLCV и смежных рядов.

Схема путей (Hive-style, читаемая и совместимая с DuckDB/pyarrow):

    {root}/{dataset}/exchange=.../market=.../symbol=.../timeframe=.../
        year=YYYY/month=MM/{dataset}.parquet

Свойства:

- **Идемпотентная дозапись.** Повторная запись того же окна не плодит дубли:
  внутри партиции строки склеиваются и дедуплицируются по ключу времени
  (``open_time``), при коллизии побеждает **новая** запись.
- **Чтение диапазона через несколько партиций** склеивается в один отсортированный
  DataFrame; партиции вне запрошенного ``[start, end]`` не читаются (month-level
  pruning).
- **Сохранность типов и UTC.** На записи данные приводятся к канонической схеме
  (``open_time`` — ``Datetime("ms", "UTC")``), поэтому round-trip не теряет типы.

Для funding / open_interest / long_short_ratio предусмотрены методы-плейсхолдеры
поверх того же движка (схема выводится из переданного DataFrame; полные схемы будут
зафиксированы в REG-1).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from trading_research.data.schema import CANDLE_SCHEMA, UTC_MS

CANDLES = "candles"
FUNDING = "funding"
OPEN_INTEREST = "open_interest"
LONG_SHORT_RATIO = "long_short_ratio"

_TIME_COL = "open_time"


class ParquetStore:
    """Файловый партиционированный store поверх Parquet."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # --- OHLCV candles ------------------------------------------------------

    def write_candles(
        self,
        df: pl.DataFrame,
        *,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
    ) -> None:
        """Идемпотентно записать свечи, приводя их к канонической схеме."""
        self._write(
            CANDLES,
            df,
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            schema=CANDLE_SCHEMA,
        )

    def read_candles(
        self,
        *,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Прочитать свечи за ``[start, end]`` (границы включительно)."""
        return self._read(
            CANDLES,
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            schema=CANDLE_SCHEMA,
        )

    # --- placeholders: funding / open_interest / long_short_ratio -----------

    def write_series(
        self,
        dataset: str,
        df: pl.DataFrame,
        *,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
    ) -> None:
        """Записать произвольный временной ряд (funding/OI/LS ratio).

        Схема выводится из ``df``; требуется лишь колонка ``open_time`` типа
        ``Datetime`` с таймзоной UTC.
        """
        self._write(
            dataset,
            df,
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            schema=None,
        )

    def read_series(
        self,
        dataset: str,
        *,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Прочитать произвольный временной ряд за ``[start, end]``."""
        return self._read(
            dataset,
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            schema=None,
        )

    # --- core ---------------------------------------------------------------

    def _write(
        self,
        dataset: str,
        df: pl.DataFrame,
        *,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
        schema: dict[str, pl.DataType] | None,
    ) -> None:
        if df.is_empty():
            return
        df = self._coerce(df, schema)
        partition_root = self._series_dir(dataset, exchange, market, symbol, timeframe)

        work = df.with_columns(
            pl.col(_TIME_COL).dt.year().alias("__year"),
            pl.col(_TIME_COL).dt.month().alias("__month"),
        )
        for (year, month), part in work.group_by("__year", "__month"):
            part = part.drop("__year", "__month")
            path = (
                partition_root
                / f"year={int(year)}"
                / f"month={int(month):02d}"
                / f"{dataset}.parquet"
            )
            merged = self._upsert(path, part)
            path.parent.mkdir(parents=True, exist_ok=True)
            merged.write_parquet(path)

    @staticmethod
    def _upsert(path: Path, incoming: pl.DataFrame) -> pl.DataFrame:
        """Склеить с существующей партицией, убрав дубли по ``open_time``.

        При коллизии времени побеждает новая запись (``keep="last"`` после
        конкатенации existing → incoming).
        """
        if path.exists():
            existing = pl.read_parquet(path)
            incoming = pl.concat([existing, incoming], how="vertical")
        return incoming.unique(subset=_TIME_COL, keep="last").sort(_TIME_COL)

    def _read(
        self,
        dataset: str,
        *,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
        start: datetime | None,
        end: datetime | None,
        schema: dict[str, pl.DataType] | None,
    ) -> pl.DataFrame:
        partition_root = self._series_dir(dataset, exchange, market, symbol, timeframe)
        start = _to_utc(start)
        end = _to_utc(end)

        frames: list[pl.DataFrame] = []
        for path in sorted(partition_root.glob(f"year=*/month=*/{dataset}.parquet")):
            ym = _parse_year_month(path)
            if ym is None or not _partition_overlaps(ym[0], ym[1], start, end):
                continue
            frames.append(pl.read_parquet(path))

        if not frames:
            return pl.DataFrame(schema=schema) if schema is not None else pl.DataFrame()

        out = pl.concat(frames, how="vertical").sort(_TIME_COL)
        if start is not None:
            out = out.filter(pl.col(_TIME_COL) >= start)
        if end is not None:
            out = out.filter(pl.col(_TIME_COL) <= end)
        return out

    def _series_dir(
        self, dataset: str, exchange: str, market: str, symbol: str, timeframe: str
    ) -> Path:
        return (
            self.root
            / dataset
            / f"exchange={exchange}"
            / f"market={market}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
        )

    @staticmethod
    def _coerce(df: pl.DataFrame, schema: dict[str, pl.DataType] | None) -> pl.DataFrame:
        if _TIME_COL not in df.columns:
            raise ValueError(f"dataframe missing required time column {_TIME_COL!r}")
        if schema is not None:
            missing = [c for c in schema if c not in df.columns]
            if missing:
                raise ValueError(f"dataframe missing required columns: {missing}")
            return df.select([pl.col(c).cast(dt) for c, dt in schema.items()])
        # Без схемы гарантируем только корректный тип ключа времени.
        if df.schema[_TIME_COL] != UTC_MS:
            df = df.with_columns(pl.col(_TIME_COL).cast(UTC_MS))
        return df


def _to_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _parse_year_month(path: Path) -> tuple[int, int] | None:
    year: int | None = None
    month: int | None = None
    for part in path.parts:
        if part.startswith("year="):
            year = int(part[len("year=") :])
        elif part.startswith("month="):
            month = int(part[len("month=") :])
    if year is None or month is None:
        return None
    return year, month


def _partition_overlaps(
    year: int, month: int, start: datetime | None, end: datetime | None
) -> bool:
    """Пересекает ли партиция (календарный месяц) запрошенный ``[start, end]``."""
    month_start = datetime(year, month, 1, tzinfo=UTC)
    next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
    month_end = datetime(next_year, next_month, 1, tzinfo=UTC)  # эксклюзивно
    if end is not None and month_start > end:
        return False
    if start is not None and month_end <= start:  # noqa: SIM103
        return False
    return True
