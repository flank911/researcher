"""DATA-5 — слой доступа к данным поверх партиционированного Parquet.

Две точки входа над тем же layout, что пишет :class:`ParquetStore` (DATA-2):

- **Polars lazy** (`scan_candles`) — `scan_parquet` по глобу партиций; чтение
  ленивое, с предикатным/проекционным pushdown, без загрузки всего набора.
- **DuckDB SQL** (`query`) — произвольный SQL поверх тех же файлов через
  `read_parquet`; в запросе файлы доступны как представление ``candles``.

Ключевая функция — выдача **непрерывного ряда с warmup-буфером**
(`read_window`): запрос окна ``[start, end]`` фактически читает
``[start - warmup_bars, end]`` и помечает прогревочную часть колонкой
``is_warmup``. Нижняя граница вычисляется по шагу timeframe, поэтому скан
ограничен окном+прогревом (при пропусках баров прогрев получается best-effort —
столько баров, сколько реально есть в диапазоне).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl

from trading_research.data.schema import CANDLE_COLUMNS, CANDLE_SCHEMA, timeframe_to_ms
from trading_research.data.storage.parquet_store import CANDLES

_TIME_COL = "open_time"
WARMUP_COL = "is_warmup"


class DuckDBStore:
    """Доступ к свечам поверх Parquet: Polars lazy scan + DuckDB SQL."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # --- paths --------------------------------------------------------------

    def _partition_root(self, *, exchange: str, market: str, symbol: str, timeframe: str) -> Path:
        return (
            self.root
            / CANDLES
            / f"exchange={exchange}"
            / f"market={market}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
        )

    def _candle_files(self, **keys: str) -> list[Path]:
        root = self._partition_root(**keys)
        return sorted(root.glob(f"year=*/month=*/{CANDLES}.parquet"))

    # --- polars lazy --------------------------------------------------------

    def scan_candles(
        self, *, exchange: str, market: str, symbol: str, timeframe: str
    ) -> pl.LazyFrame:
        """Ленивый ``scan_parquet`` по всем партициям (отсортирован по времени).

        Возвращает :class:`polars.LazyFrame` — материализация происходит только
        на ``.collect()``, с pushdown фильтров/проекций в Parquet.
        """
        files = self._candle_files(
            exchange=exchange, market=market, symbol=symbol, timeframe=timeframe
        )
        if not files:
            return pl.DataFrame(schema=CANDLE_SCHEMA).lazy()
        return (
            pl.scan_parquet(files, hive_partitioning=False).select(CANDLE_COLUMNS).sort(_TIME_COL)
        )

    def read_window(
        self,
        *,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        warmup_bars: int = 0,
    ) -> pl.DataFrame:
        """Прочитать окно ``[start, end]`` с warmup-буфером из ``warmup_bars`` баров.

        Добавляет булеву колонку ``is_warmup`` (``True`` для баров до ``start``).
        Границы включительны. Нижняя граница скана = ``start - warmup_bars`` шагов
        timeframe, поэтому в память не читается весь файл.
        """
        if warmup_bars < 0:
            raise ValueError(f"warmup_bars must be >= 0, got {warmup_bars}")

        start = _to_utc(start)
        end = _to_utc(end)
        lf = self.scan_candles(exchange=exchange, market=market, symbol=symbol, timeframe=timeframe)

        lower = start
        if start is not None and warmup_bars > 0:
            step = timeframe_to_ms(timeframe)
            lower = start - timedelta(milliseconds=step * warmup_bars)

        if lower is not None:
            lf = lf.filter(pl.col(_TIME_COL) >= lower)
        if end is not None:
            lf = lf.filter(pl.col(_TIME_COL) <= end)

        df = lf.collect()
        flag = ((pl.col(_TIME_COL) < start) if start is not None else pl.lit(False)).alias(
            WARMUP_COL
        )
        return df.with_columns(flag)

    # --- duckdb sql ---------------------------------------------------------

    def query(
        self, sql: str, *, exchange: str, market: str, symbol: str, timeframe: str
    ) -> pl.DataFrame:
        """Выполнить SQL поверх Parquet (DuckDB); файлы доступны как ``candles``.

        Пример: ``store.query("SELECT max(high) AS hi FROM candles", exchange=...,
        ...)``. Возвращает результат как :class:`polars.DataFrame`.
        """
        files = self._candle_files(
            exchange=exchange, market=market, symbol=symbol, timeframe=timeframe
        )
        con = duckdb.connect()
        try:
            if files:
                paths = [_sql_str(str(p)) for p in files]
                source = f"read_parquet([{', '.join(paths)}], hive_partitioning=false)"
            else:
                # Пустой источник с канонической схемой, чтобы SQL не падал.
                source = f"({_empty_candles_sql()})"
            con.execute(f"CREATE VIEW candles AS SELECT * FROM {source}")
            # con.sql(), не con.execute(): в duckdb 2.0 метод .pl() остался только на
            # DuckDBPyRelation, с соединения он убран (INF-6). Аннотация обязательна —
            # в стабах duckdb 2.0 .pl() объявлен как Any.
            result: pl.DataFrame = con.sql(sql).pl()
            return result
        finally:
            con.close()


def _to_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _sql_str(value: str) -> str:
    """Экранировать строковый литерал для SQL (одинарные кавычки)."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _empty_candles_sql() -> str:
    """SELECT без строк с канонической схемой свечей (для пустого источника)."""
    cols = [f"CAST(NULL AS {_sql_type(name)}) AS {name}" for name in CANDLE_SCHEMA]
    return f"SELECT {', '.join(cols)} WHERE 1=0"


def _sql_type(column: str) -> str:
    if column.endswith("_time"):
        return "TIMESTAMP"
    if column == "trades":
        return "BIGINT"
    return "DOUBLE"


__all__ = ["WARMUP_COL", "DuckDBStore"]
