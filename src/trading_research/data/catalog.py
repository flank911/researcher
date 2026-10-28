"""DATA-4 — каталог датасетов + контентное версионирование (data_hash).

Каждый снапшот данных ``[symbol, timeframe, range]`` регистрируется в SQLite-каталоге
с **контентным** ``data_hash`` (INF-3): нормализованный, стабильный, не зависящий от
порядка строк. Идентификатор версии ``id`` привязан к идентичности источника
(exchange/market/symbol/timeframe) и содержимому (``data_hash``).

Политика дозагрузки: докачка свечей меняет содержимое → новый ``data_hash`` → новая
строка каталога. Старые версии **не мутируются**, поэтому ссылки прошлых прогонов
(по ``id`` или ``data_hash``) остаются валидными.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from trading_research.data.schema import CANDLE_COLUMNS
from trading_research.domain import DatasetRef
from trading_research.hashing import stable_hash

_TIME_COL = "open_time"


def dataset_hash(candles: pl.DataFrame, *, length: int = 16) -> str:
    """Контентный хэш свечей: не зависит от порядка строк, стабилен между запусками.

    Нормализация: сортировка по ``open_time`` и выбор канонических колонок свечей.
    Хэширование идёт через INF-3 (``stable_hash``), который нормализует float-шум и
    приводит datetime к ISO-строкам.
    """
    if candles.is_empty():
        raise ValueError("cannot hash empty candles dataframe")
    columns = [c for c in CANDLE_COLUMNS if c in candles.columns]
    normalized = candles.sort(_TIME_COL).select(columns)
    payload = {col: normalized[col].to_list() for col in normalized.columns}
    return stable_hash(payload, length=length)


def _dataset_id(
    exchange: str, market: str, symbol: str, timeframe: str, data_hash: str, *, length: int = 16
) -> str:
    """Идентичность версии: источник + содержимое."""
    return stable_hash(
        {
            "exchange": exchange,
            "market": market,
            "symbol": symbol,
            "timeframe": timeframe,
            "data_hash": data_hash,
        },
        length=length,
    )


class DatasetCatalog:
    """SQLite-каталог версий датасетов."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    market TEXT NOT NULL,
                    start_at TEXT NOT NULL,
                    end_at TEXT NOT NULL,
                    data_path TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_datasets_hash ON datasets (data_hash);
                CREATE INDEX IF NOT EXISTS idx_datasets_key ON datasets (symbol, timeframe);
                """
            )
            conn.commit()

    def register(
        self,
        candles: pl.DataFrame,
        *,
        name: str,
        exchange: str,
        market: str,
        symbol: str,
        timeframe: str,
        data_path: str,
    ) -> DatasetRef:
        """Зарегистрировать снапшот данных и вернуть ``DatasetRef``.

        Идемпотентно по содержимому: повторная регистрация идентичных данных из
        того же источника возвращает существующую версию (той же ``id``).
        """
        data_hash = dataset_hash(candles)
        dataset_id = _dataset_id(exchange, market, symbol, timeframe, data_hash)

        existing = self.get(dataset_id)
        if existing is not None:
            return existing

        start_at = candles[_TIME_COL].min()
        end_at = candles[_TIME_COL].max()
        ref = DatasetRef(
            id=dataset_id,
            symbol=symbol,
            timeframe=timeframe,
            exchange=exchange,
            market=market,
            start_at=start_at,  # type: ignore[arg-type]
            end_at=end_at,  # type: ignore[arg-type]
            data_path=data_path,
            data_hash=data_hash,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO datasets (id, name, symbol, timeframe, exchange,
                    market, start_at, end_at, data_path, data_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ref.id,
                    name,
                    ref.symbol,
                    ref.timeframe,
                    ref.exchange,
                    ref.market,
                    ref.start_at.isoformat(),
                    ref.end_at.isoformat(),
                    ref.data_path,
                    ref.data_hash,
                    datetime.now(UTC).isoformat(),
                ),
            )
            conn.commit()
        return ref

    def get(self, dataset_id: str) -> DatasetRef | None:
        """Разрешить ``DatasetRef`` по ``id``."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        return _row_to_ref(row) if row is not None else None

    def get_by_hash(self, data_hash: str) -> DatasetRef | None:
        """Разрешить ``DatasetRef`` по контентному ``data_hash`` (последняя версия)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM datasets WHERE data_hash = ? ORDER BY created_at DESC LIMIT 1",
                (data_hash,),
            ).fetchone()
        return _row_to_ref(row) if row is not None else None

    def versions(self, symbol: str, timeframe: str) -> list[DatasetRef]:
        """Все версии датасета для ``[symbol, timeframe]`` (от старых к новым)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM datasets WHERE symbol = ? AND timeframe = ?
                ORDER BY created_at ASC
                """,
                (symbol, timeframe),
            ).fetchall()
        return [_row_to_ref(row) for row in rows]

    def latest(self, symbol: str, timeframe: str) -> DatasetRef | None:
        """Последняя зарегистрированная версия для ``[symbol, timeframe]``."""
        versions = self.versions(symbol, timeframe)
        return versions[-1] if versions else None


def _row_to_ref(row: sqlite3.Row) -> DatasetRef:
    return DatasetRef(
        id=row["id"],
        symbol=row["symbol"],
        timeframe=row["timeframe"],
        exchange=row["exchange"],
        market=row["market"],
        start_at=datetime.fromisoformat(row["start_at"]),
        end_at=datetime.fromisoformat(row["end_at"]),
        data_path=row["data_path"],
        data_hash=row["data_hash"],
    )
