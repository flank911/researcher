"""EXP-4 — хранилище результатов: тяжёлые артефакты в Parquet + каталог в SQLite.

Разделение ответственности (улучшение №11):

- **Тяжёлые артефакты** (``trades``, ``equity_curve``) — партиционированный Parquet
  по рану: ``results/run_hash=<hash>/{trades,equity}.parquet``. Каждый ран пишет в
  свою директорию, поэтому конкурентная запись артефактов не конфликтует.
- **Сводка** (индекс ранов ``runs`` + метрики ``run_metrics``) — SQLite-каталог.
  Открывается в режиме WAL с ``busy_timeout``, так что несколько писателей
  сериализуются без ошибок «database is locked». Для многопроцессной записи
  (EXP-5) окно первичного создания БД дополнительно защищено ретраем на lock.

Индекс рана хранит идентичность (``run_hash``, стратегия, символ, таймфрейм, слайс,
``dataset_hash``, git-commit), статус, пути к артефактам и текст ошибки. Запись
статуса/ошибки покрывает «Запись логов/ошибок run» из тикета.

Связь с DATA-4: ``dataset_hash`` — контентный хэш датасета — хранится здесь как
входной идентификатор (уже протянут через ``RunSpec`` из EXP-3). Каталог датасетов
(DATA-4) ведёт сами версии данных; здесь мы лишь ссылаемся на их хэш.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import polars as pl

from trading_research.backtest.metrics import Metrics
from trading_research.backtest.portfolio import Trade
from trading_research.data.schema import UTC_MS
from trading_research.experiments.cache import RunSpec
from trading_research.hashing import canonical_json

_BUSY_TIMEOUT_MS = 30_000
# EXP-5: при конкурентной записи из нескольких процессов окно первичного
# создания БД (до установления WAL) не покрывается busy_timeout — ретраим.
_MAX_WRITE_RETRIES = 12
_RETRY_BACKOFF_S = 0.05


def _is_locked_error(exc: sqlite3.OperationalError) -> bool:
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


TRADE_SCHEMA: dict[str, pl.DataType] = {
    "entry_time": UTC_MS,
    "exit_time": UTC_MS,
    "side": pl.String(),
    "entry_price": pl.Float64(),
    "exit_price": pl.Float64(),
    "qty": pl.Float64(),
    "pnl": pl.Float64(),
    "pnl_pct": pl.Float64(),
    "fees": pl.Float64(),
    "reason": pl.String(),
}

_METRIC_COLUMNS = (
    "total_return_pct",
    "max_drawdown_pct",
    "sharpe",
    "sortino",
    "profit_factor",
    "win_rate",
    "avg_trade_pct",
    "expectancy",
    "recovery_factor",
    "trades_count",
    "annualization_factor",
    "min_trades",
    "reliable",
)


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class RunRecord:
    """Строка каталога: идентичность рана + статус + пути + (опционально) метрики."""

    run_hash: str
    strategy: str
    symbol: str
    timeframe: str
    slice_id: str
    dataset_hash: str
    git_commit: str | None
    params: dict[str, object]
    status: RunStatus
    trades_path: str | None
    equity_path: str | None
    error: str | None
    metrics: Metrics | None


def trades_to_df(trades: list[Trade]) -> pl.DataFrame:
    """Список сделок → DataFrame с канонической схемой (enum-поля как строки)."""
    return pl.DataFrame(
        {
            "entry_time": [t.entry_time for t in trades],
            "exit_time": [t.exit_time for t in trades],
            "side": [t.side.value for t in trades],
            "entry_price": [t.entry_price for t in trades],
            "exit_price": [t.exit_price for t in trades],
            "qty": [t.qty for t in trades],
            "pnl": [t.pnl for t in trades],
            "pnl_pct": [t.pnl_pct for t in trades],
            "fees": [t.fees for t in trades],
            "reason": [t.reason.value for t in trades],
        },
        schema=TRADE_SCHEMA,
    )


class ResultsStore:
    """Хранилище результатов ранов (Parquet-артефакты + SQLite-каталог)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.results_dir = self.root / "results"
        self.db_path = self.root / "catalog.sqlite"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=_BUSY_TIMEOUT_MS / 1000)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        finally:
            conn.close()

    def _write(self, action: Callable[[sqlite3.Connection], None]) -> None:
        """Выполнить запись с ретраем на 'database is locked' (многопроцессность)."""
        last: sqlite3.OperationalError | None = None
        for attempt in range(_MAX_WRITE_RETRIES):
            try:
                with self._connect() as conn:
                    action(conn)
                    conn.commit()
                return
            except sqlite3.OperationalError as exc:
                if not _is_locked_error(exc):
                    raise
                last = exc
                time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
        assert last is not None  # noqa: S101 — инвариант цикла
        raise last

    def _init_db(self) -> None:
        metric_cols = ",\n".join(f"{c} REAL" for c in _METRIC_COLUMNS)

        def _create(conn: sqlite3.Connection) -> None:
            conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS runs (
                    run_hash TEXT PRIMARY KEY,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    slice_id TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    git_commit TEXT,
                    params_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    trades_path TEXT,
                    equity_path TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS run_metrics (
                    run_hash TEXT PRIMARY KEY,
                    {metric_cols},
                    FOREIGN KEY (run_hash) REFERENCES runs (run_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_runs_dataset ON runs (dataset_hash);
                CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status);
                """
            )

        self._write(_create)

    def _run_dir(self, run_hash: str) -> Path:
        return self.results_dir / f"run_hash={run_hash}"

    def save_run(
        self,
        spec: RunSpec,
        metrics: Metrics | None,
        trades: list[Trade],
        equity_curve: pl.DataFrame,
        *,
        status: RunStatus = RunStatus.COMPLETED,
        error: str | None = None,
    ) -> str:
        """Сохранить результат рана: артефакты в Parquet, сводку в каталог.

        Возвращает ``run_hash``. Идемпотентно: повторное сохранение того же рана
        перезаписывает его артефакты и строки каталога.
        """
        run_hash = spec.run_hash()
        trades_path: str | None = None
        equity_path: str | None = None

        if status is RunStatus.COMPLETED:
            run_dir = self._run_dir(run_hash)
            run_dir.mkdir(parents=True, exist_ok=True)
            trades_file = run_dir / "trades.parquet"
            equity_file = run_dir / "equity.parquet"
            trades_to_df(trades).write_parquet(trades_file)
            equity_curve.write_parquet(equity_file)
            trades_path = str(trades_file)
            equity_path = str(equity_file)

        now = datetime.now(UTC).isoformat()

        def _persist(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO runs (run_hash, strategy, symbol, timeframe, slice_id,
                    dataset_hash, git_commit, params_json, status, trades_path,
                    equity_path, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_hash) DO UPDATE SET
                    status=excluded.status,
                    trades_path=excluded.trades_path,
                    equity_path=excluded.equity_path,
                    error=excluded.error,
                    updated_at=excluded.updated_at
                """,
                (
                    run_hash,
                    spec.strategy,
                    spec.symbol,
                    spec.timeframe,
                    spec.slice_id,
                    spec.dataset_hash,
                    spec.git_commit,
                    canonical_json(spec.params),
                    status.value,
                    trades_path,
                    equity_path,
                    error,
                    now,
                    now,
                ),
            )
            if metrics is not None:
                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO run_metrics (run_hash, {", ".join(_METRIC_COLUMNS)})
                    VALUES ({", ".join("?" for _ in range(len(_METRIC_COLUMNS) + 1))})
                    """,
                    (run_hash, *_metric_values(metrics)),
                )

        self._write(_persist)
        return run_hash

    def get_run(self, run_hash: str) -> RunRecord | None:
        """Найти ран по ``run_hash`` (метрики + пути к артефактам)."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            run_row = conn.execute("SELECT * FROM runs WHERE run_hash = ?", (run_hash,)).fetchone()
            if run_row is None:
                return None
            metric_row = conn.execute(
                "SELECT * FROM run_metrics WHERE run_hash = ?", (run_hash,)
            ).fetchone()
        return _row_to_record(run_row, metric_row)

    def all_runs(self) -> list[RunRecord]:
        """Все раны каталога с метриками (для аналитики, AN-1)."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            run_rows = conn.execute("SELECT * FROM runs ORDER BY run_hash").fetchall()
            metric_rows = {
                r["run_hash"]: r for r in conn.execute("SELECT * FROM run_metrics").fetchall()
            }
        return [_row_to_record(rr, metric_rows.get(rr["run_hash"])) for rr in run_rows]

    def load_trades(self, run_hash: str) -> pl.DataFrame:
        """Прочитать сделки рана (пустой DataFrame со схемой, если артефакта нет)."""
        path = self._run_dir(run_hash) / "trades.parquet"
        if not path.exists():
            return pl.DataFrame(schema=TRADE_SCHEMA)
        return pl.read_parquet(path)

    def load_equity(self, run_hash: str) -> pl.DataFrame:
        """Прочитать equity-кривую рана (пустой DataFrame, если артефакта нет)."""
        path = self._run_dir(run_hash) / "equity.parquet"
        if not path.exists():
            return pl.DataFrame()
        return pl.read_parquet(path)

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        return int(row[0])


def _metric_values(metrics: Metrics) -> tuple[object, ...]:
    return (
        metrics.total_return_pct,
        metrics.max_drawdown_pct,
        metrics.sharpe,
        metrics.sortino,
        metrics.profit_factor,
        metrics.win_rate,
        metrics.avg_trade_pct,
        metrics.expectancy,
        metrics.recovery_factor,
        float(metrics.trades_count),
        metrics.annualization_factor,
        float(metrics.min_trades),
        float(metrics.reliable),
    )


def _row_to_record(run_row: sqlite3.Row, metric_row: sqlite3.Row | None) -> RunRecord:
    metrics: Metrics | None = None
    if metric_row is not None:
        metrics = Metrics(
            total_return_pct=metric_row["total_return_pct"],
            max_drawdown_pct=metric_row["max_drawdown_pct"],
            sharpe=metric_row["sharpe"],
            sortino=metric_row["sortino"],
            profit_factor=metric_row["profit_factor"],
            win_rate=metric_row["win_rate"],
            avg_trade_pct=metric_row["avg_trade_pct"],
            expectancy=metric_row["expectancy"],
            recovery_factor=metric_row["recovery_factor"],
            trades_count=int(metric_row["trades_count"]),
            annualization_factor=metric_row["annualization_factor"],
            min_trades=int(metric_row["min_trades"]),
            reliable=bool(metric_row["reliable"]),
        )
    return RunRecord(
        run_hash=run_row["run_hash"],
        strategy=run_row["strategy"],
        symbol=run_row["symbol"],
        timeframe=run_row["timeframe"],
        slice_id=run_row["slice_id"],
        dataset_hash=run_row["dataset_hash"],
        git_commit=run_row["git_commit"],
        params=json.loads(run_row["params_json"]),
        status=RunStatus(run_row["status"]),
        trades_path=run_row["trades_path"],
        equity_path=run_row["equity_path"],
        error=run_row["error"],
        metrics=metrics,
    )
