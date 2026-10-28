"""AN-1 — центральная аналитическая таблица ``parameter_performance_by_slice``.

Материализует одну строку на ``(эксперимент × символ × таймфрейм × параметры ×
временное окно)`` с метриками прогона и слотами под фичи режима (заполняются
позже в E6/REG — здесь это null-плейсхолдеры). Таблица — основа для AN-2..5:
ранжирования, stability-score, хитмапов.

Строится из результатов экспериментов (``ResultsStore``, EXP-4) либо из готовых
``PerformanceRecord``. Поверх таблицы — срезы ``params ↔ time / symbol / timeframe``
(DuckDB-группировки по ``param_hash`` и выбранной оси).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from trading_research.backtest.metrics import Metrics
from trading_research.data.schema import UTC_MS
from trading_research.domain import TimeSlice
from trading_research.experiments.results_store import ResultsStore
from trading_research.hashing import canonical_json, stable_hash

# Метрики (в порядке полей Metrics) + их типы в таблице.
_METRIC_TYPES: dict[str, pl.DataType] = {
    "total_return_pct": pl.Float64(),
    "max_drawdown_pct": pl.Float64(),
    "sharpe": pl.Float64(),
    "sortino": pl.Float64(),
    "profit_factor": pl.Float64(),
    "win_rate": pl.Float64(),
    "avg_trade_pct": pl.Float64(),
    "expectancy": pl.Float64(),
    "recovery_factor": pl.Float64(),
    "trades_count": pl.Int64(),
    "annualization_factor": pl.Float64(),
    "min_trades": pl.Int64(),
    "reliable": pl.Boolean(),
}
METRIC_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Metrics))

# Числовые метрики, по которым можно агрегировать в срезах.
QUERYABLE_METRICS: frozenset[str] = frozenset(
    name for name, dt in _METRIC_TYPES.items() if dt != pl.Boolean()
)

# Слоты под фичи рыночного режима (E6/REG); пока null.
REGIME_SLOTS: tuple[str, ...] = (
    "market_return_pct",
    "volatility",
    "trend_strength",
    "funding_avg",
    "oi_change_pct",
    "ls_ratio_avg",
    "regime_label",
)

PERFORMANCE_SCHEMA: dict[str, pl.DataType] = {
    "experiment_id": pl.String(),
    "strategy_name": pl.String(),
    "symbol": pl.String(),
    "timeframe": pl.String(),
    "time_slice_id": pl.String(),
    "time_slice_kind": pl.String(),
    "time_slice_start": UTC_MS,
    "time_slice_end": UTC_MS,
    "time_slice_role": pl.String(),
    "param_hash": pl.String(),
    "params_json": pl.String(),
    **_METRIC_TYPES,
    **{slot: (pl.String() if slot == "regime_label" else pl.Float64()) for slot in REGIME_SLOTS},
}
PERFORMANCE_COLUMNS: tuple[str, ...] = tuple(PERFORMANCE_SCHEMA)

# Оси среза params ↔ ... → колонка таблицы.
_AXES: dict[str, str] = {
    "time": "time_slice_id",
    "symbol": "symbol",
    "timeframe": "timeframe",
}


@dataclass(frozen=True)
class PerformanceRecord:
    """Одна ячейка анализа: параметры на конкретном окне + метрики прогона."""

    experiment_id: str
    strategy_name: str
    symbol: str
    timeframe: str
    time_slice: TimeSlice
    params: dict[str, Any]
    metrics: Metrics


def param_hash(params: dict[str, Any], *, length: int = 16) -> str:
    """Стабильный хэш набора параметров (не зависит от порядка ключей, INF-3)."""
    return stable_hash(params, length=length)


def build_parameter_performance(records: Iterable[PerformanceRecord]) -> pl.DataFrame:
    """Собрать каноническую таблицу ``parameter_performance_by_slice``."""
    rows: list[dict[str, Any]] = []
    for r in records:
        sl = r.time_slice
        rows.append(
            {
                "experiment_id": r.experiment_id,
                "strategy_name": r.strategy_name,
                "symbol": r.symbol,
                "timeframe": r.timeframe,
                "time_slice_id": sl.id,
                "time_slice_kind": sl.kind.value,
                "time_slice_start": sl.start,
                "time_slice_end": sl.end,
                "time_slice_role": sl.role.value,
                "param_hash": param_hash(r.params),
                "params_json": canonical_json(r.params),
                **asdict(r.metrics),
                **dict.fromkeys(REGIME_SLOTS),  # None-плейсхолдеры режима
            }
        )
    return pl.DataFrame(rows, schema=PERFORMANCE_SCHEMA)


def performance_from_store(
    store: ResultsStore,
    slices: Sequence[TimeSlice],
    *,
    experiment_id: str,
) -> pl.DataFrame:
    """Построить таблицу из каталога ``ResultsStore`` (раны с метриками).

    ``slices`` сопоставляются с ``run.slice_id`` для заполнения ``time_slice_*``.
    Раны без метрик пропускаются.
    """
    slice_map = {s.id: s for s in slices}
    records: list[PerformanceRecord] = []
    for run in store.all_runs():
        if run.metrics is None:
            continue
        sl = slice_map.get(run.slice_id)
        if sl is None:
            raise KeyError(f"slice_id {run.slice_id!r} not found in provided slices")
        records.append(
            PerformanceRecord(
                experiment_id=experiment_id,
                strategy_name=run.strategy,
                symbol=run.symbol,
                timeframe=run.timeframe,
                time_slice=sl,
                params=run.params,
                metrics=run.metrics,
            )
        )
    return build_parameter_performance(records)


def slice_param_performance(
    table: pl.DataFrame, *, axis: str, metric: str = "total_return_pct"
) -> pl.DataFrame:
    """Срез ``params ↔ axis``: среднее ``metric`` по ``(param_hash × ось)``.

    ``axis`` — один из ``time`` / ``symbol`` / ``timeframe``.
    """
    if axis not in _AXES:
        raise ValueError(f"unknown axis {axis!r}; expected one of {sorted(_AXES)}")
    if metric not in QUERYABLE_METRICS:
        raise ValueError(
            f"metric {metric!r} not queryable; expected one of {sorted(QUERYABLE_METRICS)}"
        )
    col = _AXES[axis]
    con = duckdb.connect()
    try:
        con.register("perf", table.to_arrow())
        query = (
            f"SELECT param_hash, any_value(params_json) AS params_json, "  # noqa: S608 — col/metric из whitelist
            f"{col} AS axis_value, avg({metric}) AS {metric}_avg, count(*) AS n "
            f"FROM perf GROUP BY param_hash, {col} ORDER BY param_hash, {col}"
        )
        return con.execute(query).pl()
    finally:
        con.close()


def rank_params(
    table: pl.DataFrame,
    *,
    metric: str = "total_return_pct",
    top: int = 20,
    by: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    descending: bool = True,
) -> pl.DataFrame:
    """Top-N комбинаций параметров по ``metric`` (AN-2).

    ``by`` (``time``/``symbol``/``timeframe``) ранжирует внутри каждого значения
    оси (top-N на группу); без ``by`` — глобальный рейтинг. ``symbol``/``timeframe``
    фильтруют строки до ранжирования. Значение метрики усредняется по строкам,
    попавшим в группу (``param_hash`` × ось). Ранг 1 — лучший.
    """
    if metric not in QUERYABLE_METRICS:
        raise ValueError(
            f"metric {metric!r} not queryable; expected one of {sorted(QUERYABLE_METRICS)}"
        )
    if by is not None and by not in _AXES:
        raise ValueError(f"unknown axis {by!r}; expected one of {sorted(_AXES)}")
    if top <= 0:
        raise ValueError(f"top must be > 0, got {top}")

    df = table
    if symbol is not None:
        df = df.filter(pl.col("symbol") == symbol)
    if timeframe is not None:
        df = df.filter(pl.col("timeframe") == timeframe)

    group_keys = ["param_hash"] if by is None else [_AXES[by], "param_hash"]
    agg = df.group_by(group_keys).agg(
        pl.first("params_json").alias("params_json"),
        pl.col(metric).mean().alias(metric),
        pl.len().alias("n"),
    )

    if by is None:
        ranked = agg.sort(metric, descending=descending, nulls_last=True).head(top)
        return ranked.with_row_index("rank", offset=1)

    axis_col = _AXES[by]
    ranked = agg.with_columns(
        pl.col(metric)
        .rank(method="ordinal", descending=descending)
        .over(axis_col)
        .cast(pl.Int64)
        .alias("rank")
    )
    return ranked.filter(pl.col("rank") <= top).sort([axis_col, "rank"])


def save_performance_table(table: pl.DataFrame, path: str | Path) -> Path:
    """Сохранить таблицу производительности в Parquet (создаёт директории)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(p)
    return p


def load_performance_table(path: str | Path) -> pl.DataFrame:
    """Прочитать таблицу производительности из Parquet."""
    return pl.read_parquet(path)


def params_by_time(table: pl.DataFrame, *, metric: str = "total_return_pct") -> pl.DataFrame:
    """Срез params ↔ время (по ``time_slice_id``)."""
    return slice_param_performance(table, axis="time", metric=metric)


def params_by_symbol(table: pl.DataFrame, *, metric: str = "total_return_pct") -> pl.DataFrame:
    """Срез params ↔ символ."""
    return slice_param_performance(table, axis="symbol", metric=metric)


def params_by_timeframe(table: pl.DataFrame, *, metric: str = "total_return_pct") -> pl.DataFrame:
    """Срез params ↔ таймфрейм."""
    return slice_param_performance(table, axis="timeframe", metric=metric)


__all__ = [
    "PERFORMANCE_COLUMNS",
    "PERFORMANCE_SCHEMA",
    "QUERYABLE_METRICS",
    "REGIME_SLOTS",
    "PerformanceRecord",
    "build_parameter_performance",
    "load_performance_table",
    "param_hash",
    "params_by_symbol",
    "params_by_time",
    "params_by_timeframe",
    "performance_from_store",
    "rank_params",
    "save_performance_table",
    "slice_param_performance",
]
