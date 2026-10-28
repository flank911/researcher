"""AN-2 — бизнес-логика команды ``analyze-results``.

Читает каноническую таблицу ``parameter_performance_by_slice`` (её пишет
``run-experiment``, AN-1) и возвращает top-N комбинаций параметров по выбранной
метрике, опционально в разрезе time/symbol/timeframe и с фильтрами.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from trading_research.analytics.queries import load_performance_table, rank_params
from trading_research.cli.run_experiment import RESULTS_FILE


def experiment_results_path(results_root: str | Path, experiment: str) -> Path:
    """Путь к таблице результатов эксперимента (как пишет ``run-experiment``)."""
    return Path(results_root) / experiment / RESULTS_FILE


def analyze_results(
    results_path: str | Path,
    *,
    metric: str = "total_return_pct",
    top: int = 20,
    by: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    descending: bool = True,
) -> pl.DataFrame:
    """Загрузить таблицу и вернуть top-N параметров по ``metric``."""
    table = load_performance_table(results_path)
    return rank_params(
        table,
        metric=metric,
        top=top,
        by=by,
        symbol=symbol,
        timeframe=timeframe,
        descending=descending,
    )


__all__ = ["analyze_results", "experiment_results_path"]
