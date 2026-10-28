"""AN-4 — бизнес-логика команды ``plot-heatmap``.

Читает каноническую таблицу эксперимента (её пишет ``run-experiment``) и рендерит
heatmap ``x × y`` по выбранной метрике в PNG.
"""

from __future__ import annotations

from pathlib import Path

from trading_research.analytics.plots import heatmap_to_png
from trading_research.analytics.queries import load_performance_table


def plot_experiment_heatmap(
    results_path: str | Path,
    out_path: str | Path,
    *,
    x: str,
    y: str,
    metric: str = "total_return_pct",
    agg: str = "median",
    symbol: str | None = None,
    timeframe: str | None = None,
    time_slice: str | None = None,
) -> Path:
    """Построить heatmap из таблицы результатов эксперимента и сохранить PNG."""
    table = load_performance_table(results_path)
    return heatmap_to_png(
        table,
        out_path,
        x=x,
        y=y,
        metric=metric,
        agg=agg,
        symbol=symbol,
        timeframe=timeframe,
        time_slice=time_slice,
    )


__all__ = ["plot_experiment_heatmap"]
