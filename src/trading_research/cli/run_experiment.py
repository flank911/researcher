"""EXP-6 — бизнес-логика команды ``run-experiment``.

Склейка слоёв в один прогон по YAML-конфигу: данные из Parquet-store (DATA-5) →
стратегия (BT-1) → сетка параметров (EXP-1) → временные окна (DATA-6) →
непрерывный прогон с нарезкой по слайсам (EXP-2). Результат — каноническая таблица
``parameter_performance_by_slice`` (AN-1) в Parquet, готовая для аналитики
(``analyze-results``, AN-2).
"""

from __future__ import annotations

from pathlib import Path

from trading_research.analytics.queries import (
    PerformanceRecord,
    build_parameter_performance,
    save_performance_table,
)
from trading_research.config import load_experiment_config
from trading_research.data.storage.duckdb_store import DuckDBStore
from trading_research.experiments.grid_search import expand_grid
from trading_research.experiments.orchestrator import run_experiment
from trading_research.experiments.time_splitters import generate_time_slices
from trading_research.strategies import get_strategy

RESULTS_FILE = "performance.parquet"


def run_experiment_from_config(
    config_path: str | Path,
    *,
    data_root: str | Path,
    results_root: str | Path,
) -> Path:
    """Прогнать эксперимент по конфигу и записать таблицу AN-1 в Parquet.

    Возвращает путь к ``{results_root}/{name}/performance.parquet``.
    """
    cfg = load_experiment_config(config_path)
    strategy = get_strategy(cfg.strategy.name)
    combos = expand_grid(cfg.params_grid)
    store = DuckDBStore(data_root)

    records: list[PerformanceRecord] = []
    for symbol in cfg.data.symbols:
        data = store.read_window(
            exchange=cfg.data.exchange,
            market=cfg.data.market,
            symbol=symbol,
            timeframe=cfg.data.timeframe,
            start=cfg.data.start,
            end=cfg.data.end,
        )
        slices = generate_time_slices(
            cfg.time_slices.type,
            cfg.data.start,
            cfg.data.end,
            timeframe=cfg.data.timeframe,
            min_bars=cfg.time_slices.min_bars,
        )
        slice_by_id = {sl.id: sl for sl in slices}
        result = run_experiment(
            data,
            strategy,
            combos,
            slices,
            cfg.execution,
            cfg.data.timeframe,
            mode=cfg.slice_mode,
        )
        records.extend(
            PerformanceRecord(
                experiment_id=cfg.name,
                strategy_name=cfg.strategy.name,
                symbol=symbol,
                timeframe=cfg.data.timeframe,
                time_slice=slice_by_id[sm.slice_id],
                params=sm.params,
                metrics=sm.metrics,
            )
            for sm in result.per_slice
        )

    table = build_parameter_performance(records)
    return save_performance_table(table, Path(results_root) / cfg.name / RESULTS_FILE)


__all__ = ["RESULTS_FILE", "run_experiment_from_config"]
