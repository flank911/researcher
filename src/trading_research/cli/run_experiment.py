"""EXP-6 — бизнес-логика команды ``run-experiment``.

Склейка слоёв в один прогон по YAML-конфигу: данные из Parquet-store (DATA-5) →
стратегия (BT-1) → сетка параметров (EXP-1) → временные окна (DATA-6) →
непрерывный прогон с нарезкой по слайсам (EXP-2). Результат — таблица
``parameter_performance_by_slice`` (вход для аналитики AN-1) в Parquet.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import polars as pl

from trading_research.config import load_experiment_config
from trading_research.data.storage.duckdb_store import DuckDBStore
from trading_research.experiments.grid_search import expand_grid
from trading_research.experiments.orchestrator import run_experiment
from trading_research.experiments.time_splitters import generate_time_slices
from trading_research.strategies import get_strategy

RESULTS_FILE = "metrics.parquet"


def run_experiment_from_config(
    config_path: str | Path,
    *,
    data_root: str | Path,
    results_root: str | Path,
) -> Path:
    """Прогнать эксперимент по конфигу и записать пер-слайс метрики в Parquet.

    Возвращает путь к файлу результатов ``{results_root}/{name}/metrics.parquet``.
    """
    cfg = load_experiment_config(config_path)
    strategy = get_strategy(cfg.strategy.name)
    combos = expand_grid(cfg.params_grid)
    store = DuckDBStore(data_root)

    rows: list[dict[str, object]] = []
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
        result = run_experiment(
            data,
            strategy,
            combos,
            slices,
            cfg.execution,
            cfg.data.timeframe,
            mode=cfg.slice_mode,
        )
        for sm in result.per_slice:
            rows.append(
                {
                    "symbol": symbol,
                    "slice_id": sm.slice_id,
                    **sm.params,
                    **asdict(sm.metrics),
                }
            )

    out_dir = Path(results_root) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / RESULTS_FILE
    pl.DataFrame(rows).write_parquet(out_path)
    return out_path


__all__ = ["RESULTS_FILE", "run_experiment_from_config"]
