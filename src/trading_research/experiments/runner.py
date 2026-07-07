"""EXP-2 — раннер одного бэктеста.

Тонкая обёртка «стратегия → сигналы → движок». Изолирует связку генерации
сигналов и событийного прогона, чтобы оркестратор (``orchestrator.py``) не знал
деталей ни того, ни другого.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from trading_research.backtest.engine import BacktestEngine, BacktestResult
from trading_research.domain import ExecutionModel
from trading_research.strategies.base import Strategy


def run_backtest(
    strategy: Strategy,
    data: pl.DataFrame,
    params: Mapping[str, Any],
    execution: ExecutionModel,
    *,
    engine: BacktestEngine | None = None,
) -> BacktestResult:
    """Прогнать один бэктест: сгенерировать сигналы и исполнить их движком.

    ``data`` — непрерывный OHLCV-ряд (включая бары прогрева в начале истории).
    """
    signals = strategy.generate_signals(data, params)
    return (engine or BacktestEngine()).run(signals, execution)
