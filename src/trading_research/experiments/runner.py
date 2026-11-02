"""EXP-2 — раннер одного бэктеста.

Тонкая обёртка «стратегия → сигналы → движок». Изолирует связку генерации
сигналов и событийного прогона, чтобы оркестратор (``orchestrator.py``) не знал
деталей ни того, ни другого.

**WF-4 — бары прогрева греют индикаторы, но не торгуются.** Прогрев нужен затем,
чтобы на первом баре окна индикатор уже имел значение; торговля на прогреве — не
часть окна, а подмена его начальных условий. Раньше движок торговал прогрев
наравне с окном, и это было видно по цифрам: позиция, открытая на прогреве, меняла
баланс к началу окна, а при ``position_size_type: percent_balance`` — и размер
сделок внутри окна. Из-за того же на склеенной OOS-кривой между фолдами возникал
разрыв: сегмент окна начинался не с перенесённого баланса, а с него же плюс
результат торговли на прогреве. Поэтому ``warmup_bars`` здесь глушит сигналы на
первых барах: индикатор считается по всему ряду, сделок до окна нет.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from trading_research.backtest.engine import BacktestEngine, BacktestResult
from trading_research.domain import ExecutionModel
from trading_research.strategies.base import SIGNAL_COLUMNS, Strategy


def run_backtest(
    strategy: Strategy,
    data: pl.DataFrame,
    params: Mapping[str, Any],
    execution: ExecutionModel,
    *,
    engine: BacktestEngine | None = None,
    warmup_bars: int = 0,
) -> BacktestResult:
    """Прогнать один бэктест: сгенерировать сигналы и исполнить их движком.

    ``data`` — непрерывный OHLCV-ряд (включая бары прогрева в начале истории).
    ``warmup_bars`` — сколько первых баров считать прогревом: индикаторы на них
    считаются, сигналы глушатся, сделок на них не возникает (WF-4). Движок всё
    равно выдаёт по точке equity на каждый бар, то есть длина кривой не меняется —
    на этом держится нарезка окон в ``walk_forward``.
    """
    if warmup_bars < 0:
        raise ValueError(f"warmup_bars must be >= 0, got {warmup_bars}")
    signals = strategy.generate_signals(data, params)
    if warmup_bars:
        signals = _mute_signals(signals, warmup_bars)
    return (engine or BacktestEngine()).run(signals, execution)


def _mute_signals(signals: pl.DataFrame, warmup_bars: int) -> pl.DataFrame:
    """Погасить сигнальные колонки на первых ``warmup_bars`` барах."""
    if warmup_bars >= signals.height:
        # Окно целиком внутри прогрева — такого быть не должно, но тихо торговать
        # «прогретую» часть хуже, чем честно ничего не торговать.
        warmup_bars = signals.height
    index = pl.int_range(pl.len(), dtype=pl.UInt32)
    return signals.with_columns(
        [
            pl.when(index < warmup_bars).then(pl.lit(value=False)).otherwise(pl.col(col)).alias(col)
            for col in SIGNAL_COLUMNS
        ]
    )
