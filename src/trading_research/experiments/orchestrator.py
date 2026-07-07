"""EXP-2 — оркестратор research-режима: непрерывный прогон + нарезка по слайсам.

Идея (улучшение №2): вместо отдельного бэктеста на каждое окно делаем **один**
непрерывный bar-by-bar прогон на всю историю для каждой комбинации
``(символ × таймфрейм × параметры)``, а equity/сделки затем нарезаем по ``TimeSlice``.
Это снижает число полных прогонов в ``len(slices)`` раз (пример плана: 432 вместо
25 920) и устраняет проблему прогрева индикаторов на границах окон — индикаторы
уже «прогреты» непрерывным прогоном.

Перенос баланса между окнами: базой доходности слайса служит equity на баре
**перед** его началом (для первого слайса — до первой сделки это ``initial_balance``).
За счёт этого пер-слайс доходности телескопируются в доходность всего окна
исследования (см. ``test_orchestrator``).

Режим ``INDEPENDENT`` (флаг конфига ``slice_mode``) прогоняет каждое окно отдельно
со сбросом баланса к ``initial_balance`` — по одному полному прогону на окно.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl

from trading_research.backtest.engine import BacktestEngine
from trading_research.backtest.metrics import DEFAULT_MIN_TRADES, Metrics, compute_metrics
from trading_research.backtest.portfolio import Trade
from trading_research.domain import ExecutionModel, SliceMode, TimeSlice
from trading_research.experiments.runner import run_backtest
from trading_research.strategies.base import Strategy


@dataclass(frozen=True)
class SliceMetrics:
    """Метрики одной комбинации параметров на одном временном окне."""

    slice_id: str
    params: dict[str, Any]
    metrics: Metrics


@dataclass(frozen=True)
class ExperimentResult:
    per_slice: list[SliceMetrics]
    n_full_runs: int
    mode: SliceMode


def count_full_runs(n_combos: int, n_slices: int, mode: SliceMode) -> int:
    """Число полных прогонов движка для заданного режима.

    ``CONTINUOUS`` — один прогон на комбинацию (нарезка постфактум).
    ``INDEPENDENT`` — прогон на каждую пару (комбинация × окно).
    """
    if mode is SliceMode.CONTINUOUS:
        return n_combos
    return n_combos * n_slices


def run_experiment(
    data: pl.DataFrame,
    strategy: Strategy,
    param_combos: Sequence[Mapping[str, Any]],
    slices: Sequence[TimeSlice],
    execution: ExecutionModel,
    timeframe: str,
    *,
    mode: SliceMode = SliceMode.CONTINUOUS,
    min_trades: int = DEFAULT_MIN_TRADES,
) -> ExperimentResult:
    """Прогнать эксперимент и вернуть пер-слайс метрики по каждой комбинации."""
    engine = BacktestEngine()
    per_slice: list[SliceMetrics] = []

    if mode is SliceMode.CONTINUOUS:
        for params in param_combos:
            result = run_backtest(strategy, data, params, execution, engine=engine)
            per_slice.extend(
                segment_by_slice(
                    result.equity_curve,
                    result.trades,
                    slices,
                    params,
                    initial_balance=execution.initial_balance,
                    timeframe=timeframe,
                    min_trades=min_trades,
                )
            )
    else:
        per_slice.extend(
            _run_independent(
                data, strategy, param_combos, slices, execution, timeframe, min_trades, engine
            )
        )

    return ExperimentResult(
        per_slice=per_slice,
        n_full_runs=count_full_runs(len(param_combos), len(slices), mode),
        mode=mode,
    )


def segment_by_slice(
    equity_curve: pl.DataFrame,
    trades: list[Trade],
    slices: Sequence[TimeSlice],
    params: Mapping[str, Any],
    *,
    initial_balance: float,
    timeframe: str,
    min_trades: int = DEFAULT_MIN_TRADES,
) -> list[SliceMetrics]:
    """Нарезать непрерывную equity-кривую на пер-слайс метрики (CONTINUOUS).

    Для каждого окна базой служит equity на последнем баре **до** ``slice.start``
    (перенос баланса); бары прогрева в начале истории лежат до первого окна и в
    метрики не попадают.
    """
    times: list[datetime] = equity_curve["open_time"].to_list()
    equities: list[float] = [float(x) for x in equity_curve["equity"].to_list()]

    out: list[SliceMetrics] = []
    for sl in slices:
        base = initial_balance
        for t, e in zip(times, equities, strict=True):
            if t < sl.start:
                base = e
            else:
                break
        metrics = _slice_metrics(times, equities, trades, sl, base, timeframe, min_trades)
        out.append(SliceMetrics(slice_id=sl.id, params=dict(params), metrics=metrics))
    return out


def _slice_metrics(
    times: Sequence[datetime],
    equities: Sequence[float],
    trades: Sequence[Trade],
    sl: TimeSlice,
    base_equity: float,
    timeframe: str,
    min_trades: int,
) -> Metrics:
    # Сегмент equity: база (перенос баланса) + бары окна [start, end).
    seg_equity = [base_equity]
    seg_equity.extend(e for t, e in zip(times, equities, strict=True) if sl.start <= t < sl.end)
    seg_trades = [tr for tr in trades if sl.start <= tr.entry_time < sl.end]
    equity_df = pl.DataFrame({"equity": seg_equity})
    return compute_metrics(equity_df, seg_trades, timeframe, min_trades=min_trades)


def _run_independent(
    data: pl.DataFrame,
    strategy: Strategy,
    param_combos: Sequence[Mapping[str, Any]],
    slices: Sequence[TimeSlice],
    execution: ExecutionModel,
    timeframe: str,
    min_trades: int,
    engine: BacktestEngine,
) -> list[SliceMetrics]:
    data_times: list[datetime] = data["open_time"].to_list()
    out: list[SliceMetrics] = []
    for params in param_combos:
        for sl in slices:
            in_window = [i for i, t in enumerate(data_times) if sl.start <= t < sl.end]
            if not in_window:
                continue
            first, last = in_window[0], in_window[-1]
            warm_start = max(0, first - sl.warmup_bars)
            sub = data.slice(warm_start, last - warm_start + 1)

            result = run_backtest(strategy, sub, params, execution, engine=engine)
            sub_times: list[datetime] = result.equity_curve["open_time"].to_list()
            sub_equities: list[float] = [float(x) for x in result.equity_curve["equity"].to_list()]
            # Независимое окно: база сброшена к initial_balance, прогрев исключён.
            metrics = _slice_metrics(
                sub_times,
                sub_equities,
                result.trades,
                sl,
                execution.initial_balance,
                timeframe,
                min_trades,
            )
            out.append(SliceMetrics(slice_id=sl.id, params=dict(params), metrics=metrics))
    return out
