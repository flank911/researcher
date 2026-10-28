"""WF-2 — walk-forward оркестратор (честный out-of-sample, улучшение №8).

На каждом фолде (WF-1):

1. **Отбор на train.** Для каждой комбинации параметров прогоняем бэктест на
   train-окне и считаем метрику отбора (``select_by``); выбираем лучшую комбинацию
   — используя **только** данные train (прошлое).
2. **Торговля на test.** Выбранной комбинацией торгуем на test-окне, которое идёт
   строго после train (WF-1 гарантирует ``test.start == train.end``) — утечки
   будущего в отбор нет.
3. **Склейка OOS.** Test-сегменты соединяются в непрерывную out-of-sample equity с
   **переносом баланса** между фолдами (следующий фолд стартует с капитала,
   накопленного предыдущим).

OOS-метрики (BT-7) считаются по склеенной кривой. Стратегия передаётся инстансом
(реестр не нужен), поэтому слой не зависит от аналитики AN-* — метрика отбора может
быть любой числовой из :class:`Metrics` (по умолчанию ``total_return_pct``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any

import polars as pl

from trading_research.backtest.engine import BacktestEngine, BacktestResult
from trading_research.backtest.metrics import DEFAULT_MIN_TRADES, Metrics, compute_metrics
from trading_research.backtest.portfolio import Trade
from trading_research.domain import ExecutionModel, TimeSlice
from trading_research.experiments.runner import run_backtest
from trading_research.experiments.time_splitters import WalkForwardFold
from trading_research.strategies.base import Strategy

# Числовые метрики для отбора (исключаем булев ``reliable``).
_NUMERIC_METRICS = frozenset(f.name for f in fields(Metrics)) - {"reliable"}


@dataclass(frozen=True)
class FoldSelection:
    """Что выбрано на train и как это отработало на test одного фолда."""

    fold: int
    params: dict[str, Any]
    train_score: float
    test_metrics: Metrics


@dataclass(frozen=True)
class WalkForwardResult:
    selections: list[FoldSelection]
    oos_equity: pl.DataFrame  # columns: open_time, equity (склеенная OOS-кривая)
    oos_metrics: Metrics


def run_walk_forward(
    data: pl.DataFrame,
    strategy: Strategy,
    param_combos: Sequence[Mapping[str, Any]],
    folds: Sequence[WalkForwardFold],
    execution: ExecutionModel,
    timeframe: str,
    *,
    select_by: str = "total_return_pct",
    min_trades: int = DEFAULT_MIN_TRADES,
    engine: BacktestEngine | None = None,
) -> WalkForwardResult:
    """Прогнать walk-forward и вернуть OOS-кривую, метрики и выбор по фолдам."""
    if select_by not in _NUMERIC_METRICS:
        raise ValueError(f"select_by {select_by!r} must be a numeric Metrics field")
    if not param_combos:
        raise ValueError("param_combos must be non-empty")

    engine = engine or BacktestEngine()
    data_times: list[datetime] = data["open_time"].to_list()

    running_balance = execution.initial_balance
    selections: list[FoldSelection] = []
    oos_times: list[datetime] = []
    oos_equity: list[float] = []
    oos_trades: list[Trade] = []

    for fold in folds:
        train_sub = _slice_window(data, data_times, fold.train)
        test_sub = _slice_window(data, data_times, fold.test)
        if train_sub is None or test_sub is None:
            continue

        best = _select_on_train(
            strategy,
            train_sub,
            param_combos,
            execution,
            fold.train,
            timeframe,
            select_by,
            min_trades,
            engine,
        )
        if best is None:
            continue
        best_params, train_score = best

        # Торговля на test с перенесённым балансом.
        test_exec = execution.model_copy(update={"initial_balance": running_balance})
        result = run_backtest(strategy, test_sub, best_params, test_exec, engine=engine)
        seg_times, seg_equity, seg_trades = _window_segment(result, fold.test)
        if not seg_equity:
            continue

        running_balance = seg_equity[-1]
        oos_times.extend(seg_times)
        oos_equity.extend(seg_equity)
        oos_trades.extend(seg_trades)
        test_metrics = compute_metrics(
            pl.DataFrame({"equity": seg_equity}), seg_trades, timeframe, min_trades=min_trades
        )
        selections.append(
            FoldSelection(
                fold=fold.fold,
                params=dict(best_params),
                train_score=train_score,
                test_metrics=test_metrics,
            )
        )

    oos_curve = pl.DataFrame({"open_time": oos_times, "equity": oos_equity})
    equity_for_metrics = (
        pl.DataFrame({"equity": oos_equity})
        if oos_equity
        else pl.DataFrame({"equity": [execution.initial_balance]})
    )
    oos_metrics = compute_metrics(equity_for_metrics, oos_trades, timeframe, min_trades=min_trades)
    return WalkForwardResult(selections=selections, oos_equity=oos_curve, oos_metrics=oos_metrics)


def _select_on_train(
    strategy: Strategy,
    train_data: pl.DataFrame,
    param_combos: Sequence[Mapping[str, Any]],
    execution: ExecutionModel,
    train_window: TimeSlice,
    timeframe: str,
    select_by: str,
    min_trades: int,
    engine: BacktestEngine,
) -> tuple[Mapping[str, Any], float] | None:
    """Выбрать лучшую комбинацию по ``select_by`` на train-окне (только прошлое)."""
    best_params: Mapping[str, Any] | None = None
    best_score = float("-inf")
    for params in param_combos:
        result = run_backtest(strategy, train_data, params, execution, engine=engine)
        _, seg_equity, seg_trades = _window_segment(result, train_window)
        if not seg_equity:
            continue
        metrics = compute_metrics(
            pl.DataFrame({"equity": seg_equity}), seg_trades, timeframe, min_trades=min_trades
        )
        value = getattr(metrics, select_by)
        score = float("-inf") if value is None else float(value)
        if best_params is None or score > best_score:
            best_params, best_score = params, score
    if best_params is None:
        return None
    return best_params, best_score


def _slice_window(
    data: pl.DataFrame, data_times: Sequence[datetime], window: TimeSlice
) -> pl.DataFrame | None:
    """Вырезать из данных окно ``[start, end)`` с warmup-баром прогрева перед ним."""
    in_window = [i for i, t in enumerate(data_times) if window.start <= t < window.end]
    if not in_window:
        return None
    first, last = in_window[0], in_window[-1]
    warm_start = max(0, first - window.warmup_bars)
    return data.slice(warm_start, last - warm_start + 1)


def _window_segment(
    result: BacktestResult, window: TimeSlice
) -> tuple[list[datetime], list[float], list[Trade]]:
    """Отфильтровать equity/сделки прогона к барам окна ``[start, end)`` (без warmup)."""
    times: list[datetime] = result.equity_curve["open_time"].to_list()
    equity: list[float] = [float(x) for x in result.equity_curve["equity"].to_list()]
    idx = [i for i, t in enumerate(times) if window.start <= t < window.end]
    seg_times = [times[i] for i in idx]
    seg_equity = [equity[i] for i in idx]
    seg_trades = [tr for tr in result.trades if window.start <= tr.entry_time < window.end]
    return seg_times, seg_equity, seg_trades


__all__ = ["FoldSelection", "WalkForwardResult", "run_walk_forward"]
