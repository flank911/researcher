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

**RUN-8 — нарезка окон не сканирует историю.** Отбор на train — это
``фолды × комбинации`` прогонов, и всё, что лежит вокруг прогона, умножается на то
же число. Профиль на данных RUN-7 (45 фолдов × 6 комбинаций, 37 224 бара) показал,
что ~35% времени уходило не в движок: ``_slice_window`` искал границы окна
линейным ``enumerate`` по всей истории, ``_window_segment`` материализовал в Python
весь ``open_time`` и всю equity прогона (включая то, что затем выбрасывал), а
метрики получали ``pl.DataFrame`` собранный из Python-списка.

Теперь границы ищет ``bisect`` по тому же списку времён (он сортирован, и сравнение
ровно то же, что было в скане), а сегмент окна — срез фрейма equity по индексам.
Индексы считаются один раз при нарезке окна: движок выдаёт по одной точке equity на
бар входа, в том же порядке, поэтому позиция окна внутри прогона известна заранее и
искать её второй раз не нужно. Это инвариант, а не предположение — он проверяется
сравнением высот, и его нарушение поднимает ошибку, а не портит цифры молча.
"""

from __future__ import annotations

import os
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any

import polars as pl

from trading_research.backtest.engine import BacktestEngine, BacktestResult
from trading_research.backtest.metrics import DEFAULT_MIN_TRADES, Metrics, compute_metrics
from trading_research.backtest.portfolio import Trade
from trading_research.domain import ExecutionModel, TimeSlice
from trading_research.experiments.parallel import MP_CONTEXT
from trading_research.experiments.runner import run_backtest
from trading_research.experiments.time_splitters import WalkForwardFold
from trading_research.strategies.base import Strategy

# Числовые метрики для отбора (исключаем булев ``reliable``).
_NUMERIC_METRICS = frozenset(f.name for f in fields(Metrics)) - {"reliable"}


@dataclass(frozen=True)
class _SlicedWindow:
    """Окно, нарезанное из истории: фрейм с прогревом и позиция окна внутри него.

    ``offset`` и ``length`` избавляют от повторного поиска границ в результате
    прогона: движок выдаёт по точке equity на каждый бар ``frame``, поэтому бары
    окна лежат в кривой ровно по этим индексам.
    """

    frame: pl.DataFrame
    offset: int
    length: int


@dataclass(frozen=True)
class _FoldWindows:
    """Нарезанные окна одного фолда — считаются до отбора, один раз."""

    fold: WalkForwardFold
    train: _SlicedWindow
    test: _SlicedWindow


@dataclass(frozen=True)
class _SelectTask:
    """Единица параллельной работы: отбор на train одного фолда.

    Всё поле уходит в дочерний процесс, поэтому здесь только picklable-значения:
    срез train, окно, стратегия, комбинации, модель исполнения и движок.
    """

    train: _SlicedWindow
    train_window: TimeSlice
    strategy: Strategy
    param_combos: tuple[Mapping[str, Any], ...]
    execution: ExecutionModel
    timeframe: str
    select_by: str
    min_trades: int
    engine: BacktestEngine


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
    workers: int = 1,
) -> WalkForwardResult:
    """Прогнать walk-forward и вернуть OOS-кривую, метрики и выбор по фолдам.

    ``workers`` — процессов на отбор на train: ``1`` (по умолчанию) считает в
    текущем процессе, ``0`` берёт число ядер. Результат от этого не зависит:
    отбор по фолдам независим (см. ниже), а порядок задаёт ``map``, а не порядок
    завершения.
    """
    if select_by not in _NUMERIC_METRICS:
        raise ValueError(f"select_by {select_by!r} must be a numeric Metrics field")
    if not param_combos:
        raise ValueError("param_combos must be non-empty")

    engine = engine or BacktestEngine()
    data_times: list[datetime] = data["open_time"].to_list()

    running_balance = execution.initial_balance
    selections: list[FoldSelection] = []
    oos_segments: list[pl.DataFrame] = []
    oos_trades: list[Trade] = []

    prepared: list[_FoldWindows] = []
    for fold in folds:
        train_sub = _slice_window(data, data_times, fold.train)
        test_sub = _slice_window(data, data_times, fold.test)
        if train_sub is None or test_sub is None:
            continue
        prepared.append(_FoldWindows(fold=fold, train=train_sub, test=test_sub))

    # Отбор на train вынесен из цикла торговли целиком — в этом и возможность
    # параллелизма (RUN-8). Он от переноса баланса не зависит: отбор считается на
    # исходном ``execution``, а ``running_balance`` влияет только на test-плечо.
    # Если это когда-нибудь перестанет быть правдой, параллельный отбор станет
    # неверным — поэтому зависимость названа здесь, а не оставлена на чтение кода.
    picks = _select_all(
        prepared,
        strategy=strategy,
        param_combos=tuple(param_combos),
        execution=execution,
        timeframe=timeframe,
        select_by=select_by,
        min_trades=min_trades,
        engine=engine,
        workers=workers,
    )

    for windows, best in zip(prepared, picks, strict=True):
        fold, test_sub = windows.fold, windows.test
        if best is None:
            continue
        best_params, train_score = best

        # Торговля на test с перенесённым балансом.
        test_exec = execution.model_copy(update={"initial_balance": running_balance})
        result = run_backtest(strategy, test_sub.frame, best_params, test_exec, engine=engine)
        seg, seg_trades = _window_segment(result, test_sub, fold.test)
        if seg.is_empty():
            continue

        running_balance = float(seg["equity"][-1])
        oos_segments.append(seg)
        oos_trades.extend(seg_trades)
        test_metrics = compute_metrics(seg, seg_trades, timeframe, min_trades=min_trades)
        selections.append(
            FoldSelection(
                fold=fold.fold,
                params=dict(best_params),
                train_score=train_score,
                test_metrics=test_metrics,
            )
        )

    # Склейка срезов вместо списков Python: dtypes и значения приходят из кривых
    # прогонов как есть, без обхода через float().
    oos_curve = (
        pl.concat(oos_segments, how="vertical")
        if oos_segments
        else pl.DataFrame({"open_time": [], "equity": []})
    )
    equity_for_metrics = (
        oos_curve if oos_segments else pl.DataFrame({"equity": [execution.initial_balance]})
    )
    oos_metrics = compute_metrics(equity_for_metrics, oos_trades, timeframe, min_trades=min_trades)
    return WalkForwardResult(selections=selections, oos_equity=oos_curve, oos_metrics=oos_metrics)


def _select_all(
    prepared: Sequence[_FoldWindows],
    *,
    strategy: Strategy,
    param_combos: tuple[Mapping[str, Any], ...],
    execution: ExecutionModel,
    timeframe: str,
    select_by: str,
    min_trades: int,
    engine: BacktestEngine,
    workers: int,
) -> list[tuple[Mapping[str, Any], float] | None]:
    """Отбор на train по всем фолдам — последовательно либо в пуле процессов.

    Единица работы — фолд, а не комбинация параметров: на фолде комбинации делят
    один и тот же срез train, и отправлять этот срез в пул по разу на комбинацию
    значило бы платить за сериализацию тех же данных ``комбинации`` раз. Замеры
    RUN-7 показали, что на сериализацию уходит больше половины выигрыша, так что
    гранулярность здесь — не стилистика.

    Порядок результатов задаёт ``map``, а не порядок завершения, поэтому выбор не
    зависит ни от числа воркеров, ни от того, какой процесс закончил первым.
    """
    tasks = [
        _SelectTask(
            train=windows.train,
            train_window=windows.fold.train,
            strategy=strategy,
            param_combos=param_combos,
            execution=execution,
            timeframe=timeframe,
            select_by=select_by,
            min_trades=min_trades,
            engine=engine,
        )
        for windows in prepared
    ]
    max_workers = (os.cpu_count() or 1) if workers == 0 else workers
    if max_workers <= 1 or len(tasks) < 2:
        return [_run_select_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=MP_CONTEXT) as pool:
        return list(pool.map(_run_select_task, tasks))


def _run_select_task(task: _SelectTask) -> tuple[Mapping[str, Any], float] | None:
    """Точка входа воркера: уровень модуля, иначе задачу не запикловать."""
    best = _select_on_train(
        task.strategy,
        task.train,
        task.param_combos,
        task.execution,
        task.train_window,
        task.timeframe,
        task.select_by,
        task.min_trades,
        task.engine,
    )
    if best is None:
        return None
    params, score = best
    # Обратно через процессную границу уходит обычный dict, а не чей-то Mapping.
    return dict(params), score


def _select_on_train(
    strategy: Strategy,
    train_data: _SlicedWindow,
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
        result = run_backtest(strategy, train_data.frame, params, execution, engine=engine)
        seg, seg_trades = _window_segment(result, train_data, train_window)
        if seg.is_empty():
            continue
        metrics = compute_metrics(seg, seg_trades, timeframe, min_trades=min_trades)
        value = getattr(metrics, select_by)
        score = float("-inf") if value is None else float(value)
        if best_params is None or score > best_score:
            best_params, best_score = params, score
    if best_params is None:
        return None
    return best_params, best_score


def _slice_window(
    data: pl.DataFrame, data_times: Sequence[datetime], window: TimeSlice
) -> _SlicedWindow | None:
    """Вырезать окно ``[start, end)`` с барами прогрева перед ним.

    Границы ищет ``bisect`` по ``data_times``: список отсортирован по времени, и
    сравнение здесь ровно то же, что было в линейном скане, — но без обхода всей
    истории на каждом фолде.
    """
    first = bisect_left(data_times, window.start)
    stop = bisect_left(data_times, window.end)
    if first >= stop:
        return None
    warm_start = max(0, first - window.warmup_bars)
    return _SlicedWindow(
        frame=data.slice(warm_start, stop - warm_start),
        offset=first - warm_start,
        length=stop - first,
    )


def _window_segment(
    result: BacktestResult, sliced: _SlicedWindow, window: TimeSlice
) -> tuple[pl.DataFrame, list[Trade]]:
    """Сегмент equity/сделок прогона внутри окна ``[start, end)`` (без warmup).

    Позиция окна известна из нарезки, поэтому это срез по индексам, а не поиск.
    Опора — контракт движка: по одной точке equity на каждый бар входа, в том же
    порядке. Контракт проверяется, а не предполагается: расхождение высот означало
    бы, что срез берётся не по тем барам, то есть молча неверные OOS-цифры.
    """
    curve = result.equity_curve
    if curve.height != sliced.frame.height:
        raise RuntimeError(
            f"equity curve has {curve.height} row(s) for {sliced.frame.height} input "
            "bar(s): walk-forward relies on one equity point per bar to locate the window"
        )
    seg = curve.slice(sliced.offset, sliced.length)
    seg_trades = [tr for tr in result.trades if window.start <= tr.entry_time < window.end]
    return seg, seg_trades


__all__ = ["FoldSelection", "WalkForwardResult", "run_walk_forward"]
