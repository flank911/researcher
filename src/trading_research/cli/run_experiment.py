"""EXP-6 + RUN-1 — бизнес-логика команды ``run-experiment``.

Склейка слоёв в один прогон по YAML-конфигу: данные из Parquet-store (DATA-5) →
стратегия (BT-1) → сетка параметров (EXP-1) → временные окна (DATA-6) →
непрерывный прогон с нарезкой по слайсам (EXP-2). Результат — каноническая таблица
``parameter_performance_by_slice`` (AN-1) в Parquet, готовая для аналитики
(``analyze-results``, AN-2).

RUN-1 подключил сюда три модуля, которые до этого существовали только как
библиотеки:

- **EXP-3** — каждый ран ``(параметры × слайс)`` получает ``run_hash``; готовые
  раны пропускаются (dedupe), прерванный прогон доделывается (resume).
- **EXP-4** — артефакты и метрики пишутся в ``ResultsStore``, а таблица AN-1
  строится уже из каталога (``performance_from_store``). Флаги конфига
  ``output.save_trades`` / ``save_equity_curve`` наконец на что-то влияют.
- **EXP-5** — сетка исполняется в пуле процессов.

**Единица параллельной работы — комбинация параметров, а не отдельный ран.**
В дефолтном режиме ``CONTINUOUS`` на комбинацию делается один прогон по всей
истории, а метрики окна берут базой equity на баре перед его началом, поэтому
слайсы одной комбинации не независимы и считаются вместе
(``execute_run_groups_parallel``). В ``INDEPENDENT`` окна независимы, но группа всё
равно остаётся комбинацией — так меньше задач и меньше накладных расходов на пул.

``dataset_hash`` считается контентно по фактически загруженным свечам (DATA-4), а
``git_commit`` берётся из INF-3, поэтому смена данных или кода инвалидирует кэш.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl

from trading_research.analytics.queries import performance_from_store, save_performance_table
from trading_research.backtest.metrics import DEFAULT_MIN_TRADES
from trading_research.backtest.portfolio import Trade
from trading_research.config import load_experiment_config
from trading_research.data.catalog import dataset_hash
from trading_research.data.storage.duckdb_store import DuckDBStore
from trading_research.domain import ExecutionModel, SliceMode, TimeSlice
from trading_research.experiments.cache import RunLedger, RunSpec, build_run_specs
from trading_research.experiments.grid_search import expand_grid
from trading_research.experiments.orchestrator import SliceMetrics, segment_by_slice
from trading_research.experiments.parallel import execute_run_groups_parallel
from trading_research.experiments.runner import run_backtest
from trading_research.hashing import git_commit_hash
from trading_research.strategies import get_strategy

RESULTS_FILE = "performance.parquet"
LEDGER_DIR = "ledger"
_TIME_COL = "open_time"


@dataclass(frozen=True)
class WorkerContext:
    """Picklable-контекст прогона: всё, что воркеру нужно помимо самой группы.

    Стратегия передаётся **именем** (резолвится в воркере), данные читаются из
    Parquet-store на месте — так в дочерний процесс не уезжает ни объект стратегии,
    ни DataFrame истории.
    """

    data_root: Path
    store_root: Path
    strategy_name: str
    exchange: str
    market: str
    timeframe: str
    start: datetime
    end: datetime
    slices: tuple[TimeSlice, ...]
    mode: SliceMode
    min_trades: int
    save_trades: bool
    save_equity: bool


def run_experiment_from_config(
    config_path: str | Path,
    *,
    data_root: str | Path,
    results_root: str | Path,
    workers: int | None = None,
    resume: bool = True,
) -> Path:
    """Прогнать эксперимент по конфигу и записать таблицу AN-1 в Parquet.

    Возвращает путь к ``{results_root}/{name}/performance.parquet``.

    ``workers=None`` → число процессов по умолчанию (``os.cpu_count()``);
    ``workers=1`` — без пула, в текущем процессе. ``resume=False`` сбрасывает
    маркеры завершённых ранов, то есть считает всё заново.
    """
    from trading_research.experiments.results_store import ResultsStore

    cfg = load_experiment_config(config_path)
    combos = expand_grid(cfg.params_grid)
    slices = generate_slices_for(cfg)
    if not slices:
        raise ValueError(
            "config produced no time slices; check data range and time_slices.min_bars"
        )

    store_root = Path(results_root) / cfg.name
    store = ResultsStore(store_root)
    ledger = RunLedger(store_root / LEDGER_DIR)
    if not resume:
        _reset_ledger(ledger)

    commit = git_commit_hash()
    data_store = DuckDBStore(data_root)

    for symbol in cfg.data.symbols:
        candles = data_store.read_window(
            exchange=cfg.data.exchange,
            market=cfg.data.market,
            symbol=symbol,
            timeframe=cfg.data.timeframe,
            start=cfg.data.start,
            end=cfg.data.end,
        )
        specs = build_run_specs(
            strategy=cfg.strategy.name,
            symbol=symbol,
            timeframe=cfg.data.timeframe,
            dataset_hash=dataset_hash(candles),
            execution=cfg.execution,
            param_combos=combos,
            slices=slices,
            git_commit=commit,
        )
        ctx = WorkerContext(
            data_root=Path(data_root),
            store_root=store_root,
            strategy_name=cfg.strategy.name,
            exchange=cfg.data.exchange,
            market=cfg.data.market,
            timeframe=cfg.data.timeframe,
            start=cfg.data.start,
            end=cfg.data.end,
            slices=tuple(slices),
            mode=cfg.slice_mode,
            min_trades=DEFAULT_MIN_TRADES,
            save_trades=cfg.output.save_trades,
            save_equity=cfg.output.save_equity_curve,
        )
        execute_run_groups_parallel(
            group_specs_by_params(specs),
            ledger,
            partial(compute_group, ctx=ctx),
            max_workers=workers,
        )

    table = performance_from_store(store, slices, experiment_id=cfg.name)
    return save_performance_table(table, store_root / RESULTS_FILE)


def generate_slices_for(cfg: Any) -> list[TimeSlice]:
    """Временные окна конфига (вынесено, чтобы CLI и воркер считали их одинаково)."""
    from trading_research.experiments.time_splitters import generate_time_slices

    return generate_time_slices(
        cfg.time_slices.type,
        cfg.data.start,
        cfg.data.end,
        timeframe=cfg.data.timeframe,
        min_bars=cfg.time_slices.min_bars,
    )


def group_specs_by_params(specs: list[RunSpec]) -> list[list[RunSpec]]:
    """Сгруппировать раны по комбинации параметров, сохраняя порядок сетки.

    Ключ — ``run_hash`` без ``slice_id`` не считаем: достаточно канонического
    представления параметров, поскольку остальные поля в рамках одного символа
    совпадают.
    """
    groups: dict[str, list[RunSpec]] = {}
    for spec in specs:
        key = _params_key(spec)
        groups.setdefault(key, []).append(spec)
    return list(groups.values())


def compute_group(specs: list[RunSpec], *, ctx: WorkerContext) -> None:
    """Посчитать и сохранить все раны одной комбинации параметров.

    Выполняется в дочернем процессе: читает данные, прогоняет бэктест (один на всю
    историю в ``CONTINUOUS``), нарезает результат по окнам и сохраняет каждый ран в
    ``ResultsStore``.
    """
    from trading_research.experiments.results_store import ResultsStore

    if not specs:
        return

    params = specs[0].params
    execution = specs[0].execution
    strategy = get_strategy(ctx.strategy_name)
    store = ResultsStore(ctx.store_root)

    data = DuckDBStore(ctx.data_root).read_window(
        exchange=ctx.exchange,
        market=ctx.market,
        symbol=specs[0].symbol,
        timeframe=ctx.timeframe,
        start=ctx.start,
        end=ctx.end,
    )

    by_slice = {sl.id: sl for sl in ctx.slices}
    wanted = [by_slice[s.slice_id] for s in specs if s.slice_id in by_slice]

    if ctx.mode is SliceMode.CONTINUOUS:
        result = run_backtest(strategy, data, params, execution)
        metrics_by_slice = {
            m.slice_id: m
            for m in segment_by_slice(
                result.equity_curve,
                result.trades,
                wanted,
                params,
                initial_balance=execution.initial_balance,
                timeframe=ctx.timeframe,
                min_trades=ctx.min_trades,
            )
        }
        segments = {
            sl.id: _segment_artifacts(result.equity_curve, result.trades, sl) for sl in wanted
        }
    else:
        metrics_by_slice, segments = _independent_segments(
            strategy, data, params, execution, wanted, ctx
        )

    empty_equity = _empty_equity()
    for spec in specs:
        sm = metrics_by_slice.get(spec.slice_id)
        if sm is None:
            continue
        equity, trades = segments.get(spec.slice_id, (empty_equity, []))
        # Выключенный флаг обнуляет артефакт, но сохраняет схему (``clear()``) —
        # пустой файл остаётся читаемым, как и ``trades_to_df([])``.
        store.save_run(
            spec,
            sm.metrics,
            trades if ctx.save_trades else [],
            equity if ctx.save_equity else equity.clear(),
        )


def _empty_equity() -> pl.DataFrame:
    """Пустая equity-кривая с канонической схемой (для окон без баров)."""
    return pl.DataFrame(schema={_TIME_COL: pl.Datetime("ms", "UTC"), "equity": pl.Float64()})


def _params_key(spec: RunSpec) -> str:
    from trading_research.hashing import canonical_json

    return canonical_json(spec.params)


def _reset_ledger(ledger: RunLedger) -> None:
    """Удалить маркеры завершённых ранов (``--no-resume``)."""
    for marker in ledger.root.glob("*.json"):
        marker.unlink()


def _segment_artifacts(
    equity_curve: pl.DataFrame, trades: list[Trade], sl: TimeSlice
) -> tuple[pl.DataFrame, list[Trade]]:
    """Вырезать equity-бары и сделки окна ``[start, end)`` для артефактов рана."""
    equity = equity_curve.filter((pl.col(_TIME_COL) >= sl.start) & (pl.col(_TIME_COL) < sl.end))
    seg_trades = [t for t in trades if sl.start <= t.entry_time < sl.end]
    return equity, seg_trades


def _independent_segments(
    strategy: Any,
    data: pl.DataFrame,
    params: dict[str, Any],
    execution: ExecutionModel,
    wanted: list[TimeSlice],
    ctx: WorkerContext,
) -> tuple[dict[str, SliceMetrics], dict[str, tuple[pl.DataFrame, list[Trade]]]]:
    """Режим ``INDEPENDENT``: отдельный прогон на окно со сбросом баланса."""
    from trading_research.backtest.metrics import compute_metrics

    metrics_by_slice: dict[str, SliceMetrics] = {}
    segments: dict[str, tuple[pl.DataFrame, list[Trade]]] = {}
    times: list[datetime] = data[_TIME_COL].to_list()

    for sl in wanted:
        in_window = [i for i, t in enumerate(times) if sl.start <= t < sl.end]
        if not in_window:
            continue
        first, last = in_window[0], in_window[-1]
        warm_start = max(0, first - sl.warmup_bars)
        sub = data.slice(warm_start, last - warm_start + 1)

        result = run_backtest(strategy, sub, params, execution)
        equity, seg_trades = _segment_artifacts(result.equity_curve, result.trades, sl)
        seg_equity = [execution.initial_balance]
        seg_equity.extend(float(x) for x in equity["equity"].to_list())
        metrics = compute_metrics(
            pl.DataFrame({"equity": seg_equity}),
            seg_trades,
            ctx.timeframe,
            min_trades=ctx.min_trades,
        )
        metrics_by_slice[sl.id] = SliceMetrics(slice_id=sl.id, params=dict(params), metrics=metrics)
        segments[sl.id] = (equity, seg_trades)

    return metrics_by_slice, segments


__all__ = [
    "LEDGER_DIR",
    "RESULTS_FILE",
    "WorkerContext",
    "compute_group",
    "group_specs_by_params",
    "run_experiment_from_config",
]
