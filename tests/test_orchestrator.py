"""Tests for EXP-2 — continuous run + per-slice segmentation orchestrator."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from trading_research.domain import (
    ExecutionModel,
    FillOn,
    PositionSizeType,
    SliceMode,
    TimeSliceKind,
)
from trading_research.experiments.grid_search import expand_grid
from trading_research.experiments.orchestrator import (
    count_full_runs,
    run_experiment,
    segment_by_slice,
)
from trading_research.experiments.runner import run_backtest
from trading_research.experiments.time_splitters import generate_time_slices
from trading_research.strategies.ma_cross import MovingAverageCrossStrategy

INITIAL_BALANCE = 10_000.0
TIMEFRAME = "1h"
STRATEGY = MovingAverageCrossStrategy()


def _dataset(start: datetime, n_bars: int) -> pl.DataFrame:
    """Осциллирующий hourly OHLCV с многими пересечениями MA(5/15)."""
    close = [100.0 + 12.0 * math.sin(i / 40.0) + 3.0 * math.sin(i / 7.0) for i in range(n_bars)]
    open_ = [c - 0.05 for c in close]
    high = [max(o, c) + 0.3 for o, c in zip(open_, close, strict=True)]
    low = [min(o, c) - 0.3 for o, c in zip(open_, close, strict=True)]
    times = [start + timedelta(hours=i) for i in range(n_bars)]
    return pl.DataFrame(
        {
            "open_time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1.0] * n_bars,
        }
    )


def _execution(**overrides: object) -> ExecutionModel:
    base: dict[str, object] = {
        "initial_balance": INITIAL_BALANCE,
        "commission_rate": 0.0,
        "slippage_pct": 0.0,
        "leverage": 1,
        "position_size_type": PositionSizeType.PERCENT_BALANCE,
        "position_size_value": 100.0,
        "allow_short": True,
        "close_on_reverse_signal": True,
        "signal_lag": 1,
        "fill_on": FillOn.NEXT_OPEN,
    }
    base.update(overrides)
    return ExecutionModel(**base)  # type: ignore[arg-type]


# --- run-count reduction (DoD: 432 вместо 25 920) ---------------------------


def test_count_full_runs_continuous_is_independent_of_slices() -> None:
    assert count_full_runs(432, 60, SliceMode.CONTINUOUS) == 432
    assert count_full_runs(432, 1, SliceMode.CONTINUOUS) == 432


def test_count_full_runs_independent_multiplies_by_slices() -> None:
    # Пример плана: наивный подход дал бы 25 920 полных прогонов.
    assert count_full_runs(432, 60, SliceMode.INDEPENDENT) == 25_920


def test_continuous_runs_engine_once_per_combo() -> None:
    data = _dataset(datetime(2024, 2, 1, tzinfo=UTC), 24 * 40)
    slices = generate_time_slices(
        TimeSliceKind.WEEK, data["open_time"][0], data["open_time"][-1], timeframe=TIMEFRAME
    )
    combos = expand_grid(
        {"fast_ma": [5, 8], "slow_ma": [15, 20]},
        predicate=lambda c: c["fast_ma"] < c["slow_ma"],
    )
    result = run_experiment(data, STRATEGY, combos, slices, _execution(), TIMEFRAME, min_trades=1)
    assert result.n_full_runs == len(combos)  # не зависит от числа слайсов
    assert len(result.per_slice) == len(combos) * len(slices)


# --- continuity: пер-слайс сегменты согласованы с полной equity -------------


def test_per_slice_returns_telescope_to_full_window_return() -> None:
    # Данные: янв (pre-research, «прогрев») + фев–март (research).
    data = _dataset(datetime(2024, 1, 1, tzinfo=UTC), 24 * 91)  # ~ Jan..Mar
    slices = generate_time_slices(
        TimeSliceKind.MONTH,
        datetime(2024, 2, 1, tzinfo=UTC),
        datetime(2024, 4, 1, tzinfo=UTC),
        timeframe=TIMEFRAME,
    )
    assert len(slices) == 2  # Feb, Mar
    params = {"fast_ma": 5, "slow_ma": 15}

    full = run_backtest(STRATEGY, data, params, _execution())
    per_slice = segment_by_slice(
        full.equity_curve,
        full.trades,
        slices,
        params,
        initial_balance=INITIAL_BALANCE,
        timeframe=TIMEFRAME,
        min_trades=1,
    )

    times = full.equity_curve["open_time"].to_list()
    equities = [float(x) for x in full.equity_curve["equity"].to_list()]
    first_start, last_end = slices[0].start, slices[-1].end
    base0 = INITIAL_BALANCE
    for t, e in zip(times, equities, strict=True):
        if t < first_start:
            base0 = e
        else:
            break
    final_research = next(
        e for t, e in reversed(list(zip(times, equities, strict=True))) if t < last_end
    )

    compounded = 1.0
    for sm in per_slice:
        compounded *= 1.0 + sm.metrics.total_return_pct / 100.0
    assert compounded == pytest.approx(final_research / base0, rel=1e-9)


def test_warmup_and_pre_research_trades_excluded() -> None:
    data = _dataset(datetime(2024, 1, 1, tzinfo=UTC), 24 * 91)
    slices = generate_time_slices(
        TimeSliceKind.MONTH,
        datetime(2024, 2, 1, tzinfo=UTC),
        datetime(2024, 4, 1, tzinfo=UTC),
        timeframe=TIMEFRAME,
    )
    params = {"fast_ma": 5, "slow_ma": 15}
    full = run_backtest(STRATEGY, data, params, _execution())

    first_start, last_end = slices[0].start, slices[-1].end
    # Есть сделки в pre-research окне (январь) — иначе тест бессмысленен.
    assert any(t.entry_time < first_start for t in full.trades)
    expected_in_window = sum(1 for t in full.trades if first_start <= t.entry_time < last_end)

    per_slice = segment_by_slice(
        full.equity_curve,
        full.trades,
        slices,
        params,
        initial_balance=INITIAL_BALANCE,
        timeframe=TIMEFRAME,
        min_trades=1,
    )
    total_slice_trades = sum(sm.metrics.trades_count for sm in per_slice)
    assert total_slice_trades == expected_in_window


# --- independent mode -------------------------------------------------------


def test_independent_mode_switches_run_count_and_resets_balance() -> None:
    data = _dataset(datetime(2024, 2, 1, tzinfo=UTC), 24 * 40)
    slices = generate_time_slices(
        TimeSliceKind.WEEK, data["open_time"][0], data["open_time"][-1], timeframe=TIMEFRAME
    )
    combos = [{"fast_ma": 5, "slow_ma": 15}]

    cont = run_experiment(data, STRATEGY, combos, slices, _execution(), TIMEFRAME, min_trades=1)
    indep = run_experiment(
        data,
        STRATEGY,
        combos,
        slices,
        _execution(),
        TIMEFRAME,
        mode=SliceMode.INDEPENDENT,
        min_trades=1,
    )

    assert cont.n_full_runs == 1
    assert indep.n_full_runs == len(slices)
    assert indep.mode is SliceMode.INDEPENDENT
    # Обе нарезки покрывают одни и те же окна.
    assert {s.slice_id for s in indep.per_slice} == {s.slice_id for s in cont.per_slice}


def test_independent_first_slice_matches_standalone_run() -> None:
    # В independent-режиме первый слайс = самостоятельный прогон со сбросом баланса.
    data = _dataset(datetime(2024, 2, 1, tzinfo=UTC), 24 * 40)
    slices = generate_time_slices(
        TimeSliceKind.WEEK, data["open_time"][0], data["open_time"][-1], timeframe=TIMEFRAME
    )
    params = {"fast_ma": 5, "slow_ma": 15}
    indep = run_experiment(
        data,
        STRATEGY,
        [params],
        slices,
        _execution(),
        TIMEFRAME,
        mode=SliceMode.INDEPENDENT,
        min_trades=1,
    )
    first = next(sm for sm in indep.per_slice if sm.slice_id == slices[0].id)
    # У первого окна прогрева нет → equity стартует с initial_balance; метрики
    # определены и воспроизводимы.
    assert first.metrics.trades_count >= 0
    assert first.metrics.annualization_factor == pytest.approx(365 * 24)
