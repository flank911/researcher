"""Tests for RUN-8 — ускорение walk-forward не имеет права менять цифры.

Оптимизация тут опаснее обычного кода: она трогает нарезку окон, то есть ровно то,
что определяет, какие бары попали в честный out-of-sample. Ошибка здесь не падает,
а тихо меняет результат исследования. Поэтому тесты проверяют не «стало быстрее», а
эквивалентность:

- границы окна совпадают с линейным сканом, который был до `bisect`;
- OOS-кривая, метрики и выбор по фолдам совпадают с эталоном, снятым с реализации
  **до** оптимизации (значения ниже получены прогоном кода из `main`);
- параллельный отбор даёт тот же результат при любом числе воркеров;
- инвариант «по точке equity на бар», на который опирается срез по индексам,
  проверяется, а не предполагается.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from trading_research.backtest.engine import BacktestResult
from trading_research.data.schema import CANDLE_SCHEMA
from trading_research.domain import ExecutionModel, TimeSlice, TimeSliceRole
from trading_research.experiments.time_splitters import generate_walk_forward_folds
from trading_research.experiments.walk_forward import (
    _slice_window,
    _window_segment,
    run_walk_forward,
)
from trading_research.strategies import get_strategy

T0 = datetime(2024, 1, 1, tzinfo=UTC)
BARS = 24 * 220


def _candles(n: int = BARS) -> pl.DataFrame:
    times = [T0 + timedelta(hours=i) for i in range(n)]
    # Пила с трендом: нужны сделки и у быстрой, и у медленной комбинации. Числа
    # заведомо нереалистичные — фикстура закрепляет эквивалентность, не доходность.
    close = [100.0 + 0.01 * i + 8.0 * math.sin(i / 17.0) for i in range(n)]
    return pl.DataFrame(
        {
            "open_time": times,
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [10.0 + (i % 7) for i in range(n)],
            "close_time": [t + timedelta(hours=1) for t in times],
            "quote_volume": [1000.0] * n,
            "trades": [5] * n,
            "taker_buy_base": [4.0] * n,
            "taker_buy_quote": [400.0] * n,
        },
        schema=CANDLE_SCHEMA,
    )


def _execution() -> ExecutionModel:
    return ExecutionModel(
        initial_balance=10_000,
        commission_rate=0.0006,
        slippage_pct=0.0003,
        leverage=3,
        position_size_type="percent_balance",
        position_size_value=20,
        signal_lag=1,
        fill_on="next_open",
    )


def _window(offset_h: int, span_h: int, *, warmup: int) -> TimeSlice:
    return TimeSlice(
        id=f"w{offset_h}-{span_h}",
        label=f"+{offset_h}h",
        start=T0 + timedelta(hours=offset_h),
        end=T0 + timedelta(hours=offset_h + span_h),
        role=TimeSliceRole.TRAIN,
        warmup_bars=warmup,
    )


def _folds() -> list:
    return generate_walk_forward_folds(
        T0,
        T0 + timedelta(hours=BARS),
        train=timedelta(days=30),
        test=timedelta(days=10),
        step=timedelta(days=10),
        warmup_bars=50,
        anchored=False,
    )


_COMBOS = [{"fast_ma": f, "slow_ma": s} for f in (5, 10, 20) for s in (50, 100)]


def _run(**kwargs):
    return run_walk_forward(
        _candles(),
        get_strategy("ma_cross"),
        _COMBOS,
        _folds(),
        _execution(),
        "1h",
        select_by="sharpe",
        **kwargs,
    )


# --- границы окна: bisect против линейного скана ----------------------------


def _scan_bounds(times: list[datetime], window: TimeSlice) -> tuple[int, int] | None:
    """Прежняя реализация поиска границ — эталон для `bisect`."""
    in_window = [i for i, t in enumerate(times) if window.start <= t < window.end]
    if not in_window:
        return None
    return in_window[0], in_window[-1]


@pytest.mark.parametrize("warmup", [0, 1, 50, 10_000])
def test_window_bounds_match_the_linear_scan(warmup: int) -> None:
    data = _candles(500)
    times = data["open_time"].to_list()

    for offset_h, span_h in ((0, 24), (7, 1), (100, 73), (499, 1), (480, 100)):
        window = _window(offset_h, span_h, warmup=warmup)
        sliced = _slice_window(data, times, window)
        expected = _scan_bounds(times, window)

        assert expected is not None
        first, last = expected
        warm_start = max(0, first - warmup)
        assert sliced is not None
        assert sliced.frame.height == last - warm_start + 1
        assert sliced.offset == first - warm_start
        assert sliced.length == last - first + 1
        # Срез указывает на те же бары, что отобрал бы скан.
        window_bars = sliced.frame.slice(sliced.offset, sliced.length)
        assert window_bars["open_time"].to_list() == times[first : last + 1]


def test_window_outside_data_is_none_as_before() -> None:
    data = _candles(100)
    times = data["open_time"].to_list()
    window = _window(24 * 400, 24, warmup=0)

    assert _slice_window(data, times, window) is None
    assert _scan_bounds(times, window) is None


# --- инвариант «точка equity на бар» ---------------------------------------


def test_misaligned_equity_curve_raises_instead_of_silently_shifting() -> None:
    """Срез по индексам верен только при совпадении высот — молчать тут нельзя."""
    data = _candles(200)
    times = data["open_time"].to_list()
    window = _window(10, 50, warmup=5)
    sliced = _slice_window(data, times, window)
    assert sliced is not None

    truncated = BacktestResult(
        equity_curve=pl.DataFrame(
            {"open_time": times[:10], "equity": [10_000.0] * 10},
        ),
        trades=[],
    )

    with pytest.raises(RuntimeError, match="one equity point per bar"):
        _window_segment(truncated, sliced, window)


# --- эквивалентность результата --------------------------------------------

# Эталон снят прогоном реализации до оптимизации (коммит main перед RUN-8) на
# фикстуре выше. Любое расхождение означает, что ускорение изменило цифры.
#
# Побитовая сверка здесь законна только потому, что метрики складывают через
# ``math.fsum`` (INF-8): точно округлённая сумма одинакова на всех поддерживаемых
# интерпретаторах. Со встроенным ``sum`` этот assert проверял версию питона —
# на 3.11 сложение идёт слева направо, на 3.12+ компенсированно, и ``sharpe``
# расходился в 13-м знаке при той же кривой капитала. Так тест и был красным на
# 3.11 с момента мержа RUN-8.
_EXPECTED_FOLDS = 19
_EXPECTED_EQUITY_ROWS = 4560
_EXPECTED_RETURN_PCT = 1290.5557116086459
_EXPECTED_SHARPE = 40.52356944974927
_EXPECTED_MAX_DD_PCT = 2.5913200917675665
_EXPECTED_TRADES = 85
_EXPECTED_FIRST_TRAIN_SCORE = 39.500409246484296


def test_oos_result_matches_pre_optimization_reference() -> None:
    result = _run()

    assert len(result.selections) == _EXPECTED_FOLDS
    assert result.oos_equity.height == _EXPECTED_EQUITY_ROWS
    assert result.oos_equity.columns == ["open_time", "equity"]
    # Побитово, а не approx: ускорение не имеет права менять последний знак.
    # Ослабить допуском было бы проще, но это сняло бы с теста ровно то, ради чего
    # он написан, — чувствительность к составу окна в последних битах.
    assert result.oos_metrics.total_return_pct == _EXPECTED_RETURN_PCT
    assert result.oos_metrics.sharpe == _EXPECTED_SHARPE
    assert result.oos_metrics.max_drawdown_pct == _EXPECTED_MAX_DD_PCT
    assert result.oos_metrics.trades_count == _EXPECTED_TRADES
    assert result.selections[0].train_score == _EXPECTED_FIRST_TRAIN_SCORE


def test_oos_curve_dtypes_survive_the_polars_concat() -> None:
    """Кривая теперь склеивается из срезов, а не собирается из Python-списков."""
    curve = _run().oos_equity

    assert curve.schema["equity"] == pl.Float64
    assert isinstance(curve.schema["open_time"], pl.Datetime)
    assert curve.schema["open_time"].time_zone == "UTC"
    # Время строго возрастает: склейка не перепутала порядок фолдов.
    times = curve["open_time"].to_list()
    assert all(a < b for a, b in zip(times, times[1:], strict=False))


# --- параллельный отбор ----------------------------------------------------


@pytest.mark.parametrize("workers", [2, 3])
def test_parallel_selection_matches_sequential(workers: int) -> None:
    """DoD RUN-8: результат не зависит от числа воркеров."""
    sequential = _run(workers=1)
    parallel = _run(workers=workers)

    assert [s.params for s in parallel.selections] == [s.params for s in sequential.selections]
    assert [s.train_score for s in parallel.selections] == [
        s.train_score for s in sequential.selections
    ]
    assert parallel.oos_metrics == sequential.oos_metrics
    assert parallel.oos_equity.equals(sequential.oos_equity)


def test_zero_workers_means_cpu_count_and_changes_nothing() -> None:
    assert _run(workers=0).oos_metrics == _run(workers=1).oos_metrics
