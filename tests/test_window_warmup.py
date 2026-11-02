"""Tests for WF-4 — прогрев окон берётся из стратегии и не торгуется.

Что здесь закрепляется:

- окно, вырезанное из середины истории, видит те же сигналы, что непрерывный
  прогон на этом участке (до WF-4 не видело: окно стартовало с холодного
  индикатора, и на 4h тестовое окно при ``slow_ma=200`` не прогревалось никогда);
- прогрев просит стратегия, а число из конфига работает **нижней** границей;
- бары прогрева греют индикаторы и не торгуются;
- резерв holdout греется хвостом research-части, и этот прогрев не попадает ни в
  сделки, ни в метрики.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
import pytest

from trading_research.backtest.engine import BacktestEngine
from trading_research.backtest.metrics import Metrics
from trading_research.backtest.portfolio import Trade
from trading_research.domain import ExecutionModel, TimeSlice, TimeSliceRole
from trading_research.experiments.holdout import evaluate_holdout
from trading_research.experiments.runner import run_backtest
from trading_research.experiments.time_splitters import generate_walk_forward_folds
from trading_research.experiments.walk_forward import (
    _narrow,
    _select_on_train,
    _slice_window,
    run_walk_forward,
)
from trading_research.strategies import get_strategy
from trading_research.strategies.base import LONG_SIGNAL, SHORT_SIGNAL

T0 = datetime(2024, 1, 1, tzinfo=UTC)
HOUR = timedelta(hours=1)
SLOW = 200
PARAMS = {"fast_ma": 20, "slow_ma": SLOW}


def _candles(n: int) -> pl.DataFrame:
    """Пила с трендом: кроссы случаются и на коротком участке."""
    close = [100.0 + 0.01 * i + 8.0 * math.sin(i / 17.0) for i in range(n)]
    times = [T0 + i * HOUR for i in range(n)]
    return pl.DataFrame(
        {
            "open_time": times,
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [10.0] * n,
        }
    )


def _execution(**kw: object) -> ExecutionModel:
    base: dict[str, object] = {
        "initial_balance": 10_000.0,
        "commission_rate": 0.0006,
        "slippage_pct": 0.0003,
        "leverage": 3,
        "position_size_type": "percent_balance",
        "position_size_value": 20,
        "signal_lag": 1,
        "fill_on": "next_open",
    }
    base.update(kw)
    return ExecutionModel(**base)  # type: ignore[arg-type]


def _entries(trades: list[Trade], start: datetime, end: datetime) -> list[tuple[datetime, str]]:
    return [(t.entry_time, t.side.value) for t in trades if start <= t.entry_time < end]


# --- окно против непрерывного прогона ---------------------------------------


def test_window_from_the_middle_sees_the_same_signals_as_full_history() -> None:
    """Главный инвариант WF-4: нарезка окна не меняет того, что видит стратегия."""
    data = _candles(1200)
    strategy = get_strategy("ma_cross")
    warmup = strategy.warmup_bars(PARAMS)
    assert warmup == SLOW  # иначе фикстура ниже ничего не проверяет

    w_start, w_len = 800, 200
    window_from, window_to = T0 + w_start * HOUR, T0 + (w_start + w_len) * HOUR

    full = run_backtest(strategy, data, PARAMS, _execution())
    reference = _entries(full.trades, window_from, window_to)
    assert reference, "фикстура обязана давать сделки внутри окна"

    warm = run_backtest(
        strategy,
        data.slice(w_start - warmup, warmup + w_len),
        PARAMS,
        _execution(),
        warmup_bars=warmup,
    )
    assert _entries(warm.trades, window_from, window_to) == reference

    # Тест различающий: без прогрева окно молчит — это и был дефект.
    cold = run_backtest(strategy, data.slice(w_start, w_len), PARAMS, _execution())
    assert _entries(cold.trades, window_from, window_to) == []


def test_warmup_bars_are_warmed_but_not_traded() -> None:
    data = _candles(600)
    strategy = get_strategy("ma_cross")
    result = run_backtest(strategy, data, PARAMS, _execution(), warmup_bars=400)

    # Длина кривой не меняется — на этом держится нарезка окон по индексам.
    assert result.equity_curve.height == data.height
    equity = result.equity_curve["equity"].to_list()
    assert set(equity[:400]) == {_execution().initial_balance}
    assert all(t.entry_time >= T0 + 400 * HOUR for t in result.trades)


def test_signal_on_the_last_warmup_bar_is_not_traded() -> None:
    """Соглашение о границе: окно начинается плоским, отложенных сигналов нет.

    При ``signal_lag=1`` сигнал на последнем баре прогрева исполнился бы на первом
    баре окна. Он погашен: окно торгует только то, что произошло в нём самом.
    Соглашение стоит зафиксировать, потому что оно съедает один бар на окно — это
    решение, а не случайность.
    """
    data = _candles(10)
    longs = [i == 4 for i in range(10)]
    frame = data.with_columns(pl.Series(LONG_SIGNAL, longs), pl.Series(SHORT_SIGNAL, [False] * 10))

    class _Fixed:
        name = "fixed"

        def warmup_bars(self, params: Mapping[str, Any]) -> int:
            return 0

        def generate_signals(self, data: pl.DataFrame, params: Mapping[str, Any]) -> pl.DataFrame:
            return data

    traded = run_backtest(_Fixed(), frame, {}, _execution(), warmup_bars=4)
    muted = run_backtest(_Fixed(), frame, {}, _execution(), warmup_bars=5)
    assert len(traded.trades) == 1
    assert traded.trades[0].entry_time == T0 + 5 * HOUR
    assert muted.trades == []


# --- прогрев в walk-forward -------------------------------------------------


class _SpyStrategy:
    """MA Cross, который помнит, что у него спрашивали и с какими фреймами."""

    name = "spy"

    def __init__(self, warmup: int) -> None:
        self._warmup = warmup
        self.asked: list[dict[str, Any]] = []
        self.frame_heights: list[int] = []

    def warmup_bars(self, params: Mapping[str, Any]) -> int:
        self.asked.append(dict(params))
        return self._warmup

    def generate_signals(self, data: pl.DataFrame, params: Mapping[str, Any]) -> pl.DataFrame:
        self.frame_heights.append(data.height)
        return data.with_columns(
            pl.Series(LONG_SIGNAL, [False] * data.height),
            pl.Series(SHORT_SIGNAL, [False] * data.height),
        )


def _folds(warmup_bars: int) -> list:
    return generate_walk_forward_folds(
        T0,
        T0 + 1200 * HOUR,
        train=timedelta(days=20),
        test=timedelta(days=5),
        step=timedelta(days=5),
        warmup_bars=warmup_bars,
        anchored=False,
    )


def _run_with(strategy: _SpyStrategy, config_warmup: int) -> None:
    run_walk_forward(
        _candles(1200),
        strategy,
        [{"p": 1}],
        _folds(config_warmup),
        _execution(),
        "1h",
        select_by="total_return_pct",
    )


def test_walk_forward_asks_the_strategy_for_warmup() -> None:
    """Контракт ``warmup_bars`` должен иметь вызывающего, а не только тест на себя.

    До WF-4 ``Strategy.warmup_bars`` не вызывался нигде в ``src`` — прогрев брался
    из конфига. Этот тест ловит возврат к такому положению.
    """
    spy = _SpyStrategy(warmup=120)
    _run_with(spy, config_warmup=0)
    assert spy.asked, "walk-forward не спросил стратегию о прогреве"


@pytest.mark.parametrize(
    ("strategy_warmup", "config_warmup", "expected"),
    [
        (120, 0, 120),  # конфиг молчит — прогрев даёт стратегия
        (120, 300, 300),  # ручное число больше — оно и работает (запас)
        (300, 120, 300),  # ручное число меньше — отнять прогрев им нельзя
    ],
)
def test_config_warmup_is_a_floor(strategy_warmup: int, config_warmup: int, expected: int) -> None:
    spy = _SpyStrategy(warmup=strategy_warmup)
    _run_with(spy, config_warmup)

    # Окна train (20 дней = 480 бар) и test (5 дней = 120 бар) плюс прогрев.
    # Берём фолды из середины: у первых прогрев урезан началом истории.
    heights = set(spy.frame_heights)
    assert 480 + expected in heights
    assert 120 + expected in heights


# --- прогрев резерва holdout ------------------------------------------------


def test_holdout_is_primed_with_the_bars_before_the_reserve() -> None:
    data = _candles(1200)
    reserve_from = 1000
    strategy = get_strategy("ma_cross")
    reserve = data.slice(reserve_from)
    warmup = data.slice(reserve_from - SLOW, SLOW)

    cold = evaluate_holdout(reserve, strategy, PARAMS, _execution(), "1h", min_trades=1)
    primed = evaluate_holdout(
        reserve, strategy, PARAMS, _execution(), "1h", min_trades=1, warmup_data=warmup
    )

    # Резерв в 200 бар при slow_ma=200 без прогрева не даёт ни одной сделки.
    assert cold.trades_count == 0
    assert primed.trades_count > 0


def test_primed_reserve_sees_what_continuous_trading_would_see() -> None:
    """Прогретый резерв совпадает со взглядом непрерывного прогона на тот же участок.

    Это тот же инвариант, что и для окон фолда, только для резерва: прогрев
    восстанавливает состояние индикатора, а не добавляет сделок от себя.
    """
    data = _candles(1200)
    reserve_from = 900
    reserve_start = T0 + reserve_from * HOUR
    strategy = get_strategy("ma_cross")

    full = run_backtest(strategy, data, PARAMS, _execution())
    expected = len(_entries(full.trades, reserve_start, T0 + 1200 * HOUR))

    metrics = evaluate_holdout(
        data.slice(reserve_from),
        strategy,
        PARAMS,
        _execution(),
        "1h",
        min_trades=1,
        warmup_data=data.slice(reserve_from - SLOW, SLOW),
    )
    assert metrics.trades_count == expected


def test_holdout_warmup_must_be_contiguous_with_the_reserve() -> None:
    data = _candles(1200)
    reserve = data.slice(900)
    with pytest.raises(ValueError, match="must end before the reserve starts"):
        evaluate_holdout(
            reserve,
            get_strategy("ma_cross"),
            PARAMS,
            _execution(),
            "1h",
            warmup_data=data.slice(950, SLOW),
        )


def test_window_warmup_is_clamped_at_the_start_of_history() -> None:
    """У начала истории прогрева взять негде — берём сколько есть, без ошибки."""
    data = _candles(300)
    times = data["open_time"].to_list()
    window = TimeSlice(
        id="w",
        label="w",
        start=T0 + 50 * HOUR,
        end=T0 + 150 * HOUR,
        role=TimeSliceRole.TEST,
        warmup_bars=0,
    )
    sliced = _slice_window(data, times, window, SLOW)
    assert sliced is not None
    assert sliced.offset == 50  # запрошено 200, доступно 50
    assert sliced.length == 100

    # Укоротить прогрев можно, удлинить сверх доступного — нет.
    assert _narrow(sliced, 10).offset == 10
    assert _narrow(sliced, SLOW).offset == 50


# --- результат комбинации не зависит от состава сетки ------------------------


class _EwmStrategy:
    """Вход по пересечению цены и EWM-среднего: значение зависит от всей предыстории.

    Нужна именно такая стратегия: у скользящего среднего лишние бары прогрева
    ничего не меняют, а у экспоненциального — меняют. На ней видно, что комбинация
    считается на **своём** прогреве, а не на максимуме по сетке.

    Прогрев здесь равен ``span``, то есть заведомо куцый — ровно тот случай, о
    котором предупреждает контракт ``Strategy.warmup_bars``: вес начального
    состояния после ``span`` баров ещё около ``e**-2``. В фикстуре это и нужно,
    чтобы разница между своим и чужим прогревом была видна в цифрах.
    """

    name = "ewm"

    def warmup_bars(self, params: Mapping[str, Any]) -> int:
        return int(params["span"])

    def generate_signals(self, data: pl.DataFrame, params: Mapping[str, Any]) -> pl.DataFrame:
        span = int(params["span"])
        out = data.with_columns(pl.col("close").ewm_mean(span=span).alias("ewm"))
        above = pl.col("close") > pl.col("ewm")
        return out.with_columns(
            (above & ~above.shift(1).fill_null(value=False)).alias(LONG_SIGNAL),
            (~above & above.shift(1).fill_null(value=False)).alias(SHORT_SIGNAL),
        )


def test_combo_result_does_not_depend_on_the_rest_of_the_grid() -> None:
    """Добавление длинной комбинации в сетку не должно двигать цифры короткой.

    Окно режется под самую долгую комбинацию сетки, поэтому без укорачивания
    прогрева короткая комбинация получила бы чужой прогрев — и, на EWM-подобном
    индикаторе, другие значения.
    """
    data = _candles(1200)
    times = data["open_time"].to_list()
    strategy = _EwmStrategy()
    params = {"span": 50}
    own = strategy.warmup_bars(params)
    grid_max = strategy.warmup_bars({"span": 200})
    window = TimeSlice(
        id="w",
        label="w",
        start=T0 + 600 * HOUR,
        end=T0 + 900 * HOUR,
        role=TimeSliceRole.TRAIN,
        warmup_bars=0,
    )

    def score(sliced_warmup: int, combo_warmup: int) -> float:
        sliced = _slice_window(data, times, window, sliced_warmup)
        assert sliced is not None
        picked = _select_on_train(
            strategy,
            sliced,
            [params],
            [combo_warmup],
            _execution(),
            window,
            "1h",
            "total_return_pct",
            1,
            BacktestEngine(),
        )
        assert picked is not None
        return picked[1]

    alone = score(own, own)
    in_wide_grid = score(grid_max, own)
    assert in_wide_grid == alone

    # Фикстура различающая: с чужим прогревом цифра другая, иначе тест пуст.
    assert score(grid_max, grid_max) != alone


def test_reserve_metrics_do_not_depend_on_how_long_the_warmup_was() -> None:
    """Прогрев резерва не попадает в метрики — ни сделкой, ни плоским баром.

    У скользящего среднего прогрева в ``slow_ma`` баров достаточно: всё сверх него
    ничего не меняет в значениях индикатора. Значит и метрики обязаны совпасть
    побитово. Если бы участок прогрева оставался в кривой, лишние плоские бары
    развели бы ``sharpe``: он считается по побарным доходностям.
    """
    data = _candles(1400)
    reserve_from = 1000
    reserve = data.slice(reserve_from)
    strategy = get_strategy("ma_cross")

    def with_warmup(bars: int) -> Metrics:
        return evaluate_holdout(
            reserve,
            strategy,
            PARAMS,
            _execution(),
            "1h",
            min_trades=1,
            warmup_data=data.slice(reserve_from - bars, bars),
        )

    short, long = with_warmup(SLOW), with_warmup(SLOW * 3)
    assert short.sharpe == long.sharpe
    assert short.total_return_pct == long.total_return_pct
    assert short.trades_count == long.trades_count
