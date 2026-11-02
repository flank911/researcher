"""Tests for RES-1 — индикатор RSI и стратегия возврата к среднему.

Индикатор проверяется против значений, посчитанных **независимо**: вручную по
определению Уайлдера (числа в комментариях выведены арифметически, не прогоном
кода) и против pandas, где сглаживание считает чужая реализация.

Стратегия проверяется на то, ради чего она добавлена: сигналы — события входа в
зону, значения не зависят от того, с какого бара начался срез (при объявленном
прогреве), и сигнал на баре ``i`` не зависит от баров после ``i``.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from trading_research.features.indicators import rsi
from trading_research.strategies import available_strategies, get_strategy
from trading_research.strategies.base import LONG_SIGNAL, SHORT_SIGNAL, Strategy
from trading_research.strategies.rsi_reversion import (
    RSI_COLUMN,
    WARMUP_PERIODS,
    RsiReversionStrategy,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)
HOUR = timedelta(hours=1)
LEVELS = {"oversold": 30.0, "overbought": 70.0}


def _frame(closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open_time": [T0 + i * HOUR for i in range(len(closes))],
            "close": [float(c) for c in closes],
        }
    )


def _wave(n: int, *, amplitude: float = 8.0, period_bars: float = 40.0) -> list[float]:
    """Колебание вокруг тренда: RSI ходит через оба порога."""
    return [
        100.0 + 0.01 * i + amplitude * math.sin(2 * math.pi * i / period_bars) for i in range(n)
    ]


# --- индикатор: сверка с ручным счётом --------------------------------------


def test_rsi_matches_hand_computed_wilder_values() -> None:
    """Четыре значения, выведенные по определению вручную.

    Ряд ``[100, 101, 100, 102, 101, 103, 102, 104]``, ``period=4``. Приросты:
    ``+1, -1, +2, -1, +2, -1, +2``; все числа — точные двоичные дроби, поэтому
    арифметика ниже воспроизводима без допусков.

    Затравка (индекс 4): рост ``(1+0+2+0)/4 = 0,75``, падение ``(0+1+0+1)/4 = 0,5``
    → ``100 * 0,75 / 1,25 = 60``.
    Индекс 5 (прирост +2): рост ``(0,75*3 + 2)/4 = 1,0625``, падение
    ``0,5*3/4 = 0,375`` → ``100 * 1,0625 / 1,4375 = 73,913…``
    Индекс 6 (прирост −1): рост ``1,0625*3/4 = 0,796875``, падение
    ``(0,375*3 + 1)/4 = 0,53125`` → ``100 * 0,796875 / 1,328125 = 60`` ровно.
    Индекс 7 (прирост +2): рост ``(0,796875*3 + 2)/4 = 1,09765625``, падение
    ``0,53125*3/4 = 0,3984375``; в долях ``2**-8`` это ``281`` и ``102``, значит
    ``RSI = 100 * 281 / 383``.
    """
    values = rsi([100, 101, 100, 102, 101, 103, 102, 104], period=4)

    assert values[:4] == [None, None, None, None]
    assert values[4] == 60.0
    assert values[5] == pytest.approx(100 * 1.0625 / 1.4375, rel=1e-15)
    assert values[6] == 60.0
    assert values[7] == pytest.approx(100 * 281 / 383, rel=1e-15)


def test_rsi_matches_a_pandas_reference() -> None:
    """Сверка с чужой реализацией сглаживания: ту же рекурсию считает pandas."""
    pd = pytest.importorskip("pandas")
    closes = _wave(400, amplitude=6.0, period_bars=37.0)

    def reference(period: int) -> list[float | None]:
        deltas = pd.Series(closes, dtype="float64").diff().dropna().reset_index(drop=True)
        gain, loss = deltas.clip(lower=0), (-deltas).clip(lower=0)

        def smoothed(x: object) -> object:
            seed = x.iloc[:period].mean()  # type: ignore[attr-defined]
            tail = pd.concat([pd.Series([seed]), x.iloc[period:]], ignore_index=True)  # type: ignore[attr-defined]
            return tail.ewm(alpha=1 / period, adjust=False).mean()

        avg_gain, avg_loss = smoothed(gain), smoothed(loss)
        out: list[float | None] = [None] * len(closes)
        for k in range(len(avg_gain)):  # type: ignore[arg-type]
            total = avg_gain.iloc[k] + avg_loss.iloc[k]  # type: ignore[attr-defined]
            out[period + k] = None if total == 0 else 100 * avg_gain.iloc[k] / total  # type: ignore[attr-defined]
        return out

    for period in (7, 14, 21):
        mine, theirs = rsi(closes, period=period), reference(period)
        pairs = [(a, b) for a, b in zip(mine, theirs, strict=True) if a is not None]
        assert len(pairs) == len(closes) - period
        assert max(abs(a - b) for a, b in pairs) < 1e-10  # type: ignore[operator]


def test_rsi_has_no_value_before_the_seed() -> None:
    values = rsi(_wave(60), period=14)
    assert values[:14] == [None] * 14
    assert values[14] is not None


def test_flat_series_has_no_rsi_instead_of_zero() -> None:
    """Ряд стоит на месте — значение не определено. Ноль здесь был бы сигналом."""
    assert rsi([100.0] * 30, period=14) == [None] * 30


def test_rsi_is_100_without_losses_and_0_without_gains() -> None:
    assert rsi([100, 101, 102, 103, 104, 105], period=3)[-1] == 100.0
    assert rsi([105, 104, 103, 102, 101, 100], period=3)[-1] == 0.0


def test_rsi_at_bar_i_does_not_depend_on_later_bars() -> None:
    """Антипод look-ahead на уровне индикатора: значение считается по префиксу."""
    closes = _wave(200)
    full = rsi(closes, period=14)
    for i in (20, 57, 123, 199):
        assert rsi(closes[: i + 1], period=14)[i] == full[i]


def test_rsi_rejects_degenerate_period() -> None:
    with pytest.raises(ValueError, match="period must be >= 1"):
        rsi([1.0, 2.0], period=0)


# --- стратегия --------------------------------------------------------------


def _params(period: int = 14) -> dict[str, float]:
    return {"rsi_period": period, **LEVELS}


def test_registry_resolves_both_strategies() -> None:
    assert available_strategies() == ["ma_cross", "rsi_reversion"]
    strat = get_strategy("rsi_reversion")
    assert strat.name == "rsi_reversion"
    assert isinstance(strat, Strategy)


def test_warmup_is_a_multiple_of_the_period_not_the_period() -> None:
    """Прогрев — осознанный запас: у сглаживания Уайлдера точного прогрева нет."""
    strat = RsiReversionStrategy()
    assert strat.warmup_bars(_params(14)) == WARMUP_PERIODS * 14
    assert strat.warmup_bars(_params(7)) == WARMUP_PERIODS * 7
    assert WARMUP_PERIODS > 1


def test_signals_are_clean_booleans_and_silent_on_warmup() -> None:
    strat = RsiReversionStrategy()
    df = strat.generate_signals(_frame(_wave(300)), _params())
    for col in (LONG_SIGNAL, SHORT_SIGNAL):
        assert df.schema[col] == pl.Boolean
        assert df[col].null_count() == 0
    # На участке без значения RSI сигналов нет.
    assert not any(df[LONG_SIGNAL][:14].to_list())
    assert not any(df[SHORT_SIGNAL][:14].to_list())
    assert df[LONG_SIGNAL].sum() > 0
    assert df[SHORT_SIGNAL].sum() > 0


def test_signal_is_the_crossing_not_the_state() -> None:
    """Сигнал — вход в зону. Пока RSI внутри зоны, нового сигнала нет."""
    strat = RsiReversionStrategy()
    df = strat.generate_signals(_frame(_wave(300)), _params())
    rows = df.select(RSI_COLUMN, LONG_SIGNAL, SHORT_SIGNAL).rows()

    inside_oversold = [i for i, (v, _, _) in enumerate(rows) if v is not None and v < 30.0]
    longs = [i for i, (_, lg, _) in enumerate(rows) if lg]
    assert longs, "фикстура обязана заводить RSI в перепроданность"
    # Каждый сигнал — первый бар своей серии внутри зоны.
    for i in longs:
        assert i in inside_oversold
        assert i - 1 not in inside_oversold
    # И наоборот: внутри зоны сигналов больше, чем входов в неё.
    assert len(inside_oversold) > len(longs)


def test_trend_separates_the_two_strategies() -> None:
    """Зачем нужна вторая стратегия: на тренде их сигналы противоположны.

    MA Cross на устойчивом росте входит в long, возврат к среднему — в short. Это
    и делает прогон разделяющим: совпадение их выводов нельзя будет объяснить тем,
    что обе смотрят на одно и то же.
    """
    # Сначала боковик, потом устойчивый рост: без боковика у MA Cross нет события
    # пересечения — быстрая средняя лежит выше медленной с самого начала.
    flat = [100.0 + 0.4 * math.sin(2 * math.pi * i / 13.0) for i in range(120)]
    closes = flat + [flat[-1] * (1.004 ** (i + 1)) for i in range(320)]
    frame = _frame(closes)
    trend = get_strategy("ma_cross").generate_signals(frame, {"fast_ma": 10, "slow_ma": 50})
    reversion = RsiReversionStrategy().generate_signals(frame, _params())

    def fired(frame: pl.DataFrame, column: str) -> list[int]:
        return [i for i, v in enumerate(frame[column].to_list()) if v]

    # На самом переломе обе стратегии срабатывают на одном баре — в разные стороны.
    turn = fired(trend, LONG_SIGNAL)[-1]
    assert fired(reversion, SHORT_SIGNAL) == [turn]
    # Дальше тренд монотонный: пересечений нет ни у одной, и long у возврата к
    # среднему не появляется — перепроданности на росте не бывает.
    assert [i for i in fired(trend, SHORT_SIGNAL) if i >= turn] == []
    assert fired(reversion, LONG_SIGNAL) == []


def test_signal_at_bar_i_does_not_depend_on_later_bars() -> None:
    strat = RsiReversionStrategy()
    closes = _wave(300)
    full = strat.generate_signals(_frame(closes), _params())
    for i in (60, 150, 299):
        cut = strat.generate_signals(_frame(closes[: i + 1]), _params())
        assert cut[LONG_SIGNAL][i] == full[LONG_SIGNAL][i]
        assert cut[SHORT_SIGNAL][i] == full[SHORT_SIGNAL][i]


def test_signals_after_warmup_do_not_depend_on_where_the_slice_began() -> None:
    """DoD RES-1: объявленного прогрева хватает, чтобы срез совпал с полной историей.

    И обратное: с прогревом в один период сигналы уже расходятся — значит запас взят
    не «на всякий случай», а по необходимости.
    """
    strat = RsiReversionStrategy()
    closes = _wave(1200, amplitude=7.0, period_bars=31.0)
    params = _params()
    start, length = 600, 300
    declared = strat.warmup_bars(params)

    def window_signals(warmup: int) -> list[tuple[bool, bool]]:
        piece = strat.generate_signals(_frame(closes[start - warmup : start + length]), params)
        return list(
            zip(
                piece[LONG_SIGNAL][warmup:].to_list(),
                piece[SHORT_SIGNAL][warmup:].to_list(),
                strict=True,
            )
        )

    full = strat.generate_signals(_frame(closes), params)
    reference = list(
        zip(
            full[LONG_SIGNAL][start : start + length].to_list(),
            full[SHORT_SIGNAL][start : start + length].to_list(),
            strict=True,
        )
    )
    assert any(lg or sh for lg, sh in reference), "в окне обязаны быть сигналы"

    assert window_signals(declared) == reference
    assert window_signals(int(params["rsi_period"])) != reference


def test_invalid_params_raise() -> None:
    strat = RsiReversionStrategy()
    with pytest.raises(ValueError, match="rsi_period must be >= 2"):
        strat.warmup_bars({"rsi_period": 1, **LEVELS})
    with pytest.raises(ValueError, match="0 < oversold < overbought < 100"):
        strat.generate_signals(
            _frame(_wave(50)), {"rsi_period": 14, "oversold": 70, "overbought": 30}
        )
    with pytest.raises(KeyError, match="rsi_reversion requires param"):
        strat.warmup_bars({"rsi_period": 14, "oversold": 30})
