"""Tests for REG-2 — market-regime features + regime_label."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from trading_research.data.schema import UTC_MS
from trading_research.domain import TimeSlice, TimeSliceKind
from trading_research.features.funding_features import funding_avg
from trading_research.features.indicators import (
    adx,
    drawdown_from_ath,
    market_return_pct,
    trend_strength,
    volatility_pct,
    volume_zscore,
)
from trading_research.features.ls_ratio_features import long_short_ratio_avg
from trading_research.features.market_regime import (
    TIME_SLICE_FEATURE_COLUMNS,
    RegimeLabel,
    RegimeThresholds,
    build_time_slice_features,
    classify_regime,
    compute_slice_features,
)
from trading_research.features.oi_features import oi_change_pct

T0 = datetime(2024, 1, 1, tzinfo=UTC)
H = timedelta(hours=1)
THR = RegimeThresholds()


def _candles(closes: list[float], *, start: datetime = T0, volume: float = 10.0) -> pl.DataFrame:
    n = len(closes)
    return pl.DataFrame(
        {
            "open_time": [start + i * H for i in range(n)],
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [volume] * n,
        },
        schema_overrides={"open_time": UTC_MS},
    )


# --- indicators -------------------------------------------------------------


def test_market_return_and_drawdown() -> None:
    assert market_return_pct([100.0, 110.0]) == pytest.approx(10.0)
    assert market_return_pct([100.0]) == 0.0
    assert drawdown_from_ath([100.0, 80.0, 120.0]) == pytest.approx(-20.0)
    assert drawdown_from_ath([100.0, 110.0, 120.0]) == pytest.approx(0.0)


def test_trend_strength_sign() -> None:
    assert trend_strength([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)
    assert trend_strength([4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert trend_strength([5.0, 5.0, 5.0]) == 0.0  # нет дисперсии


def test_volatility_and_volume_zscore() -> None:
    assert volatility_pct([100.0, 100.0, 100.0]) == pytest.approx(0.0)
    assert volatility_pct([100.0, 110.0, 99.0, 120.0]) > 0.0
    assert volume_zscore([10.0, 10.0, 10.0]) == 0.0
    assert volume_zscore([10.0, 10.0, 10.0, 30.0]) > 0.0


def test_adx_trending_gt_sideways() -> None:
    up = [100.0 + i for i in range(40)]
    choppy = [100.0 + (5.0 if i % 2 else -5.0) for i in range(40)]
    trend_adx = adx([c + 1 for c in up], [c - 1 for c in up], up)
    chop_adx = adx([c + 1 for c in choppy], [c - 1 for c in choppy], choppy)
    assert trend_adx > chop_adx
    assert adx([1.0, 2.0], [0.5, 1.5], [1.0, 2.0]) == 0.0  # мало данных


# --- funding / oi / ls ------------------------------------------------------


def test_series_features_and_none() -> None:
    funding = pl.DataFrame(
        {"open_time": [T0], "funding_rate": [0.001]}, schema_overrides={"open_time": UTC_MS}
    )
    oi = pl.DataFrame(
        {"open_time": [T0, T0 + H], "open_interest": [1000.0, 1100.0]},
        schema_overrides={"open_time": UTC_MS},
    )
    ls = pl.DataFrame(
        {"open_time": [T0], "long_short_ratio": [1.5]}, schema_overrides={"open_time": UTC_MS}
    )
    assert funding_avg(funding) == pytest.approx(0.001)
    assert oi_change_pct(oi) == pytest.approx(10.0)
    assert long_short_ratio_avg(ls) == pytest.approx(1.5)
    assert funding_avg(None) is None
    assert oi_change_pct(None) is None
    assert long_short_ratio_avg(None) is None


# --- classify_regime rules --------------------------------------------------


def _label(**kw: float | None) -> RegimeLabel:
    base: dict[str, float | None] = {
        "market_return": 0.0,
        "volatility": 0.1,
        "trend": 0.0,
        "drawdown": 0.0,
        "funding": None,
    }
    base.update(kw)
    return classify_regime(**base, thresholds=THR)  # type: ignore[arg-type]


def test_classify_rules() -> None:
    assert _label(funding=0.001) is RegimeLabel.OVERHEATED_FUNDING
    assert _label(market_return=-20.0, volatility=5.0) is RegimeLabel.PANIC_DUMP
    assert _label(market_return=10.0, drawdown=-15.0) is RegimeLabel.RECOVERY
    assert _label(trend=0.9, market_return=20.0) is RegimeLabel.BULL_TREND
    assert _label(trend=-0.9, market_return=-20.0, volatility=0.1) is RegimeLabel.BEAR_TREND
    assert _label(volatility=2.0) is RegimeLabel.SIDEWAYS_HIGH_VOL
    assert _label(volatility=0.1) is RegimeLabel.SIDEWAYS_LOW_VOL


# --- compute / build --------------------------------------------------------


def test_compute_slice_features_bull() -> None:
    closes = [100.0 * (1.01**i) for i in range(30)]  # устойчивый рост
    feat = compute_slice_features(
        _candles(closes), symbol="BTCUSDT", timeframe="1h", time_slice_id="2024-01"
    )
    assert feat.regime_label == RegimeLabel.BULL_TREND.value
    assert feat.market_return_pct > 0
    assert feat.trend_strength > 0.5


def test_build_time_slice_features_table() -> None:
    # Два месячных слайса: январь (рост) и февраль (падение).
    jan_closes = [100.0 + i for i in range(48)]
    feb_closes = [200.0 - i for i in range(48)]
    candles = pl.concat(
        [
            _candles(jan_closes, start=T0),
            _candles(feb_closes, start=datetime(2024, 2, 1, tzinfo=UTC)),
        ]
    )
    slices = [
        TimeSlice(
            id="2024-01",
            start=T0,
            end=datetime(2024, 2, 1, tzinfo=UTC),
            label="2024-01",
            kind=TimeSliceKind.MONTH,
        ),
        TimeSlice(
            id="2024-02",
            start=datetime(2024, 2, 1, tzinfo=UTC),
            end=datetime(2024, 3, 1, tzinfo=UTC),
            label="2024-02",
            kind=TimeSliceKind.MONTH,
        ),
    ]
    funding = pl.DataFrame(
        {"open_time": [T0, T0 + H], "funding_rate": [0.0001, 0.0002]},
        schema_overrides={"open_time": UTC_MS},
    )

    table = build_time_slice_features(
        candles, slices, symbol="BTCUSDT", timeframe="1h", funding=funding
    )
    assert tuple(table.columns) == TIME_SLICE_FEATURE_COLUMNS
    assert table.height == 2
    # Колонки режима совпадают по именам со слотами AN-1.
    assert {
        "market_return_pct",
        "volatility",
        "trend_strength",
        "funding_avg",
        "oi_change_pct",
        "ls_ratio_avg",
        "regime_label",
    } <= set(table.columns)

    jan = table.filter(pl.col("time_slice_id") == "2024-01")
    feb = table.filter(pl.col("time_slice_id") == "2024-02")
    # Каждый слайс — только из своих баров (нет утечки будущего).
    assert jan["market_return_pct"][0] > 0  # январь растёт
    assert feb["market_return_pct"][0] < 0  # февраль падает
    # funding есть только в январе → в феврале null.
    assert jan["funding_avg"][0] is not None
    assert feb["funding_avg"][0] is None


def test_build_skips_empty_slices() -> None:
    candles = _candles([100.0 + i for i in range(10)])
    empty_slice = [
        TimeSlice(
            id="2025-01",
            start=datetime(2025, 1, 1, tzinfo=UTC),
            end=datetime(2025, 2, 1, tzinfo=UTC),
            label="2025-01",
            kind=TimeSliceKind.MONTH,
        )
    ]
    table = build_time_slice_features(candles, empty_slice, symbol="BTCUSDT", timeframe="1h")
    assert table.height == 0
    assert tuple(table.columns) == TIME_SLICE_FEATURE_COLUMNS
