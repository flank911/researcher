"""Tests for AN-3 — robust stability vector."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl
import pytest

from trading_research.analytics.queries import (
    PerformanceRecord,
    build_parameter_performance,
    param_hash,
)
from trading_research.analytics.stability import rank_by_stability, stability_vector
from trading_research.backtest.metrics import Metrics
from trading_research.domain import TimeSlice, TimeSliceKind


def _metrics(total: float) -> Metrics:
    return Metrics(
        total_return_pct=total,
        max_drawdown_pct=-5.0,
        sharpe=1.0,
        sortino=1.0,
        profit_factor=2.0,
        win_rate=55.0,
        avg_trade_pct=0.3,
        expectancy=1.1,
        recovery_factor=2.0,
        trades_count=7,
        annualization_factor=8760.0,
        min_trades=30,
        reliable=True,
    )


def _slice(sid: str) -> TimeSlice:
    month = int(sid.split("-")[1])
    return TimeSlice(
        id=sid,
        start=datetime(2024, month, 1, tzinfo=UTC),
        end=datetime(2024, month + 1, 1, tzinfo=UTC),
        label=sid,
        kind=TimeSliceKind.MONTH,
    )


def _rec(params: dict[str, Any], sid: str, total: float) -> PerformanceRecord:
    return PerformanceRecord(
        experiment_id="exp1",
        strategy_name="ma_cross",
        symbol="BTCUSDT",
        timeframe="1h",
        time_slice=_slice(sid),
        params=params,
        metrics=_metrics(total),
    )


SLICES = ["2024-01", "2024-02", "2024-03", "2024-04"]
PEAK = {"fast_ma": 5, "slow_ma": 20}  # Params A — один пик, остальное минус
STABLE = {"fast_ma": 10, "slow_ma": 50}  # Params B — ровные плюсы


def _ab_table() -> pl.DataFrame:
    peak_returns = [100.0, -20.0, -20.0, -20.0]
    stable_returns = [8.0, 10.0, 9.0, 11.0]
    records = [_rec(PEAK, s, r) for s, r in zip(SLICES, peak_returns, strict=True)]
    records += [_rec(STABLE, s, r) for s, r in zip(SLICES, stable_returns, strict=True)]
    return build_parameter_performance(records)


# --- core: stable beats peak ------------------------------------------------


def test_stable_ranks_above_peak() -> None:
    vec = stability_vector(_ab_table(), top_quantile=0.5)
    by_hash = {row["param_hash"]: row for row in vec.to_dicts()}
    peak = by_hash[param_hash(PEAK)]
    stable = by_hash[param_hash(STABLE)]

    assert stable["stability_score"] > peak["stability_score"]
    assert stable["positive_slices_ratio"] == pytest.approx(1.0)
    assert peak["positive_slices_ratio"] == pytest.approx(0.25)
    assert stable["top_quantile_freq"] > peak["top_quantile_freq"]
    assert stable["worst_slice"] == pytest.approx(8.0)
    assert peak["worst_slice"] == pytest.approx(-20.0)

    ranked = rank_by_stability(vec)
    assert ranked["rank"].to_list() == [1, 2]
    assert ranked["param_hash"][0] == param_hash(STABLE)


def test_median_is_robust_to_peak() -> None:
    vec = stability_vector(_ab_table())
    peak = vec.filter(pl.col("param_hash") == param_hash(PEAK))
    # Медиана [100,-20,-20,-20] = -20 (пик не тянет её вверх).
    assert peak["median_return"][0] == pytest.approx(-20.0)


# --- top_quantile_freq set correctness --------------------------------------


def test_top_quantile_freq_uses_within_slice_set() -> None:
    # Один слайс, 4 комбинации; верхний квантиль 25% → ровно 1 «топ».
    params = [{"p": i} for i in range(4)]
    returns = [40.0, 30.0, 20.0, 10.0]
    table = build_parameter_performance(
        [_rec(p, "2024-01", r) for p, r in zip(params, returns, strict=True)]
    )
    vec = stability_vector(table, top_quantile=0.25)
    freq = {row["param_hash"]: row["top_quantile_freq"] for row in vec.to_dicts()}
    assert freq[param_hash({"p": 0})] == pytest.approx(1.0)  # лучший
    assert freq[param_hash({"p": 3})] == pytest.approx(0.0)  # худший


# --- boundary cases: no NaN/inf ---------------------------------------------


def _assert_finite(vec: pl.DataFrame) -> None:
    for col in ("median_return", "iqr", "return_std", "stability_score", "top_quantile_freq"):
        series = vec[col]
        assert series.null_count() == 0
        assert series.is_finite().all()


def test_single_slice_no_nan_inf() -> None:
    table = build_parameter_performance([_rec(STABLE, "2024-01", 10.0)])
    vec = stability_vector(table)
    assert vec["n_slices"][0] == 1
    assert vec["iqr"][0] == pytest.approx(0.0)
    assert vec["return_std"][0] == pytest.approx(0.0)
    _assert_finite(vec)


def test_zero_variance_no_nan_inf() -> None:
    table = build_parameter_performance([_rec(STABLE, s, 5.0) for s in SLICES])
    vec = stability_vector(table)
    assert vec["iqr"][0] == pytest.approx(0.0)
    assert vec["stability_score"][0] == pytest.approx(5.0)  # 5*1 - 0.5*0
    _assert_finite(vec)


# --- validation -------------------------------------------------------------


def test_invalid_args_raise() -> None:
    table = _ab_table()
    with pytest.raises(ValueError, match="return_metric"):
        stability_vector(table, return_metric="reliable")
    with pytest.raises(ValueError, match="top_quantile"):
        stability_vector(table, top_quantile=0.0)
    with pytest.raises(ValueError, match="top_quantile"):
        stability_vector(table, top_quantile=1.5)
    with pytest.raises(ValueError, match="top"):
        rank_by_stability(stability_vector(table), top=0)


def test_vector_columns() -> None:
    vec = stability_vector(_ab_table())
    assert {
        "param_hash",
        "params_json",
        "n_slices",
        "median_return",
        "positive_slices_ratio",
        "worst_slice",
        "top_quantile_freq",
        "iqr",
        "return_std",
        "stability_score",
    } == set(vec.columns)
