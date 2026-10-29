"""Tests for REG-3 — correlations params ↔ market regime."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nbformat
import polars as pl
import pytest

from trading_research.analytics.correlations import (
    NUMERIC_REGIME_FEATURES,
    attach_regime_features,
    best_params_in_regime,
    regime_correlations,
    regime_metric_summary,
)
from trading_research.analytics.queries import (
    PerformanceRecord,
    build_parameter_performance,
    param_hash,
)
from trading_research.backtest.metrics import Metrics
from trading_research.domain import TimeSlice, TimeSliceKind

SLICES = ["2024-01", "2024-02", "2024-03", "2024-04"]
P1 = {"fast_ma": 5, "slow_ma": 20}
P2 = {"fast_ma": 10, "slow_ma": 50}


def _metrics(total: float) -> Metrics:
    return Metrics(total, -5.0, 1.0, 1.0, 2.0, 55.0, 0.3, 1.1, 2.0, 7, 8760.0, 30, True)


def _slice(sid: str) -> TimeSlice:
    m = int(sid.split("-")[1])
    return TimeSlice(
        id=sid,
        start=datetime(2024, m, 1, tzinfo=UTC),
        end=datetime(2024, m + 1, 1, tzinfo=UTC),
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


def _performance() -> pl.DataFrame:
    p1 = [10.0, 20.0, 30.0, 40.0]  # растёт вместе с volatility
    p2 = [40.0, 30.0, 20.0, 10.0]  # падает
    recs = [_rec(P1, s, t) for s, t in zip(SLICES, p1, strict=True)]
    recs += [_rec(P2, s, t) for s, t in zip(SLICES, p2, strict=True)]
    return build_parameter_performance(recs)


def _features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "timeframe": ["1h"] * 4,
            "time_slice_id": SLICES,
            "market_return_pct": [1.0, 2.0, 3.0, 4.0],
            "volatility": [1.0, 2.0, 3.0, 4.0],
            "trend_strength": [4.0, 3.0, 2.0, 1.0],
            "funding_avg": [0.0001, 0.0002, 0.0003, 0.0004],
            "oi_change_pct": [1.0, 1.0, 1.0, 1.0],
            "ls_ratio_avg": [1.5, 1.5, 1.5, 1.5],
            "regime_label": [
                "sideways_low_vol",
                "sideways_high_vol",
                "bull_trend",
                "bull_trend",
            ],
        }
    )


# --- attach -----------------------------------------------------------------


def test_attach_fills_regime_slots() -> None:
    joined = attach_regime_features(_performance(), _features())
    assert joined.height == 8  # 2 params × 4 slices, left-join сохранил все
    assert joined["volatility"].null_count() == 0
    assert "regime_label" in joined.columns
    # Значения режима подтянуты из features по слайсу.
    jan = joined.filter(
        (pl.col("param_hash") == param_hash(P1)) & (pl.col("time_slice_id") == "2024-01")
    )
    assert jan["volatility"][0] == pytest.approx(1.0)


def test_attach_missing_keys_raise() -> None:
    bad = _features().drop("time_slice_id")
    with pytest.raises(ValueError, match="join keys missing"):
        attach_regime_features(_performance(), bad)


# --- correlations -----------------------------------------------------------


def test_regime_correlations_matrix_per_param() -> None:
    joined = attach_regime_features(_performance(), _features())
    corr = regime_correlations(joined, metric="total_return_pct")
    assert set(NUMERIC_REGIME_FEATURES) <= set(corr.columns)

    p1 = corr.filter(pl.col("param_hash") == param_hash(P1))
    assert p1["volatility"][0] == pytest.approx(1.0)  # total_return растёт с volatility
    assert p1["trend_strength"][0] == pytest.approx(-1.0)  # и падает с trend

    p2 = corr.filter(pl.col("param_hash") == param_hash(P2))
    assert p2["volatility"][0] == pytest.approx(-1.0)


def test_regime_correlations_global() -> None:
    joined = attach_regime_features(_performance(), _features())
    corr = regime_correlations(joined, by=None)
    assert corr.height == 1
    assert "volatility" in corr.columns


def test_regime_correlations_invalid_metric() -> None:
    joined = attach_regime_features(_performance(), _features())
    with pytest.raises(ValueError, match="metric"):
        regime_correlations(joined, metric="nope")


# --- summary / best-in-regime (вопросы раздела 9) ---------------------------


def test_regime_metric_summary() -> None:
    joined = attach_regime_features(_performance(), _features())
    summary = regime_metric_summary(joined, metric="total_return_pct")
    # P1 в bull_trend: слайсы 03,04 → среднее (30+40)/2 = 35.
    row = summary.filter(
        (pl.col("param_hash") == param_hash(P1)) & (pl.col("regime_label") == "bull_trend")
    )
    assert row["total_return_pct"][0] == pytest.approx(35.0)
    assert row["n"][0] == 2


def test_best_params_in_regime() -> None:
    joined = attach_regime_features(_performance(), _features())
    best = best_params_in_regime(joined, regime_label="bull_trend", metric="total_return_pct")
    # В bull_trend P1 (ср. 35) > P2 (ср. 15) → P1 первый.
    assert best["param_hash"][0] == param_hash(P1)
    assert best["total_return_pct"][0] == pytest.approx(35.0)


def test_best_params_in_regime_invalid_top() -> None:
    joined = attach_regime_features(_performance(), _features())
    with pytest.raises(ValueError, match="top"):
        best_params_in_regime(joined, regime_label="bull_trend", top=0)


# --- notebook ---------------------------------------------------------------

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "003_market_regimes.ipynb"


def test_notebook_003_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TR_PERFORMANCE", raising=False)
    monkeypatch.delenv("TR_FEATURES", raising=False)
    monkeypatch.chdir(tmp_path)
    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n\n".join(c.source for c in nb.cells if c.cell_type == "code")
    namespace: dict[str, object] = {"__name__": "__main__"}
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)  # noqa: S102 — доверенный ноутбук
    out = namespace["out"]
    assert isinstance(out, Path)
    assert out.read_bytes()[:4] == b"\x89PNG"
