"""Tests for AN-4 — heatmap карты параметров."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from trading_research.analytics.plots import build_heatmap, heatmap_to_png, plot_heatmap
from trading_research.analytics.queries import (
    PerformanceRecord,
    build_parameter_performance,
    save_performance_table,
)
from trading_research.backtest.metrics import Metrics
from trading_research.cli.analyze_results import experiment_results_path
from trading_research.cli.main import app
from trading_research.domain import TimeSlice, TimeSliceKind

runner = CliRunner()


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


def _rec(
    fast: int, slow: int, sid: str, total: float, *, symbol: str = "BTCUSDT"
) -> PerformanceRecord:
    return PerformanceRecord(
        experiment_id="exp1",
        strategy_name="ma_cross",
        symbol=symbol,
        timeframe="1h",
        time_slice=_slice(sid),
        params={"fast_ma": fast, "slow_ma": slow},
        metrics=_metrics(total),
    )


def _grid_table() -> pl.DataFrame:
    # median по 2 слайсам: (5,20)=15, (5,50)=0, (10,20)=30, (10,50)=10.
    recs = [
        _rec(5, 20, "2024-01", 10.0),
        _rec(5, 20, "2024-02", 20.0),
        _rec(5, 50, "2024-01", 0.0),
        _rec(5, 50, "2024-02", 0.0),
        _rec(10, 20, "2024-01", 30.0),
        _rec(10, 20, "2024-02", 30.0),
        _rec(10, 50, "2024-01", 5.0),
        _rec(10, 50, "2024-02", 15.0),
    ]
    return build_parameter_performance(recs)


# --- build_heatmap ----------------------------------------------------------


def test_build_heatmap_axes_and_values() -> None:
    data = build_heatmap(_grid_table(), x="fast_ma", y="slow_ma", metric="total_return_pct")
    assert data.xs == [5, 10]
    assert data.ys == [20, 50]
    # values[y][x]
    assert data.values[0][0] == pytest.approx(15.0)  # slow=20, fast=5
    assert data.values[0][1] == pytest.approx(30.0)  # slow=20, fast=10
    assert data.values[1][0] == pytest.approx(0.0)  # slow=50, fast=5
    assert data.values[1][1] == pytest.approx(10.0)  # slow=50, fast=10


def test_build_heatmap_mean_agg() -> None:
    data = build_heatmap(_grid_table(), x="fast_ma", y="slow_ma", agg="mean")
    assert data.values[0][0] == pytest.approx(15.0)  # mean(10,20)


def test_build_heatmap_to_frame() -> None:
    frame = build_heatmap(_grid_table(), x="fast_ma", y="slow_ma").to_frame()
    assert set(frame.columns) == {"fast_ma", "slow_ma", "total_return_pct"}
    assert frame.height == 4


def test_build_heatmap_filter_by_time_slice() -> None:
    data = build_heatmap(_grid_table(), x="fast_ma", y="slow_ma", time_slice="2024-01")
    assert data.values[1][1] == pytest.approx(5.0)  # (10,50) только янв = 5


def test_build_heatmap_missing_param_raises() -> None:
    with pytest.raises(ValueError, match="atr_period"):
        build_heatmap(_grid_table(), x="atr_period", y="slow_ma")


def test_build_heatmap_invalid_metric_agg_raise() -> None:
    table = _grid_table()
    with pytest.raises(ValueError, match="metric"):
        build_heatmap(table, x="fast_ma", y="slow_ma", metric="reliable")
    with pytest.raises(ValueError, match="agg"):
        build_heatmap(table, x="fast_ma", y="slow_ma", agg="nope")


def test_build_heatmap_empty_after_filter_raises() -> None:
    with pytest.raises(ValueError, match="no rows"):
        build_heatmap(_grid_table(), x="fast_ma", y="slow_ma", symbol="NONE")


# --- PNG rendering ----------------------------------------------------------


def _assert_png(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_plot_heatmap_writes_png(tmp_path: Path) -> None:
    data = build_heatmap(_grid_table(), x="fast_ma", y="slow_ma")
    out = plot_heatmap(data, tmp_path / "sub" / "hm.png")
    _assert_png(out)


def test_heatmap_to_png_combined(tmp_path: Path) -> None:
    out = heatmap_to_png(_grid_table(), tmp_path / "hm.png", x="fast_ma", y="slow_ma")
    _assert_png(out)


# --- CLI smoke --------------------------------------------------------------


def test_cli_plot_heatmap_smoke(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    save_performance_table(_grid_table(), experiment_results_path(results_root, "exp1"))
    result = runner.invoke(
        app,
        [
            "plot-heatmap",
            "--experiment",
            "exp1",
            "--x",
            "fast_ma",
            "--y",
            "slow_ma",
            "--metric",
            "total_return_pct",
            "--results-root",
            str(results_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "heatmap ->" in result.output
    pngs = list(results_root.rglob("*.png"))
    assert len(pngs) == 1
    _assert_png(pngs[0])
