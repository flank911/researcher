"""Tests for AN-5 — export-results (CSV/Parquet)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from trading_research.analytics.queries import (
    PerformanceRecord,
    build_parameter_performance,
    save_performance_table,
)
from trading_research.backtest.metrics import Metrics
from trading_research.cli.analyze_results import experiment_results_path
from trading_research.cli.export_results import export_results
from trading_research.cli.main import app
from trading_research.domain import TimeSlice, TimeSliceKind

runner = CliRunner()


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


def _table() -> pl.DataFrame:
    recs = [
        PerformanceRecord(
            "exp1",
            "ma_cross",
            "BTCUSDT",
            "1h",
            _slice(sid),
            {"fast_ma": f, "slow_ma": s},
            _metrics(t),
        )
        for f, s, sid, t in [
            (5, 20, "2024-01", 10.0),
            (5, 20, "2024-02", 20.0),
            (10, 50, "2024-01", 5.0),
            (10, 50, "2024-02", 15.0),
        ]
    ]
    return build_parameter_performance(recs)


def _results(tmp_path: Path) -> Path:
    return save_performance_table(_table(), experiment_results_path(tmp_path / "results", "exp1"))


# --- business ---------------------------------------------------------------


def test_export_performance_csv(tmp_path: Path) -> None:
    out = export_results(_results(tmp_path), tmp_path / "perf.csv", kind="performance", fmt="csv")
    assert out.exists()
    df = pl.read_csv(out)
    assert df.height == 4
    assert {"param_hash", "time_slice_id", "total_return_pct"} <= set(df.columns)


def test_export_ranking_parquet(tmp_path: Path) -> None:
    out = export_results(
        _results(tmp_path), tmp_path / "rank.parquet", kind="ranking", fmt="parquet"
    )
    df = pl.read_parquet(out)
    assert "rank" in df.columns
    assert df.height == 2  # 2 комбинации параметров


def test_export_stability_csv(tmp_path: Path) -> None:
    out = export_results(_results(tmp_path), tmp_path / "stab.csv", kind="stability", fmt="csv")
    df = pl.read_csv(out)
    assert "stability_score" in df.columns


def test_export_invalid_fmt_and_kind(tmp_path: Path) -> None:
    results = _results(tmp_path)
    with pytest.raises(ValueError, match="fmt"):
        export_results(results, tmp_path / "x", fmt="xlsx")
    with pytest.raises(ValueError, match="kind"):
        export_results(results, tmp_path / "x", kind="nope")


# --- CLI smoke --------------------------------------------------------------


def test_cli_export_results_smoke(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    save_performance_table(_table(), experiment_results_path(results_root, "exp1"))
    result = runner.invoke(
        app,
        [
            "export-results",
            "--experiment",
            "exp1",
            "--kind",
            "ranking",
            "--format",
            "csv",
            "--results-root",
            str(results_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "export ->" in result.output
    exported = results_root / "exp1" / "export_ranking.csv"
    assert exported.exists()
    assert pl.read_csv(exported).height == 2
