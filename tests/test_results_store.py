"""Tests for EXP-4 — results store (Parquet artifacts + SQLite catalog)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from trading_research.backtest.metrics import compute_metrics
from trading_research.backtest.orders import OrderReason, Side
from trading_research.backtest.portfolio import Trade
from trading_research.domain import ExecutionModel, PositionSizeType
from trading_research.experiments.cache import RunSpec
from trading_research.experiments.results_store import (
    ResultsStore,
    RunStatus,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _execution() -> ExecutionModel:
    return ExecutionModel(
        initial_balance=10_000.0,
        commission_rate=0.0006,
        slippage_pct=0.0003,
        leverage=3,
        position_size_type=PositionSizeType.PERCENT_BALANCE,
        position_size_value=20.0,
    )


def _spec(slice_id: str = "2024-01", **overrides: object) -> RunSpec:
    base: dict[str, object] = {
        "strategy": "ma_cross",
        "params": {"fast_ma": 20, "slow_ma": 100},
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "slice_id": slice_id,
        "dataset_hash": "abc123",
        "execution": _execution(),
        "git_commit": "deadbeef",
    }
    base.update(overrides)
    return RunSpec(**base)  # type: ignore[arg-type]


def _trades() -> list[Trade]:
    return [
        Trade(
            entry_time=T0,
            exit_time=T0 + timedelta(hours=2),
            side=Side.LONG,
            entry_price=100.0,
            exit_price=110.0,
            qty=1.5,
            pnl=15.0,
            pnl_pct=10.0,
            fees=0.2,
            reason=OrderReason.REVERSE,
        ),
        Trade(
            entry_time=T0 + timedelta(hours=2),
            exit_time=T0 + timedelta(hours=4),
            side=Side.SHORT,
            entry_price=110.0,
            exit_price=105.0,
            qty=1.5,
            pnl=7.5,
            pnl_pct=4.5,
            fees=0.2,
            reason=OrderReason.FINAL,
        ),
    ]


def _equity() -> pl.DataFrame:
    times = [T0 + timedelta(hours=i) for i in range(5)]
    return pl.DataFrame(
        {"open_time": times, "equity": [10000.0, 10010.0, 10015.0, 10012.0, 10022.5]}
    )


def _metrics() -> object:
    return compute_metrics(_equity(), _trades(), "1h", min_trades=1)


# --- roundtrip --------------------------------------------------------------


def test_save_and_get_run_roundtrip(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path)
    spec = _spec()
    metrics = compute_metrics(_equity(), _trades(), "1h", min_trades=1)
    run_hash = store.save_run(spec, metrics, _trades(), _equity())

    rec = store.get_run(run_hash)
    assert rec is not None
    assert rec.run_hash == run_hash
    assert rec.status is RunStatus.COMPLETED
    assert rec.dataset_hash == "abc123"
    assert rec.git_commit == "deadbeef"
    assert rec.params == {"fast_ma": 20, "slow_ma": 100}
    assert rec.metrics is not None
    assert rec.metrics.total_return_pct == pytest.approx(metrics.total_return_pct)
    assert rec.metrics.trades_count == 2
    assert rec.metrics.reliable is True


def test_artifacts_readable_by_run_hash(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path)
    run_hash = store.save_run(_spec(), _metrics(), _trades(), _equity())  # type: ignore[arg-type]

    trades = store.load_trades(run_hash)
    equity = store.load_equity(run_hash)
    assert trades.height == 2
    assert set(trades["side"].to_list()) == {"long", "short"}
    assert equity.equals(_equity())


def test_artifacts_partitioned_by_run(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path)
    run_hash = store.save_run(_spec(), _metrics(), _trades(), _equity())  # type: ignore[arg-type]
    run_dir = tmp_path / "results" / f"run_hash={run_hash}"
    assert (run_dir / "trades.parquet").exists()
    assert (run_dir / "equity.parquet").exists()


def test_get_missing_run_returns_none(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path)
    assert store.get_run("nope") is None


def test_failed_run_records_error_without_metrics(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path)
    run_hash = store.save_run(
        _spec(),
        None,
        [],
        pl.DataFrame(),
        status=RunStatus.FAILED,
        error="engine blew up",
    )
    rec = store.get_run(run_hash)
    assert rec is not None
    assert rec.status is RunStatus.FAILED
    assert rec.error == "engine blew up"
    assert rec.metrics is None
    assert rec.trades_path is None


def test_resave_is_idempotent(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path)
    spec = _spec()
    store.save_run(spec, _metrics(), _trades(), _equity())  # type: ignore[arg-type]
    store.save_run(spec, _metrics(), _trades(), _equity())  # type: ignore[arg-type]
    assert store.count() == 1


# --- concurrency / scale ----------------------------------------------------


def test_concurrent_writers_1000_runs(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path)
    metrics = compute_metrics(_equity(), _trades(), "1h", min_trades=1)
    n = 1000

    def write(i: int) -> str:
        # Уникальный slice_id → уникальный run_hash на каждую задачу.
        return store.save_run(_spec(slice_id=f"s{i:04d}"), metrics, _trades(), _equity())

    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(write, range(n)))

    assert len(set(hashes)) == n
    assert store.count() == n
    # Каждый ран читается по run_hash.
    sample = hashes[0]
    rec = store.get_run(sample)
    assert rec is not None and rec.metrics is not None
    assert store.load_trades(sample).height == 2
