"""Tests for EXP-5 — параллельное исполнение прогонов."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import polars as pl
import pytest

from trading_research.domain import ExecutionModel, PositionSizeType, TimeSlice
from trading_research.experiments.cache import RunLedger, RunSpec, build_run_specs
from trading_research.experiments.parallel import execute_runs_parallel
from trading_research.experiments.results_store import ResultsStore


def _execution() -> ExecutionModel:
    return ExecutionModel(
        initial_balance=10_000.0,
        commission_rate=0.0006,
        slippage_pct=0.0003,
        leverage=3,
        position_size_type=PositionSizeType.PERCENT_BALANCE,
        position_size_value=20.0,
    )


def _slice(sid: str) -> TimeSlice:
    return TimeSlice(
        id=sid,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 2, 1, tzinfo=UTC),
        label=sid,
    )


def _specs(n_combos: int, n_slices: int) -> list[RunSpec]:
    combos = [{"fast_ma": 10 + i, "slow_ma": 100} for i in range(n_combos)]
    slices = [_slice(f"2024-{i + 1:02d}") for i in range(n_slices)]
    return build_run_specs(
        strategy="ma_cross",
        symbol="BTCUSDT",
        timeframe="1h",
        dataset_hash="abc",
        execution=_execution(),
        param_combos=combos,
        slices=slices,
    )


# --- picklable worker callables (module-level → переживают spawn) -----------


def _touch(out_dir: Path, spec: RunSpec) -> None:
    (out_dir / spec.run_hash()).write_text("ok", encoding="utf-8")


def _touch_or_fail(out_dir: Path, fail: frozenset[str], spec: RunSpec) -> None:
    if spec.run_hash() in fail:
        raise RuntimeError("worker boom")
    _touch(out_dir, spec)


def _sleep(seconds: float, spec: RunSpec) -> None:
    time.sleep(seconds)


def _save(root: Path, spec: RunSpec) -> None:
    store = ResultsStore(root)
    equity = pl.DataFrame({"open_time": [datetime(2024, 1, 1, tzinfo=UTC)], "equity": [10_000.0]})
    store.save_run(spec, metrics=None, trades=[], equity_curve=equity)


# --- dedupe -----------------------------------------------------------------


def test_runs_all_then_skips_on_rerun(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "ledger")
    out = tmp_path / "out"
    out.mkdir()
    specs = _specs(3, 2)  # 6

    first = execute_runs_parallel(specs, ledger, partial(_touch, out), max_workers=2)
    assert first.executed == 6
    assert first.skipped == 0
    assert first.failed == 0
    assert len(list(out.glob("*"))) == 6
    assert ledger.completed() == {s.run_hash() for s in specs}

    second = execute_runs_parallel(specs, ledger, partial(_touch, out), max_workers=2)
    assert second.executed == 0
    assert second.skipped == 6


# --- failure isolation + resume ---------------------------------------------


def test_worker_failure_does_not_abort_others_and_resume_completes(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "ledger")
    out = tmp_path / "out"
    out.mkdir()
    specs = _specs(3, 2)  # 6
    doomed = specs[2].run_hash()

    first = execute_runs_parallel(
        specs, ledger, partial(_touch_or_fail, out, frozenset({doomed})), max_workers=3
    )
    assert first.failed == 1
    assert first.executed == 5
    assert first.failed_hashes == [doomed]
    assert doomed not in ledger.completed()
    assert len(ledger.completed()) == 5

    # Рестарт без сбоя: готовые пропущены, упавший доведён.
    resume = execute_runs_parallel(specs, ledger, partial(_touch, out), max_workers=3)
    assert resume.skipped == 5
    assert resume.executed == 1
    assert resume.executed_hashes == [doomed]
    assert len(ledger.completed()) == 6


# --- no races under load (EXP-4 concurrency via process pool) ---------------


def test_concurrent_store_writes_no_races(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "ledger")
    root = tmp_path / "store"
    specs = _specs(10, 5)  # 50 ранов

    summary = execute_runs_parallel(specs, ledger, partial(_save, root), max_workers=4)
    assert summary.executed == 50
    assert summary.failed == 0

    store = ResultsStore(root)
    assert store.count() == 50
    for spec in specs:
        rec = store.get_run(spec.run_hash())
        assert rec is not None
        assert rec.strategy == "ma_cross"


# --- speedup ----------------------------------------------------------------


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="нужно >= 2 ядер для ускорения")
def test_parallel_speedup(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "ledger")
    specs = _specs(8, 1)  # 8 ранов по 0.25с
    sleep_s = 0.25
    serial_total = len(specs) * sleep_s  # 2.0с при последовательном исполнении

    t0 = time.perf_counter()
    summary = execute_runs_parallel(specs, ledger, partial(_sleep, sleep_s), max_workers=4)
    elapsed = time.perf_counter() - t0

    assert summary.executed == 8
    # С 4 воркерами идеал ≈ 0.5с; с запасом на накладные требуем заметное ускорение.
    assert elapsed < serial_total * 0.6
