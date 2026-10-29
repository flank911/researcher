"""Tests for EXP-5 — параллельное исполнение прогонов."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import polars as pl
import pytest

from trading_research.domain import ExecutionModel, PositionSizeType, TimeSlice
from trading_research.experiments.cache import RunLedger, RunSpec, build_run_specs
from trading_research.experiments.parallel import (
    _MP_CONTEXT,
    execute_run_groups_parallel,
    execute_runs_parallel,
)
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


def _touch_group(out_dir: Path, specs: Sequence[RunSpec]) -> None:
    """Один «прогон» на группу: отмечает все её раны (как CONTINUOUS)."""
    (out_dir / f"group-{specs[0].run_hash()}").write_text(str(len(specs)), encoding="utf-8")
    for spec in specs:
        _touch(out_dir, spec)


def _fail_group(out_dir: Path, fail_first: frozenset[str], specs: Sequence[RunSpec]) -> None:
    if specs[0].run_hash() in fail_first:
        raise RuntimeError("group boom")
    _touch_group(out_dir, specs)


def _by_params(specs: list[RunSpec]) -> list[list[RunSpec]]:
    groups: dict[str, list[RunSpec]] = {}
    for spec in specs:
        groups.setdefault(str(sorted(spec.params.items())), []).append(spec)
    return list(groups.values())


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


# --- группы (RUN-1): гранулярность CONTINUOUS -------------------------------


def test_groups_run_once_per_combination(tmp_path: Path) -> None:
    """Группа считается одним вызовом, но помечает все свои раны."""
    ledger = RunLedger(tmp_path / "ledger")
    out = tmp_path / "out"
    out.mkdir()
    specs = _specs(3, 4)  # 3 комбинации × 4 слайса = 12 ранов
    groups = _by_params(specs)
    assert len(groups) == 3

    summary = execute_run_groups_parallel(groups, ledger, partial(_touch_group, out), max_workers=2)
    assert summary.total == 12
    assert summary.executed == 12  # счёт в ранах, не в группах
    assert summary.skipped == 0
    # Но вызовов воркера было ровно 3 — по одному на комбинацию.
    assert len(list(out.glob("group-*"))) == 3
    assert ledger.completed() == {s.run_hash() for s in specs}

    second = execute_run_groups_parallel(groups, ledger, partial(_touch_group, out), max_workers=2)
    assert second.executed == 0
    assert second.skipped == 12


def test_partial_group_is_recomputed(tmp_path: Path) -> None:
    """Частично готовая группа считается заново — слайс нельзя досчитать отдельно."""
    ledger = RunLedger(tmp_path / "ledger")
    out = tmp_path / "out"
    out.mkdir()
    specs = _specs(2, 3)  # 2 группы по 3 рана
    groups = _by_params(specs)

    # Помечаем готовым только один ран первой группы.
    ledger.mark_done(groups[0][0].run_hash())

    summary = execute_run_groups_parallel(groups, ledger, partial(_touch_group, out), max_workers=1)
    # Ни одна группа не пропущена: в первой готов лишь один ран, вторая не начата.
    assert summary.skipped == 0
    assert len(list(out.glob("group-*"))) == 2


def test_group_failure_isolated_and_resumable(tmp_path: Path) -> None:
    """Падение группы не рушит остальные и не помечает её раны готовыми."""
    ledger = RunLedger(tmp_path / "ledger")
    out = tmp_path / "out"
    out.mkdir()
    specs = _specs(3, 2)
    groups = _by_params(specs)
    doomed = groups[1]
    doomed_hashes = {s.run_hash() for s in doomed}

    first = execute_run_groups_parallel(
        groups,
        ledger,
        partial(_fail_group, out, frozenset({doomed[0].run_hash()})),
        max_workers=2,
    )
    assert first.failed == len(doomed)
    assert set(first.failed_hashes) == doomed_hashes
    assert not (doomed_hashes & ledger.completed())

    resume = execute_run_groups_parallel(groups, ledger, partial(_touch_group, out), max_workers=2)
    assert resume.executed == len(doomed)
    assert set(resume.executed_hashes) == doomed_hashes
    assert ledger.completed() == {s.run_hash() for s in specs}


def test_workers_one_runs_inline_same_semantics(tmp_path: Path) -> None:
    """``max_workers=1`` исполняет без пула, результат тот же."""
    ledger = RunLedger(tmp_path / "ledger")
    out = tmp_path / "out"
    out.mkdir()
    specs = _specs(2, 2)
    groups = _by_params(specs)

    summary = execute_run_groups_parallel(groups, ledger, partial(_touch_group, out), max_workers=1)
    assert summary.executed == 4
    assert summary.failed == 0
    assert ledger.completed() == {s.run_hash() for s in specs}


# --- старт процессов --------------------------------------------------------


def test_pool_never_forks() -> None:
    """Пул не должен наследовать состояние родителя через fork.

    На Linux fork — дефолт, и форк процесса, где уже работал polars, вешает
    ребёнка навсегда (пул потоков rayon не переносится). Локально на macOS
    дефолт spawn, поэтому регрессию видно только в CI — ловим её здесь.
    """
    assert _MP_CONTEXT.get_start_method() == "spawn"


# --- speedup ----------------------------------------------------------------


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="нужно >= 2 ядер для ускорения")
def test_parallel_speedup(tmp_path: Path) -> None:
    """4 воркера заметно быстрее одного на той же работе.

    Сравниваем с *измеренным* одним воркером, а не с арифметикой: пул spawn'ится
    и в том, и в другом прогоне, так что старт процессов (импорт polars в ребёнке)
    вычитается сам и не делает тест флапающим на холодном раннере.
    """
    specs = _specs(8, 1)  # 8 ранов по 0.25с
    sleep_s = 0.25
    compute = partial(_sleep, sleep_s)

    t0 = time.perf_counter()
    serial = execute_runs_parallel(specs, RunLedger(tmp_path / "ledger-1"), compute, max_workers=1)
    serial_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    parallel = execute_runs_parallel(
        specs, RunLedger(tmp_path / "ledger-4"), compute, max_workers=4
    )
    parallel_elapsed = time.perf_counter() - t0

    assert serial.executed == parallel.executed == 8
    # Идеал — вчетверо; с запасом на накладные требуем заметное ускорение.
    assert parallel_elapsed < serial_elapsed * 0.6
