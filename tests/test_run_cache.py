"""Tests for EXP-3 — run_hash, dedupe, resume."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trading_research.domain import ExecutionModel, PositionSizeType, TimeSlice
from trading_research.experiments.cache import (
    RunLedger,
    RunSpec,
    build_run_specs,
    execute_runs,
)


def _execution(**overrides: object) -> ExecutionModel:
    base: dict[str, object] = {
        "initial_balance": 10_000.0,
        "commission_rate": 0.0006,
        "slippage_pct": 0.0003,
        "leverage": 3,
        "position_size_type": PositionSizeType.PERCENT_BALANCE,
        "position_size_value": 20.0,
    }
    base.update(overrides)
    return ExecutionModel(**base)  # type: ignore[arg-type]


def _spec(**overrides: object) -> RunSpec:
    base: dict[str, object] = {
        "strategy": "ma_cross",
        "params": {"fast_ma": 20, "slow_ma": 100},
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "slice_id": "2024-01",
        "dataset_hash": "abc123",
        "execution": _execution(),
        "git_commit": "deadbeef",
    }
    base.update(overrides)
    return RunSpec(**base)  # type: ignore[arg-type]


def _slice(sid: str) -> TimeSlice:
    return TimeSlice(
        id=sid,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 2, 1, tzinfo=UTC),
        label=sid,
    )


# --- run_hash determinism ---------------------------------------------------


def test_identical_inputs_same_hash() -> None:
    assert _spec().run_hash() == _spec().run_hash()


def test_param_key_order_irrelevant() -> None:
    a = _spec(params={"fast_ma": 20, "slow_ma": 100})
    b = _spec(params={"slow_ma": 100, "fast_ma": 20})
    assert a.run_hash() == b.run_hash()


@pytest.mark.parametrize(
    "override",
    [
        {"strategy": "other"},
        {"params": {"fast_ma": 21, "slow_ma": 100}},
        {"symbol": "ETHUSDT"},
        {"timeframe": "4h"},
        {"slice_id": "2024-02"},
        {"dataset_hash": "xyz789"},
        {"execution": _execution(commission_rate=0.001)},
        {"git_commit": "cafebabe"},
        {"git_commit": None},
    ],
)
def test_any_input_change_changes_hash(override: dict[str, object]) -> None:
    assert _spec().run_hash() != _spec(**override).run_hash()


# --- build_run_specs --------------------------------------------------------


def test_build_run_specs_is_cartesian_and_deterministic() -> None:
    combos = [{"fast_ma": 10, "slow_ma": 50}, {"fast_ma": 20, "slow_ma": 100}]
    slices = [_slice("2024-01"), _slice("2024-02"), _slice("2024-03")]
    kwargs = {
        "strategy": "ma_cross",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "dataset_hash": "abc",
        "execution": _execution(),
        "param_combos": combos,
        "slices": slices,
    }
    specs_a = build_run_specs(**kwargs)  # type: ignore[arg-type]
    specs_b = build_run_specs(**kwargs)  # type: ignore[arg-type]
    assert len(specs_a) == len(combos) * len(slices)
    assert [s.run_hash() for s in specs_a] == [s.run_hash() for s in specs_b]
    assert len({s.run_hash() for s in specs_a}) == len(specs_a)  # все уникальны


# --- ledger + dedupe --------------------------------------------------------


def test_ledger_marks_and_reports_done(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path)
    h = _spec().run_hash()
    assert ledger.is_done(h) is False
    ledger.mark_done(h)
    assert ledger.is_done(h) is True
    assert ledger.completed() == {h}


def test_rerun_skips_completed_runs(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path)
    specs = build_run_specs(
        strategy="ma_cross",
        symbol="BTCUSDT",
        timeframe="1h",
        dataset_hash="abc",
        execution=_execution(),
        param_combos=[{"fast_ma": 10, "slow_ma": 50}, {"fast_ma": 20, "slow_ma": 100}],
        slices=[_slice("2024-01"), _slice("2024-02")],
    )

    first = execute_runs(specs, ledger, lambda _spec: None)
    assert first.executed == len(specs)
    assert first.skipped == 0

    # Повторный запуск: ничего не пересчитывается.
    processed: list[str] = []
    second = execute_runs(specs, ledger, lambda s: processed.append(s.run_hash()))
    assert second.executed == 0
    assert second.skipped == len(specs)
    assert processed == []


# --- resume after crash -----------------------------------------------------


def test_resume_continues_after_crash(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path)
    specs = build_run_specs(
        strategy="ma_cross",
        symbol="BTCUSDT",
        timeframe="1h",
        dataset_hash="abc",
        execution=_execution(),
        param_combos=[{"fast_ma": 10, "slow_ma": 50}, {"fast_ma": 20, "slow_ma": 100}],
        slices=[_slice("2024-01"), _slice("2024-02"), _slice("2024-03")],
    )
    assert len(specs) == 6

    # Первый прогон падает на 3-й задаче (после того как 1-2 помечены done).
    processed_before: list[str] = []

    def flaky(spec: RunSpec) -> None:
        processed_before.append(spec.run_hash())
        if len(processed_before) == 3:
            raise RuntimeError("process killed")

    with pytest.raises(RuntimeError, match="process killed"):
        execute_runs(specs, ledger, flaky)
    assert len(ledger.completed()) == 2  # 3-я не помечена — упала до mark_done

    # Рестарт: продолжаем с места остановки, готовые не трогаем.
    processed_after: list[str] = []
    summary = execute_runs(specs, ledger, lambda s: processed_after.append(s.run_hash()))

    assert summary.skipped == 2
    assert summary.executed == 4
    assert len(processed_after) == 4
    # Упавшая ран (3-я) повторяется при рестарте.
    assert processed_before[2] in processed_after
    # Все 6 ранов в итоге завершены ровно по разу.
    assert len(ledger.completed()) == 6


def test_mark_done_leaves_no_tmp_files(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path)
    ledger.mark_done(_spec().run_hash())
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*")) == []
