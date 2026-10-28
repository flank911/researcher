"""Tests for WF-1 — walk-forward train/test фолды."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trading_research.domain import TimeSliceRole
from trading_research.experiments.time_splitters import (
    generate_walk_forward_folds,
)

START = datetime(2024, 1, 1, tzinfo=UTC)
DAY = timedelta(days=1)


def _folds(**kw: object) -> list:
    base: dict[str, object] = {
        "start": START,
        "end": START + 7 * DAY,
        "train": 2 * DAY,
        "test": 1 * DAY,
        "step": 1 * DAY,
    }
    base.update(kw)
    return generate_walk_forward_folds(**base)  # type: ignore[arg-type]


# --- rolling ----------------------------------------------------------------


def test_rolling_folds_match_example() -> None:
    folds = _folds()  # train=2d, test=1d, step=1d в [01-01, 01-08)
    assert len(folds) == 5
    assert [f.fold for f in folds] == [0, 1, 2, 3, 4]

    first = folds[0]
    assert first.train.start == START
    assert first.train.end == START + 2 * DAY
    assert first.test.start == START + 2 * DAY
    assert first.test.end == START + 3 * DAY

    # Последний фолд: test заканчивается ровно на границе окна.
    assert folds[-1].test.end == START + 7 * DAY


def test_rolling_window_is_fixed_length_and_steps() -> None:
    folds = _folds()
    for f in folds:
        assert f.train.end - f.train.start == 2 * DAY
        assert f.test.end - f.test.start == 1 * DAY
    # Шаг сдвига train = step.
    starts = [f.train.start for f in folds]
    assert all(b - a == DAY for a, b in zip(starts, starts[1:], strict=False))


# --- no leakage: test strictly after train ----------------------------------


def test_test_never_overlaps_train() -> None:
    folds = _folds(end=START + 30 * DAY, train=5 * DAY, test=3 * DAY, step=2 * DAY)
    assert folds  # непустой
    for f in folds:
        assert f.test.start == f.train.end  # стык без перекрытия
        assert f.test.start >= f.train.end
        assert f.train.end <= f.test.start < f.test.end


def test_all_folds_within_range() -> None:
    folds = _folds(end=START + 30 * DAY, train=5 * DAY, test=3 * DAY, step=2 * DAY)
    for f in folds:
        assert f.train.start >= START
        assert f.test.end <= START + 30 * DAY


def test_roles_and_fold_link() -> None:
    folds = _folds()
    for f in folds:
        assert f.train.role is TimeSliceRole.TRAIN
        assert f.test.role is TimeSliceRole.TEST
        assert f.train.id == f"fold{f.fold:02d}-train"
        assert f.test.id == f"fold{f.fold:02d}-test"


def test_warmup_assigned() -> None:
    folds = _folds(warmup_bars=50)
    assert folds[0].train.warmup_bars == 50
    assert folds[0].test.warmup_bars == 50


# --- anchored (expanding) ---------------------------------------------------


def test_anchored_train_expands_from_start() -> None:
    folds = _folds(end=START + 6 * DAY, anchored=True)  # train=2d,test=1d,step=1d
    assert len(folds) == 4
    assert all(f.train.start == START for f in folds)  # train всегда от start
    lengths = [f.train.end - f.train.start for f in folds]
    assert lengths == [2 * DAY, 3 * DAY, 4 * DAY, 5 * DAY]  # растёт на step
    # test по-прежнему строго после train.
    for f in folds:
        assert f.test.start == f.train.end


# --- edges / validation -----------------------------------------------------


def test_empty_when_range_too_short() -> None:
    assert _folds(end=START + 2 * DAY) == []  # train(2d)+test(1d) не влезают


def test_invalid_args_raise() -> None:
    with pytest.raises(ValueError, match="end must be after start"):
        _folds(end=START)
    with pytest.raises(ValueError, match="train"):
        _folds(train=timedelta(0))
    with pytest.raises(ValueError, match="test"):
        _folds(test=-1 * DAY)
    with pytest.raises(ValueError, match="step"):
        _folds(step=timedelta(0))
