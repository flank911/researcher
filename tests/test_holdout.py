"""Tests for WF-3 — untouched holdout / контроль переоптимизации."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from trading_research.backtest.metrics import Metrics
from trading_research.domain import ExecutionModel, FillOn, PositionSizeType
from trading_research.experiments.holdout import (
    HoldoutReport,
    HoldoutSplit,
    assert_research_only,
    build_holdout_report,
    evaluate_holdout,
    save_holdout_report,
)
from trading_research.strategies.base import LONG_SIGNAL, SHORT_SIGNAL

START = datetime(2024, 1, 1, tzinfo=UTC)
HOUR = timedelta(hours=1)
CUTOFF = START + 168 * HOUR  # день 7


class _HoldStrategy:
    name = "hold"

    def warmup_bars(self, params: Mapping[str, Any]) -> int:
        return 0

    def generate_signals(self, data: pl.DataFrame, params: Mapping[str, Any]) -> pl.DataFrame:
        n = data.height
        return data.with_columns(
            pl.Series(LONG_SIGNAL, [i == 0 for i in range(n)]),
            pl.Series(SHORT_SIGNAL, [False] * n),
        )


def _data(n: int = 240) -> pl.DataFrame:  # 10 дней hourly
    return pl.DataFrame(
        {
            "open_time": [START + i * HOUR for i in range(n)],
            "close": [100.0 + i for i in range(n)],
        }
    )


def _metrics(total: float) -> Metrics:
    return Metrics(total, -5.0, 1.0, 1.0, 2.0, 55.0, 0.3, 1.1, 2.0, 7, 8760.0, 30, True)


def _exec() -> ExecutionModel:
    return ExecutionModel(
        initial_balance=10_000.0,
        commission_rate=0.0,
        slippage_pct=0.0,
        leverage=1,
        position_size_type=PositionSizeType.PERCENT_BALANCE,
        position_size_value=100.0,
        signal_lag=0,
        fill_on=FillOn.CURRENT_CLOSE,
    )


# --- split by time tail -----------------------------------------------------


def test_research_excludes_holdout_tail() -> None:
    split = HoldoutSplit(holdout_start=CUTOFF)
    data = _data()
    research = split.research_data(data, symbol="BTCUSDT")
    holdout = split.holdout_data(data, symbol="BTCUSDT")

    assert research["open_time"].max() < CUTOFF
    assert holdout["open_time"].min() >= CUTOFF
    assert research.height + holdout.height == data.height  # разбиение полное и без пересечений
    assert research.height == 168  # дни 0..6


def test_is_holdout_by_time() -> None:
    split = HoldoutSplit(holdout_start=CUTOFF)
    assert split.is_holdout(symbol="BTCUSDT", time=CUTOFF)
    assert split.is_holdout(symbol="BTCUSDT", time=CUTOFF + HOUR)
    assert not split.is_holdout(symbol="BTCUSDT", time=CUTOFF - HOUR)


# --- split by reserved symbols ----------------------------------------------


def test_reserved_symbol_fully_holdout() -> None:
    split = HoldoutSplit(holdout_symbols=frozenset({"ETHUSDT"}))
    data = _data()
    # Зарезервированный символ: research пуст, holdout = всё.
    assert split.research_data(data, symbol="ETHUSDT").is_empty()
    assert split.holdout_data(data, symbol="ETHUSDT").height == data.height
    # Другой символ без временного отсечения: research = всё, holdout пуст.
    assert split.research_data(data, symbol="BTCUSDT").height == data.height
    assert split.holdout_data(data, symbol="BTCUSDT").is_empty()
    assert split.is_holdout(symbol="ETHUSDT", time=START)


# --- code-level guard -------------------------------------------------------


def test_assert_research_only_guard() -> None:
    split = HoldoutSplit(holdout_start=CUTOFF)
    data = _data()
    # Полные данные содержат holdout → страховка срабатывает.
    with pytest.raises(ValueError, match="holdout bar"):
        assert_research_only(data, split, symbol="BTCUSDT")
    # Отфильтрованные research-данные проходят проверку.
    assert_research_only(split.research_data(data, symbol="BTCUSDT"), split, symbol="BTCUSDT")


# --- evaluate + report ------------------------------------------------------


def test_evaluate_holdout_returns_metrics() -> None:
    split = HoldoutSplit(holdout_start=CUTOFF)
    holdout = split.holdout_data(_data(), symbol="BTCUSDT")
    metrics = evaluate_holdout(
        holdout, _HoldStrategy(), {"side": "long"}, _exec(), "1h", min_trades=0
    )
    assert isinstance(metrics, Metrics)
    assert metrics.total_return_pct > 0  # цена растёт, long в плюсе


def test_evaluate_holdout_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluate_holdout(_data().clear(), _HoldStrategy(), {}, _exec(), "1h")


def test_build_holdout_report_degradation() -> None:
    report = build_holdout_report(
        symbol="BTCUSDT",
        params={"fast_ma": 5},
        research_value=50.0,
        holdout_metrics=_metrics(10.0),
    )
    assert report.holdout_value == pytest.approx(10.0)
    assert report.research_value == pytest.approx(50.0)
    assert report.degradation == pytest.approx(40.0)  # research - holdout
    assert report.metric == "total_return_pct"


def test_save_holdout_report_json(tmp_path: Path) -> None:
    report = build_holdout_report(
        symbol="BTCUSDT", params={"fast_ma": 5}, research_value=50.0, holdout_metrics=_metrics(10.0)
    )
    path = save_holdout_report(report, tmp_path / "sub" / "holdout.json")
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["symbol"] == "BTCUSDT"
    assert loaded["degradation"] == pytest.approx(40.0)
    assert loaded["holdout_metrics"]["total_return_pct"] == pytest.approx(10.0)


# --- validation -------------------------------------------------------------


def test_empty_split_raises() -> None:
    with pytest.raises(ValueError, match="holdout_start"):
        HoldoutSplit()


def test_report_dataclass_shape() -> None:
    report = build_holdout_report(
        symbol="X", params={}, research_value=1.0, holdout_metrics=_metrics(1.0)
    )
    assert isinstance(report, HoldoutReport)
