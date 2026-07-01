"""Tests for INF-2 — domain types and experiment config loading."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from trading_research.config import ExperimentConfig, load_experiment_config
from trading_research.domain import TimeSlice, TimeSliceKind

EXAMPLE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "experiments" / "ma_cross_btc_1h_monthly.yaml"
)


def test_example_config_loads() -> None:
    cfg = load_experiment_config(EXAMPLE_CONFIG)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.strategy.name == "ma_cross"
    assert cfg.data.symbols == ["BTCUSDT"]
    assert cfg.params_grid["fast_ma"] == [10, 20, 50]
    assert cfg.execution.leverage == 3


def test_invalid_config_raises_field_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: x\n"
        "data:\n"
        "  exchange: binance\n"
        "  market: futures\n"
        "  symbols: []\n"  # min_length=1 -> ошибка
        "  timeframe: 1h\n"
        "  start: '2021-01-01'\n"
        "  end: '2021-02-01'\n"
        "time_slices: {type: month}\n"
        "strategy: {name: ma_cross}\n"
        "execution:\n"
        "  initial_balance: 10000\n"
        "  commission_rate: 0.0006\n"
        "  slippage_pct: 0.0003\n"
        "  position_size_value: 20\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_experiment_config(bad)


def test_timeslice_rejects_bad_order() -> None:
    with pytest.raises(ValidationError):
        TimeSlice(
            id="s1",
            start=datetime(2024, 2, 1),
            end=datetime(2024, 1, 1),
            label="bad",
            kind=TimeSliceKind.MONTH,
        )


def test_timeslice_roundtrip_warmup() -> None:
    ts = TimeSlice(
        id="2024-01",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 2, 1),
        label="2024-01",
        kind=TimeSliceKind.MONTH,
        warmup_bars=200,
    )
    assert ts.warmup_bars == 200
    assert ts.model_dump()["kind"] == "month"
