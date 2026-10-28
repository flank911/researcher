"""Tests for AN-1 — parameter_performance_by_slice table + срезы."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from trading_research.analytics.queries import (
    PERFORMANCE_COLUMNS,
    PERFORMANCE_SCHEMA,
    REGIME_SLOTS,
    PerformanceRecord,
    build_parameter_performance,
    load_performance_table,
    param_hash,
    params_by_symbol,
    params_by_time,
    params_by_timeframe,
    performance_from_store,
    rank_params,
    save_performance_table,
    slice_param_performance,
)
from trading_research.backtest.metrics import Metrics
from trading_research.domain import ExecutionModel, PositionSizeType, TimeSlice, TimeSliceKind
from trading_research.experiments.cache import build_run_specs
from trading_research.experiments.results_store import ResultsStore


def _metrics(total: float = 10.0) -> Metrics:
    return Metrics(
        total_return_pct=total,
        max_drawdown_pct=-5.0,
        sharpe=1.2,
        sortino=1.5,
        profit_factor=2.0,
        win_rate=55.0,
        avg_trade_pct=0.3,
        expectancy=1.1,
        recovery_factor=2.0,
        trades_count=7,
        annualization_factor=8760.0,
        min_trades=30,
        reliable=False,
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


def _record(
    params: dict[str, Any],
    sid: str,
    *,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    total: float = 10.0,
) -> PerformanceRecord:
    return PerformanceRecord(
        experiment_id="exp1",
        strategy_name="ma_cross",
        symbol=symbol,
        timeframe=timeframe,
        time_slice=_slice(sid),
        params=params,
        metrics=_metrics(total),
    )


P1 = {"fast_ma": 5, "slow_ma": 20}
P2 = {"fast_ma": 10, "slow_ma": 50}


# --- schema -----------------------------------------------------------------


def test_schema_conformance() -> None:
    table = build_parameter_performance([_record(P1, "2024-01")])
    assert tuple(table.columns) == PERFORMANCE_COLUMNS
    assert dict(table.schema) == PERFORMANCE_SCHEMA


def test_empty_records_give_empty_table_with_schema() -> None:
    table = build_parameter_performance([])
    assert table.height == 0
    assert dict(table.schema) == PERFORMANCE_SCHEMA


def test_regime_slots_present_and_null() -> None:
    table = build_parameter_performance([_record(P1, "2024-01")])
    for slot in REGIME_SLOTS:
        assert slot in table.columns
        assert table[slot].null_count() == table.height


# --- identity ---------------------------------------------------------------


def test_param_hash_order_independent() -> None:
    assert param_hash({"fast_ma": 5, "slow_ma": 20}) == param_hash({"slow_ma": 20, "fast_ma": 5})
    assert param_hash(P1) != param_hash(P2)


def test_params_json_and_hash_columns() -> None:
    table = build_parameter_performance([_record(P1, "2024-01"), _record(P2, "2024-01")])
    assert table["param_hash"].n_unique() == 2
    assert '"fast_ma"' in table["params_json"][0]


# --- build from store -------------------------------------------------------


def _execution() -> ExecutionModel:
    return ExecutionModel(
        initial_balance=10_000.0,
        commission_rate=0.0006,
        slippage_pct=0.0003,
        leverage=3,
        position_size_type=PositionSizeType.PERCENT_BALANCE,
        position_size_value=20.0,
    )


def test_performance_from_store(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "store")
    slices = [_slice("2024-01"), _slice("2024-02")]
    specs = build_run_specs(
        strategy="ma_cross",
        symbol="BTCUSDT",
        timeframe="1h",
        dataset_hash="abc",
        execution=_execution(),
        param_combos=[P1, P2],
        slices=slices,
    )
    equity = pl.DataFrame({"open_time": [datetime(2024, 1, 1, tzinfo=UTC)], "equity": [10_000.0]})
    for spec in specs:
        store.save_run(spec, _metrics(), [], equity)

    table = performance_from_store(store, slices, experiment_id="exp1")
    assert table.height == 4  # 2 params × 2 slices
    assert table["param_hash"].n_unique() == 2
    assert set(table["time_slice_id"].to_list()) == {"2024-01", "2024-02"}
    assert table["experiment_id"].to_list() == ["exp1"] * 4


def test_performance_from_store_unknown_slice_raises(tmp_path: Path) -> None:
    store = ResultsStore(tmp_path / "store")
    specs = build_run_specs(
        strategy="ma_cross",
        symbol="BTCUSDT",
        timeframe="1h",
        dataset_hash="abc",
        execution=_execution(),
        param_combos=[P1],
        slices=[_slice("2024-01")],
    )
    equity = pl.DataFrame({"open_time": [datetime(2024, 1, 1, tzinfo=UTC)], "equity": [10_000.0]})
    store.save_run(specs[0], _metrics(), [], equity)
    with pytest.raises(KeyError, match="slice_id"):
        performance_from_store(store, [_slice("2024-02")], experiment_id="exp1")


# --- slices params ↔ axis ---------------------------------------------------


def _multi_table() -> pl.DataFrame:
    records = [
        _record(P1, "2024-01", symbol="BTCUSDT", timeframe="1h", total=10.0),
        _record(P1, "2024-02", symbol="BTCUSDT", timeframe="1h", total=20.0),
        _record(P2, "2024-01", symbol="BTCUSDT", timeframe="1h", total=5.0),
        _record(P1, "2024-01", symbol="ETHUSDT", timeframe="4h", total=30.0),
    ]
    return build_parameter_performance(records)


def test_params_by_time() -> None:
    out = params_by_time(_multi_table())
    # P1 присутствует на 2024-01 (BTC+ETH) и 2024-02; группировка по (param_hash × slice).
    assert set(out.columns) >= {"param_hash", "axis_value", "total_return_pct_avg", "n"}
    p1h = param_hash(P1)
    jan = out.filter((pl.col("param_hash") == p1h) & (pl.col("axis_value") == "2024-01"))
    assert jan.height == 1
    assert jan["total_return_pct_avg"][0] == pytest.approx(20.0)  # (10 BTC + 30 ETH)/2
    assert jan["n"][0] == 2


def test_params_by_symbol() -> None:
    out = params_by_symbol(_multi_table())
    assert set(out["axis_value"].to_list()) == {"BTCUSDT", "ETHUSDT"}


def test_params_by_timeframe() -> None:
    out = params_by_timeframe(_multi_table())
    assert set(out["axis_value"].to_list()) == {"1h", "4h"}


def test_slice_with_custom_metric() -> None:
    out = slice_param_performance(_multi_table(), axis="symbol", metric="sharpe")
    assert "sharpe_avg" in out.columns


def test_invalid_axis_and_metric_raise() -> None:
    table = _multi_table()
    with pytest.raises(ValueError, match="axis"):
        slice_param_performance(table, axis="nope")
    with pytest.raises(ValueError, match="metric"):
        slice_param_performance(table, axis="time", metric="reliable")


# --- AN-2: ranking ----------------------------------------------------------


def _ranking_table() -> pl.DataFrame:
    # P1: 10 и 20 (BTC), P2: 5 (BTC), P1: 30 (ETH/4h).
    records = [
        _record(P1, "2024-01", symbol="BTCUSDT", timeframe="1h", total=10.0),
        _record(P1, "2024-02", symbol="BTCUSDT", timeframe="1h", total=20.0),
        _record(P2, "2024-01", symbol="BTCUSDT", timeframe="1h", total=5.0),
        _record(P1, "2024-01", symbol="ETHUSDT", timeframe="4h", total=30.0),
    ]
    return build_parameter_performance(records)


def test_rank_params_global_orders_by_metric() -> None:
    ranked = rank_params(_ranking_table(), metric="total_return_pct", top=20)
    assert ranked["rank"].to_list() == [1, 2]  # 2 комбинации параметров
    # P1 среднее = (10+20+30)/3 = 20 > P2 = 5 → P1 первый.
    assert ranked["param_hash"][0] == param_hash(P1)
    assert ranked["total_return_pct"][0] == pytest.approx(20.0)


def test_rank_params_top_limits_rows() -> None:
    ranked = rank_params(_ranking_table(), top=1)
    assert ranked.height == 1
    assert ranked["param_hash"][0] == param_hash(P1)


def test_rank_params_filter_by_symbol() -> None:
    ranked = rank_params(_ranking_table(), symbol="BTCUSDT")
    # ETH-строка отфильтрована → P1 среднее = (10+20)/2 = 15.
    p1 = ranked.filter(pl.col("param_hash") == param_hash(P1))
    assert p1["total_return_pct"][0] == pytest.approx(15.0)


def test_rank_params_by_slice_ranks_within_group() -> None:
    ranked = rank_params(_ranking_table(), by="time", top=1)
    # По одному лучшему на слайс.
    assert set(ranked["time_slice_id"].to_list()) == {"2024-01", "2024-02"}
    assert ranked.filter(pl.col("rank") > 1).height == 0
    # На 2024-01 лучший — P1 (ETH 30 даёт среднее 20 > P2 5).
    jan = ranked.filter(pl.col("time_slice_id") == "2024-01")
    assert jan["param_hash"][0] == param_hash(P1)


def test_rank_params_ascending() -> None:
    ranked = rank_params(_ranking_table(), descending=False)
    assert ranked["param_hash"][0] == param_hash(P2)  # худший первым


def test_rank_params_invalid_args() -> None:
    table = _ranking_table()
    with pytest.raises(ValueError, match="metric"):
        rank_params(table, metric="reliable")
    with pytest.raises(ValueError, match="axis"):
        rank_params(table, by="nope")
    with pytest.raises(ValueError, match="top"):
        rank_params(table, top=0)


def test_save_load_performance_table_roundtrip(tmp_path: Path) -> None:
    table = _multi_table()
    path = save_performance_table(table, tmp_path / "sub" / "perf.parquet")
    assert path.exists()
    assert load_performance_table(path).equals(table)
