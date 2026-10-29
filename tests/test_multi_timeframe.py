"""Tests for RUN-6 — мультитаймфрейм в конфиге и прогоне."""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from trading_research.analytics.queries import (
    load_performance_table,
    params_by_symbol,
    params_by_timeframe,
)
from trading_research.cli.build_features import build_features_from_config
from trading_research.cli.run_experiment import (
    run_experiment_from_config,
    slices_by_timeframe,
    union_slices,
)
from trading_research.cli.walk_forward import walk_forward_from_config, wf_dir
from trading_research.config import load_experiment_config
from trading_research.data.schema import CANDLE_SCHEMA, timeframe_to_ms
from trading_research.data.storage.parquet_store import ParquetStore
from trading_research.domain import SCHEMA_VERSION
from trading_research.experiments.results_store import ResultsStore

T0 = datetime(2024, 1, 1, tzinfo=UTC)
EXP = "mtf_exp"


def _candles(n: int, timeframe: str) -> pl.DataFrame:
    step = timedelta(milliseconds=timeframe_to_ms(timeframe))
    times = [T0 + step * i for i in range(n)]
    close = [100.0 + 10.0 * ((i // 12) % 2) + 0.01 * i for i in range(n)]
    return pl.DataFrame(
        {
            "open_time": times,
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [10.0 + (i % 5) for i in range(n)],
            "close_time": [t + step for t in times],
            "quote_volume": [1000.0] * n,
            "trades": [5] * n,
            "taker_buy_base": [4.0] * n,
            "taker_buy_quote": [400.0] * n,
        },
        schema=CANDLE_SCHEMA,
    )


def _seed(
    data_root: Path,
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
    timeframes: tuple[str, ...] = ("1h", "4h"),
    days: int = 120,
) -> None:
    """Засеять одинаковый период в каждом таймфрейме (разное число баров)."""
    store = ParquetStore(data_root)
    for sym in symbols:
        for tf in timeframes:
            bars = days * 24 * 3_600_000 // timeframe_to_ms(tf)
            store.write_candles(
                _candles(bars, tf),
                exchange="binance",
                market="futures",
                symbol=sym,
                timeframe=tf,
            )


def _config(
    path: Path,
    *,
    symbols: str = "[BTCUSDT, ETHUSDT]",
    timeframes: str = "[1h, 4h]",
    min_bars: int = 50,
    version: int = SCHEMA_VERSION,
    extra: str = "",
) -> Path:
    path.write_text(
        textwrap.dedent(
            f"""
            schema_version: {version}
            name: {EXP}
            data:
              exchange: binance
              market: futures
              symbols: {symbols}
              timeframes: {timeframes}
              start: "2024-01-01"
              end: "2024-04-01"
            time_slices:
              type: month
              min_bars: {min_bars}
            strategy:
              name: ma_cross
            params_grid:
              fast_ma: [5]
              slow_ma: [20]
            execution:
              initial_balance: 10000
              commission_rate: 0.0006
              slippage_pct: 0.0003
              leverage: 3
              position_size_type: percent_balance
              position_size_value: 20
              signal_lag: 1
              fill_on: next_open
            """
        ).lstrip()
        + extra,
        encoding="utf-8",
    )
    return path


# --- конфиг ----------------------------------------------------------------


def test_timeframes_is_a_list(tmp_path: Path) -> None:
    cfg = load_experiment_config(_config(tmp_path / "e.yaml"))
    assert cfg.data.timeframes == ["1h", "4h"]


def test_empty_timeframes_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        load_experiment_config(_config(tmp_path / "e.yaml", timeframes="[]"))


def test_scalar_timeframe_rejected_loudly(tmp_path: Path) -> None:
    """v1-поле ``timeframe`` не превращается в список молча — это явная ошибка."""
    path = tmp_path / "e.yaml"
    _config(path)
    path.write_text(
        path.read_text(encoding="utf-8").replace("timeframes: [1h, 4h]", "timeframe: 1h"),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError) as exc:
        load_experiment_config(path)
    msg = str(exc.value)
    assert "timeframe" in msg
    # И об отсутствии нового поля, и о лишнем старом.
    assert "timeframes" in msg


def test_old_schema_version_rejected_with_hint(tmp_path: Path) -> None:
    """Версия формата теперь проверяется, а не просто хранится."""
    with pytest.raises(ValidationError, match="unsupported schema_version 1"):
        load_experiment_config(_config(tmp_path / "e.yaml", version=1))


# --- слайсы на таймфрейм ---------------------------------------------------


def test_slices_generated_per_timeframe(tmp_path: Path) -> None:
    cfg = load_experiment_config(_config(tmp_path / "e.yaml"))
    by_tf = slices_by_timeframe(cfg)
    assert set(by_tf) == {"1h", "4h"}
    assert all(by_tf.values())


def test_min_bars_filters_timeframes_differently(tmp_path: Path) -> None:
    """Главная причина считать окна на каждый таймфрейм отдельно.

    ``min_bars`` — порог в барах. Месяц на 1h даёт ~720 баров, на 4h ~180.
    При пороге между этими числами окна проходят фильтр на 1h и не проходят на 4h,
    поэтому единый список окон «на конфиг» дал бы неверную нарезку для части
    таймфреймов.
    """
    cfg = load_experiment_config(_config(tmp_path / "e.yaml", min_bars=300))
    by_tf = slices_by_timeframe(cfg)
    assert by_tf["1h"], "месяц на 1h должен проходить порог 300 баров"
    assert not by_tf["4h"], "месяц на 4h (~180 баров) не должен проходить порог 300"


def test_union_slices_dedupes_by_id(tmp_path: Path) -> None:
    cfg = load_experiment_config(_config(tmp_path / "e.yaml"))
    by_tf = slices_by_timeframe(cfg)
    merged = union_slices(by_tf)
    ids = [s.id for s in merged]
    assert len(ids) == len(set(ids)), "объединение не должно содержать дублей"
    assert set(ids) == {s.id for slices in by_tf.values() for s in slices}


# --- прогон по парам -------------------------------------------------------


def test_run_covers_all_symbol_timeframe_pairs(tmp_path: Path) -> None:
    data_root, results_root = tmp_path / "raw", tmp_path / "res"
    _seed(data_root)
    cfg = _config(tmp_path / "e.yaml")

    table = pl.read_parquet(
        run_experiment_from_config(cfg, data_root=data_root, results_root=results_root, workers=1)
    )
    pairs = set(zip(table["symbol"].to_list(), table["timeframe"].to_list(), strict=True))
    assert pairs == {
        ("BTCUSDT", "1h"),
        ("BTCUSDT", "4h"),
        ("ETHUSDT", "1h"),
        ("ETHUSDT", "4h"),
    }


def test_timeframes_do_not_collide_in_run_hash(tmp_path: Path) -> None:
    """Раны 1h и 4h различаются, хотя ``time_slice_id`` у них одинаковый."""
    data_root, results_root = tmp_path / "raw", tmp_path / "res"
    _seed(data_root, symbols=("BTCUSDT",))
    cfg = _config(tmp_path / "e.yaml", symbols="[BTCUSDT]")
    run_experiment_from_config(cfg, data_root=data_root, results_root=results_root, workers=1)

    runs = ResultsStore(results_root / EXP).all_runs()
    # Один и тот же slice_id встречается в обоих таймфреймах...
    shared = {r.slice_id for r in runs if r.timeframe == "1h"} & {
        r.slice_id for r in runs if r.timeframe == "4h"
    }
    assert shared, "предпосылка теста: slice_id не кодирует таймфрейм"
    # ...но run_hash у них разные, то есть коллизии нет.
    hashes = [r.run_hash for r in runs]
    assert len(hashes) == len(set(hashes))


def test_an1_slices_by_symbol_and_timeframe_non_empty(tmp_path: Path) -> None:
    """Срезы AN-1/AN-2 наполняются одним прогоном, а не склейкой нескольких."""
    data_root, results_root = tmp_path / "raw", tmp_path / "res"
    _seed(data_root)
    cfg = _config(tmp_path / "e.yaml")
    out = run_experiment_from_config(cfg, data_root=data_root, results_root=results_root, workers=1)
    table = load_performance_table(out)

    by_symbol = params_by_symbol(table)
    by_tf = params_by_timeframe(table)
    assert by_symbol.height > 0
    assert set(by_symbol["axis_value"].to_list()) == {"BTCUSDT", "ETHUSDT"}
    assert by_tf.height > 0
    assert set(by_tf["axis_value"].to_list()) == {"1h", "4h"}


# --- фичи и walk-forward по парам -----------------------------------------


def test_features_cover_all_pairs(tmp_path: Path) -> None:
    data_root, features_root = tmp_path / "raw", tmp_path / "feat"
    _seed(data_root)
    cfg = _config(tmp_path / "e.yaml")

    table = pl.read_parquet(
        build_features_from_config(cfg, data_root=data_root, features_root=features_root)
    )
    pairs = set(zip(table["symbol"].to_list(), table["timeframe"].to_list(), strict=True))
    assert pairs == {
        ("BTCUSDT", "1h"),
        ("BTCUSDT", "4h"),
        ("ETHUSDT", "1h"),
        ("ETHUSDT", "4h"),
    }


def test_walk_forward_keys_and_dirs_include_timeframe(tmp_path: Path) -> None:
    """Без таймфрейма в пути прогоны 1h и 4h перезаписывали бы друг друга."""
    data_root, results_root = tmp_path / "raw", tmp_path / "res"
    _seed(data_root, symbols=("BTCUSDT",))
    cfg = _config(
        tmp_path / "e.yaml",
        symbols="[BTCUSDT]",
        extra="walk_forward:\n  train_days: 30\n  test_days: 10\n  step_days: 10\n",
    )

    results = walk_forward_from_config(cfg, data_root=data_root, results_root=results_root)
    assert set(results) == {("BTCUSDT", "1h"), ("BTCUSDT", "4h")}
    for tf in ("1h", "4h"):
        assert (wf_dir(results_root, EXP, "BTCUSDT", tf) / "oos_equity.parquet").exists()
    # Пути различны — артефакты не затирают друг друга.
    assert wf_dir(results_root, EXP, "BTCUSDT", "1h") != wf_dir(results_root, EXP, "BTCUSDT", "4h")
