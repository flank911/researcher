"""Tests for RUN-3 — CLI построения таблицы ``time_slice_features``."""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from trading_research.cli.build_features import (
    FEATURES_FILE,
    build_features_from_config,
    features_path,
    read_series_or_none,
)
from trading_research.cli.main import app
from trading_research.cli.run_experiment import run_experiment_from_config
from trading_research.data.schema import (
    CANDLE_SCHEMA,
    FUNDING_SCHEMA,
    LONG_SHORT_RATIO_SCHEMA,
    OPEN_INTEREST_SCHEMA,
)
from trading_research.data.storage.parquet_store import (
    FUNDING,
    LONG_SHORT_RATIO,
    OPEN_INTEREST,
    ParquetStore,
)
from trading_research.features.market_regime import TIME_SLICE_FEATURE_COLUMNS

runner = CliRunner()
T0 = datetime(2024, 1, 1, tzinfo=UTC)
EXP = "feat_exp"


def _candles(n: int) -> pl.DataFrame:
    times = [T0 + timedelta(hours=i) for i in range(n)]
    close = [100.0 + 10.0 * ((i // 12) % 2) for i in range(n)]
    return pl.DataFrame(
        {
            "open_time": times,
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [10.0 + (i % 5) for i in range(n)],
            "close_time": [t + timedelta(hours=1) for t in times],
            "quote_volume": [1000.0] * n,
            "trades": [5] * n,
            "taker_buy_base": [4.0] * n,
            "taker_buy_quote": [400.0] * n,
        },
        schema=CANDLE_SCHEMA,
    )


def _keys(symbol: str = "BTCUSDT") -> dict[str, str]:
    return {
        "exchange": "binance",
        "market": "futures",
        "symbol": symbol,
        "timeframe": "1h",
    }


def _seed_candles(data_root: Path, symbols: tuple[str, ...] = ("BTCUSDT",), n: int = 24 * 60):
    store = ParquetStore(data_root)
    for sym in symbols:
        store.write_candles(_candles(n), **_keys(sym))


def _seed_series(data_root: Path, symbol: str = "BTCUSDT", n: int = 24 * 60) -> None:
    """Записать funding/OI/LS, как это делает RUN-2."""
    store = ParquetStore(data_root)
    times = [T0 + timedelta(hours=i) for i in range(n)]
    store.write_series(
        FUNDING,
        pl.DataFrame(
            {"open_time": times[::8], "funding_rate": [0.0002] * len(times[::8])},
            schema=FUNDING_SCHEMA,
        ),
        **_keys(symbol),
    )
    store.write_series(
        OPEN_INTEREST,
        pl.DataFrame(
            {
                "open_time": times,
                "open_interest": [1000.0 + i for i in range(n)],
                "open_interest_value": [5e6 + i for i in range(n)],
            },
            schema=OPEN_INTEREST_SCHEMA,
        ),
        **_keys(symbol),
    )
    store.write_series(
        LONG_SHORT_RATIO,
        pl.DataFrame(
            {
                "open_time": times,
                "long_short_ratio": [1.5] * n,
                "long_account": [0.6] * n,
                "short_account": [0.4] * n,
            },
            schema=LONG_SHORT_RATIO_SCHEMA,
        ),
        **_keys(symbol),
    )


def _config(path: Path, *, symbols: str = "[BTCUSDT]") -> Path:
    path.write_text(
        textwrap.dedent(
            f"""
            schema_version: 1
            name: {EXP}
            data:
              exchange: binance
              market: futures
              symbols: {symbols}
              timeframe: 1h
              start: "2024-01-01"
              end: "2024-03-01"
            time_slices:
              type: month
              min_bars: 50
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
        ),
        encoding="utf-8",
    )
    return path


# --- таблица строится со схемой REG-2 --------------------------------------


def test_builds_table_with_regime_schema(tmp_path: Path) -> None:
    data_root, features_root = tmp_path / "raw", tmp_path / "features"
    _seed_candles(data_root)
    _seed_series(data_root)
    cfg = _config(tmp_path / "exp.yaml")

    out = build_features_from_config(cfg, data_root=data_root, features_root=features_root)
    assert out == features_path(features_root, EXP)
    assert out.name == FEATURES_FILE

    table = pl.read_parquet(out)
    assert table.columns == list(TIME_SLICE_FEATURE_COLUMNS)
    # 2 месячных слайса (янв, фев) для одного символа.
    assert table.height == 2
    assert set(table["time_slice_id"].to_list()) == {"2024-01", "2024-02"}
    assert table["regime_label"].null_count() == 0
    # Ряды загружены — соответствующие фичи заполнены.
    assert table["funding_avg"].null_count() == 0
    assert table["oi_change_pct"].null_count() == 0
    assert table["ls_ratio_avg"].null_count() == 0


def test_multi_symbol_rows(tmp_path: Path) -> None:
    data_root, features_root = tmp_path / "raw", tmp_path / "features"
    _seed_candles(data_root, symbols=("BTCUSDT", "ETHUSDT"))
    cfg = _config(tmp_path / "exp.yaml", symbols="[BTCUSDT, ETHUSDT]")

    table = pl.read_parquet(
        build_features_from_config(cfg, data_root=data_root, features_root=features_root)
    )
    # 2 символа × 2 слайса.
    assert table.height == 4
    assert set(table["symbol"].to_list()) == {"BTCUSDT", "ETHUSDT"}


# --- главный инвариант: ключи join совпадают с прогоном ---------------------


def test_time_slice_ids_match_run_experiment(tmp_path: Path) -> None:
    """``time_slice_id`` фич и таблицы AN-1 совпадают — иначе join в RUN-5 пуст.

    Расхождение было бы тихим: left-join просто вернул бы null во всех режимных
    колонках, без ошибки.
    """
    data_root = tmp_path / "raw"
    features_root, results_root = tmp_path / "features", tmp_path / "results"
    _seed_candles(data_root)
    _seed_series(data_root)
    cfg = _config(tmp_path / "exp.yaml")

    features = pl.read_parquet(
        build_features_from_config(cfg, data_root=data_root, features_root=features_root)
    )
    performance = pl.read_parquet(
        run_experiment_from_config(cfg, data_root=data_root, results_root=results_root, workers=1)
    )

    assert set(features["time_slice_id"]) == set(performance["time_slice_id"])
    # И ключ join целиком, а не только слайс.
    join_cols = ["symbol", "timeframe", "time_slice_id"]
    assert set(map(tuple, features.select(join_cols).rows())) == set(
        map(tuple, performance.select(join_cols).unique().rows())
    )


def test_join_with_performance_is_not_empty(tmp_path: Path) -> None:
    """Фактический join фич с AN-1 даёт непустой результат и заполненные режимы."""
    from trading_research.analytics.correlations import attach_regime_features

    data_root = tmp_path / "raw"
    features_root, results_root = tmp_path / "features", tmp_path / "results"
    _seed_candles(data_root)
    _seed_series(data_root)
    cfg = _config(tmp_path / "exp.yaml")

    features = pl.read_parquet(
        build_features_from_config(cfg, data_root=data_root, features_root=features_root)
    )
    performance = pl.read_parquet(
        run_experiment_from_config(cfg, data_root=data_root, results_root=results_root, workers=1)
    )

    joined = attach_regime_features(performance, features)
    assert joined.height == performance.height
    assert joined["regime_label"].null_count() == 0, "join не склеился — ключи разошлись"
    assert joined["volatility"].null_count() == 0


# --- отсутствие рядов не ломает построение ---------------------------------


def test_missing_series_leaves_nulls(tmp_path: Path) -> None:
    """Без funding/OI/LS таблица строится, а их фичи остаются null."""
    data_root, features_root = tmp_path / "raw", tmp_path / "features"
    _seed_candles(data_root)  # только свечи, рядов нет
    cfg = _config(tmp_path / "exp.yaml")

    table = pl.read_parquet(
        build_features_from_config(cfg, data_root=data_root, features_root=features_root)
    )
    assert table.height == 2
    # Фичи из рядов — полностью null.
    assert table["funding_avg"].null_count() == table.height
    assert table["oi_change_pct"].null_count() == table.height
    assert table["ls_ratio_avg"].null_count() == table.height
    # Фичи из свечей посчитаны.
    assert table["volatility"].null_count() == 0
    assert table["regime_label"].null_count() == 0


def test_read_series_or_none_on_empty_store(tmp_path: Path) -> None:
    """``read_series`` на пустом store отдаёт фрейм без колонок → нормализуем в None.

    Пустой фрейм нельзя передавать в REG-2: он фильтрует ряд по ``open_time`` и
    упал бы с ``ColumnNotFoundError``.
    """
    store = ParquetStore(tmp_path)
    raw = store.read_series(FUNDING, **_keys())
    assert raw.columns == [], "предпосылка теста: пустой store отдаёт фрейм без колонок"
    assert read_series_or_none(store, FUNDING, **_keys()) is None


def test_no_candles_yields_empty_table_with_schema(tmp_path: Path) -> None:
    """Нет свечей вообще — пустая таблица с правильными типами, а не падение."""
    data_root, features_root = tmp_path / "raw", tmp_path / "features"
    cfg = _config(tmp_path / "exp.yaml")

    table = pl.read_parquet(
        build_features_from_config(cfg, data_root=data_root, features_root=features_root)
    )
    assert table.height == 0
    assert table.columns == list(TIME_SLICE_FEATURE_COLUMNS)
    assert table.schema["volatility"] == pl.Float64
    assert table.schema["regime_label"] == pl.String


# --- идемпотентность -------------------------------------------------------


def test_rerun_is_idempotent(tmp_path: Path) -> None:
    data_root, features_root = tmp_path / "raw", tmp_path / "features"
    _seed_candles(data_root)
    _seed_series(data_root)
    cfg = _config(tmp_path / "exp.yaml")

    first = pl.read_parquet(
        build_features_from_config(cfg, data_root=data_root, features_root=features_root)
    )
    second = pl.read_parquet(
        build_features_from_config(cfg, data_root=data_root, features_root=features_root)
    )
    assert first.equals(second)
    assert second.height == 2


# --- валидация -------------------------------------------------------------


def test_no_slices_raises(tmp_path: Path) -> None:
    data_root, features_root = tmp_path / "raw", tmp_path / "features"
    _seed_candles(data_root)
    cfg = _config(tmp_path / "exp.yaml")
    cfg.write_text(
        cfg.read_text(encoding="utf-8").replace("min_bars: 50", "min_bars: 100000"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no time slices"):
        build_features_from_config(cfg, data_root=data_root, features_root=features_root)


# --- CLI smoke -------------------------------------------------------------


def test_cli_build_features_smoke(tmp_path: Path) -> None:
    data_root, features_root = tmp_path / "raw", tmp_path / "features"
    _seed_candles(data_root)
    _seed_series(data_root)
    cfg = _config(tmp_path / "exp.yaml")

    result = runner.invoke(
        app,
        [
            "build-features",
            "--config",
            str(cfg),
            "--data-root",
            str(data_root),
            "--features-root",
            str(features_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "features ->" in result.output
    assert features_path(features_root, EXP).exists()
