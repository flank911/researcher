"""Tests for RUN-5 — CLI `analyze-regimes` и согласованность режимных слотов AN-1."""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from trading_research.analytics.correlations import NUMERIC_REGIME_FEATURES
from trading_research.analytics.queries import PERFORMANCE_COLUMNS, REGIME_SLOTS
from trading_research.cli.analyze_regimes import (
    analyze_regimes,
    available_features,
    export_regimes,
    load_joined,
    regime_summary,
)
from trading_research.cli.build_features import build_features_from_config, features_path
from trading_research.cli.main import app
from trading_research.cli.run_experiment import run_experiment_from_config
from trading_research.data.schema import CANDLE_SCHEMA
from trading_research.data.storage.parquet_store import ParquetStore
from trading_research.features.market_regime import SliceFeatures

runner = CliRunner()
T0 = datetime(2024, 1, 1, tzinfo=UTC)
EXP = "reg_exp"
_JOIN_KEYS = {"symbol", "timeframe", "time_slice_id"}


# --- главный инвариант: два списка истины согласованы ----------------------


def test_regime_slots_match_slice_features() -> None:
    """``REGIME_SLOTS`` (AN-1) ≡ поля ``SliceFeatures`` (REG-2) без ключей join.

    До RUN-5 в AN-1 не хватало ``adx``, ``volume_zscore`` и ``drawdown_from_ath``.
    Через join они в таблицу попадали, но ``NUMERIC_REGIME_FEATURES`` выводится из
    ``REGIME_SLOTS``, поэтому по умолчанию корреляции считались без них — молча.

    Списки намеренно не выведены один из другого: аналитика не зависит от слоя фич
    (см. docstring ``analytics/correlations.py``). Цена этого решения — вот этот
    тест; он обязан падать при любом расхождении.
    """
    expected = set(SliceFeatures.model_fields) - _JOIN_KEYS
    assert set(REGIME_SLOTS) == expected, (
        "REGIME_SLOTS разошлись с SliceFeatures: "
        f"нет в AN-1 {sorted(expected - set(REGIME_SLOTS))}, "
        f"лишние в AN-1 {sorted(set(REGIME_SLOTS) - expected)}"
    )


def test_numeric_regime_features_excludes_only_label() -> None:
    """Из числовых фич исключён ровно ``regime_label`` — остальные 9 участвуют."""
    assert set(NUMERIC_REGIME_FEATURES) == set(REGIME_SLOTS) - {"regime_label"}
    assert len(NUMERIC_REGIME_FEATURES) == 9


def test_regime_slots_are_in_performance_schema() -> None:
    assert set(REGIME_SLOTS) <= set(PERFORMANCE_COLUMNS)


# --- фикстуры: реальные таблицы обеих команд ------------------------------


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


def _config(path: Path, *, name: str = EXP) -> Path:
    path.write_text(
        textwrap.dedent(
            f"""
            schema_version: 2
            name: {name}
            data:
              exchange: binance
              market: futures
              symbols: [BTCUSDT]
              timeframes: [1h]
              start: "2024-01-01"
              end: "2024-04-01"
            time_slices:
              type: month
              min_bars: 50
            strategy:
              name: ma_cross
            params_grid:
              fast_ma: [5, 10]
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


def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    """Прогнать обе команды и вернуть (results_root, features_root)."""
    data_root = tmp_path / "raw"
    results_root, features_root = tmp_path / "res", tmp_path / "feat"
    ParquetStore(data_root).write_candles(
        _candles(24 * 100),
        exchange="binance",
        market="futures",
        symbol="BTCUSDT",
        timeframe="1h",
    )
    cfg = _config(tmp_path / "e.yaml")
    run_experiment_from_config(cfg, data_root=data_root, results_root=results_root, workers=1)
    build_features_from_config(cfg, data_root=data_root, features_root=features_root)
    return results_root, features_root


# --- join -----------------------------------------------------------------


def test_join_fills_all_regime_slots(tmp_path: Path) -> None:
    """Join заполняет слоты, которые прогон оставил null — включая три новые."""
    results_root, features_root = _prepare(tmp_path)

    performance = pl.read_parquet(results_root / EXP / "performance.parquet")
    # В самой таблице AN-1 слоты пусты — их заполняет анализ, а не прогон.
    for slot in REGIME_SLOTS:
        assert performance[slot].null_count() == performance.height, slot

    joined = load_joined(results_root, features_root, EXP)
    assert joined.height == performance.height
    for slot in ("adx", "volume_zscore", "drawdown_from_ath"):
        assert joined[slot].null_count() == 0, f"{slot} не доехал через join"
    assert joined["regime_label"].null_count() == 0
    assert available_features(joined) == list(NUMERIC_REGIME_FEATURES)


def test_missing_performance_table_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run 'run-experiment' first"):
        load_joined(tmp_path / "res", tmp_path / "feat", EXP)


def test_missing_features_table_raises(tmp_path: Path) -> None:
    results_root, _ = _prepare(tmp_path)
    with pytest.raises(FileNotFoundError, match="run 'build-features' first"):
        load_joined(results_root, tmp_path / "nowhere", EXP)


def test_mismatched_keys_raise_not_nulls(tmp_path: Path) -> None:
    """Главный тихий отказ: ключи разошлись → ошибка, а не матрица из null."""
    results_root, features_root = _prepare(tmp_path)
    # Портим ключи join, как если бы фичи построили по другому конфигу.
    path = features_path(features_root, EXP)
    broken = pl.read_parquet(path).with_columns(pl.lit("1900-01").alias("time_slice_id"))
    broken.write_parquet(path)

    with pytest.raises(ValueError, match="regime join produced no matches"):
        load_joined(results_root, features_root, EXP)


# --- корреляции -----------------------------------------------------------


def test_correlation_matrix_covers_all_nine_features(tmp_path: Path) -> None:
    """Матрица содержит все 9 числовых фич — ровно то, что было сломано."""
    results_root, features_root = _prepare(tmp_path)
    corr = analyze_regimes(results_root, features_root, EXP)

    assert "param_hash" in corr.columns
    assert set(NUMERIC_REGIME_FEATURES) <= set(corr.columns)
    for new in ("adx", "volume_zscore", "drawdown_from_ath"):
        assert new in corr.columns, f"{new} выпал из матрицы корреляций"
    # Строка на комбинацию параметров.
    assert corr.height == 2


def test_global_matrix_is_single_row(tmp_path: Path) -> None:
    results_root, features_root = _prepare(tmp_path)
    corr = analyze_regimes(results_root, features_root, EXP, by_param=False)
    assert corr.height == 1
    assert "param_hash" not in corr.columns


def test_unknown_metric_rejected(tmp_path: Path) -> None:
    results_root, features_root = _prepare(tmp_path)
    with pytest.raises(ValueError, match="not queryable"):
        analyze_regimes(results_root, features_root, EXP, metric="nope")


def test_regime_summary(tmp_path: Path) -> None:
    results_root, features_root = _prepare(tmp_path)
    summary = regime_summary(results_root, features_root, EXP)
    assert {"param_hash", "regime_label", "total_return_pct", "n"} <= set(summary.columns)
    assert summary.height > 0


# --- top-N внутри режима --------------------------------------------------


def test_best_params_in_regime(tmp_path: Path) -> None:
    results_root, features_root = _prepare(tmp_path)
    joined = load_joined(results_root, features_root, EXP)
    regime = joined["regime_label"].drop_nulls()[0]

    top = analyze_regimes(results_root, features_root, EXP, regime=regime, top=5)
    assert top.height > 0
    assert {"param_hash", "params_json", "total_return_pct", "n"} <= set(top.columns)
    assert top.height <= 5


def test_unknown_regime_lists_available(tmp_path: Path) -> None:
    """Опечатка в режиме — ошибка со списком доступных, а не пустая таблица."""
    results_root, features_root = _prepare(tmp_path)
    with pytest.raises(ValueError, match="not present; found:"):
        analyze_regimes(results_root, features_root, EXP, regime="no_such_regime")


# --- экспорт --------------------------------------------------------------


def test_export_csv_and_parquet(tmp_path: Path) -> None:
    results_root, features_root = _prepare(tmp_path)

    csv_out = export_regimes(results_root, features_root, EXP, tmp_path / "c.csv", fmt="csv")
    assert csv_out.exists()
    assert pl.read_csv(csv_out).height == 2

    pq_out = export_regimes(
        results_root, features_root, EXP, tmp_path / "sub" / "c.parquet", fmt="parquet"
    )
    assert pq_out.exists()
    assert set(NUMERIC_REGIME_FEATURES) <= set(pl.read_parquet(pq_out).columns)


def test_export_unknown_format_rejected(tmp_path: Path) -> None:
    results_root, features_root = _prepare(tmp_path)
    with pytest.raises(ValueError, match="unknown format"):
        export_regimes(results_root, features_root, EXP, tmp_path / "x.txt", fmt="txt")


# --- CLI smoke ------------------------------------------------------------


def test_cli_analyze_regimes_smoke(tmp_path: Path) -> None:
    results_root, features_root = _prepare(tmp_path)
    result = runner.invoke(
        app,
        [
            "analyze-regimes",
            "--experiment",
            EXP,
            "--results-root",
            str(results_root),
            "--features-root",
            str(features_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "param_hash" in result.output
    assert "volatility" in result.output


def test_cli_with_export_and_regime(tmp_path: Path) -> None:
    results_root, features_root = _prepare(tmp_path)
    joined = load_joined(results_root, features_root, EXP)
    regime = joined["regime_label"].drop_nulls()[0]
    out = tmp_path / "top.csv"

    result = runner.invoke(
        app,
        [
            "analyze-regimes",
            "--experiment",
            EXP,
            "--regime",
            regime,
            "--top",
            "3",
            "--out",
            str(out),
            "--results-root",
            str(results_root),
            "--features-root",
            str(features_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "export ->" in result.output
    assert out.exists()


def test_cli_missing_tables_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "analyze-regimes",
            "--experiment",
            "nope",
            "--results-root",
            str(tmp_path / "res"),
            "--features-root",
            str(tmp_path / "feat"),
        ],
    )
    assert result.exit_code != 0
