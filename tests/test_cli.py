"""Tests for EXP-6 — CLI: fetch-data / run-experiment (smoke + бизнес-логика)."""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from trading_research.cli.fetch_data import fetch_data
from trading_research.cli.main import app
from trading_research.cli.run_experiment import run_experiment_from_config
from trading_research.data.schema import CANDLE_SCHEMA
from trading_research.data.storage.parquet_store import ParquetStore

runner = CliRunner()
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _candles(n: int, start: datetime = T0) -> pl.DataFrame:
    times = [start + timedelta(hours=i) for i in range(n)]
    # Пила по цене, чтобы MA Cross давала сделки.
    close = [100.0 + 10.0 * ((i // 12) % 2) for i in range(n)]
    return pl.DataFrame(
        {
            "open_time": times,
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [10.0] * n,
            "close_time": [t + timedelta(hours=1) for t in times],
            "quote_volume": [1000.0] * n,
            "trades": [5] * n,
            "taker_buy_base": [4.0] * n,
            "taker_buy_quote": [400.0] * n,
        },
        schema=CANDLE_SCHEMA,
    )


class _StubFetcher:
    """Подставной источник свечей (без сети)."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def fetch_klines(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        return self._df


def _seed_store(data_root: Path, n: int = 24 * 60) -> None:
    ParquetStore(data_root).write_candles(
        _candles(n), exchange="binance", market="futures", symbol="BTCUSDT", timeframe="1h"
    )


def _write_config(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            schema_version: 1
            name: smoke_ma_cross
            data:
              exchange: binance
              market: futures
              symbols: [BTCUSDT]
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


# --- fetch-data -------------------------------------------------------------


def test_fetch_data_business_writes_candles(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    n = fetch_data(
        exchange="binance",
        market="futures",
        symbol="BTCUSDT",
        timeframe="1h",
        start=T0,
        end=T0 + timedelta(hours=24),
        data_root=root,
        fetcher=_StubFetcher(_candles(24)),
    )
    assert n == 24
    back = ParquetStore(root).read_candles(
        exchange="binance", market="futures", symbol="BTCUSDT", timeframe="1h"
    )
    assert back.height == 24


def test_cli_fetch_data_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Подменяем реальный клиент на стаб (без сети).
    monkeypatch.setattr(
        "trading_research.cli.fetch_data.BinanceFuturesClient",
        lambda: _StubFetcher(_candles(24)),
    )
    root = tmp_path / "raw"
    result = runner.invoke(
        app,
        [
            "fetch-data",
            "--symbol",
            "BTCUSDT",
            "--timeframe",
            "1h",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-02",
            "--data-root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "wrote 24 candles" in result.output
    assert any(root.glob("candles/**/*.parquet"))


# --- run-experiment ---------------------------------------------------------


def test_run_experiment_business_writes_metrics(tmp_path: Path) -> None:
    data_root = tmp_path / "raw"
    results_root = tmp_path / "results"
    _seed_store(data_root)
    cfg = tmp_path / "exp.yaml"
    _write_config(cfg)

    out = run_experiment_from_config(cfg, data_root=data_root, results_root=results_root)
    assert out.exists()
    df = pl.read_parquet(out)
    # 1 символ × 1 комбинация × 2 месячных слайса (янв, фев).
    assert df.height == 2
    assert {
        "experiment_id",
        "symbol",
        "timeframe",
        "time_slice_id",
        "param_hash",
        "params_json",
        "total_return_pct",
        "trades_count",
    } <= set(df.columns)
    assert set(df["time_slice_id"].to_list()) == {"2024-01", "2024-02"}
    assert df["experiment_id"].to_list() == ["smoke_ma_cross"] * 2


def test_cli_run_experiment_smoke(tmp_path: Path) -> None:
    data_root = tmp_path / "raw"
    results_root = tmp_path / "results"
    _seed_store(data_root)
    cfg = tmp_path / "exp.yaml"
    _write_config(cfg)

    result = runner.invoke(
        app,
        [
            "run-experiment",
            "--config",
            str(cfg),
            "--data-root",
            str(data_root),
            "--results-root",
            str(results_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "results ->" in result.output
    assert (results_root / "smoke_ma_cross" / "performance.parquet").exists()


def test_cli_analyze_results_smoke(tmp_path: Path) -> None:
    data_root = tmp_path / "raw"
    results_root = tmp_path / "results"
    _seed_store(data_root)
    cfg = tmp_path / "exp.yaml"
    _write_config(cfg)
    # Сначала прогон — он пишет таблицу результатов.
    runner.invoke(
        app,
        [
            "run-experiment",
            "--config",
            str(cfg),
            "--data-root",
            str(data_root),
            "--results-root",
            str(results_root),
        ],
    )
    result = runner.invoke(
        app,
        [
            "analyze-results",
            "--experiment",
            "smoke_ma_cross",
            "--metric",
            "total_return_pct",
            "--top",
            "5",
            "--results-root",
            str(results_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "param_hash" in result.output
    assert "rank" in result.output


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip()
