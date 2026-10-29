"""Tests for RUN-4 — CLI walk-forward / holdout и секции конфига."""

from __future__ import annotations

import json
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from trading_research.cli.main import app
from trading_research.cli.walk_forward import (
    HOLDOUT_REPORT_FILE,
    OOS_EQUITY_FILE,
    SELECTIONS_FILE,
    build_folds,
    build_split,
    walk_forward_from_config,
    wf_dir,
)
from trading_research.config import load_experiment_config
from trading_research.data.schema import CANDLE_SCHEMA
from trading_research.data.storage.parquet_store import ParquetStore

runner = CliRunner()
T0 = datetime(2024, 1, 1, tzinfo=UTC)
EXP = "wf_exp"


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
            "volume": [10.0] * n,
            "close_time": [t + timedelta(hours=1) for t in times],
            "quote_volume": [1000.0] * n,
            "trades": [5] * n,
            "taker_buy_base": [4.0] * n,
            "taker_buy_quote": [400.0] * n,
        },
        schema=CANDLE_SCHEMA,
    )


def _keys(symbol: str = "BTCUSDT") -> dict[str, str]:
    return {"exchange": "binance", "market": "futures", "symbol": symbol, "timeframe": "1h"}


def _seed(data_root: Path, symbols: tuple[str, ...] = ("BTCUSDT",), n: int = 24 * 200) -> None:
    store = ParquetStore(data_root)
    for sym in symbols:
        store.write_candles(_candles(n), **_keys(sym))


def _config(
    path: Path,
    *,
    symbols: str = "[BTCUSDT]",
    wf: str | None = "train_days: 30\n  test_days: 10\n  step_days: 10",
    holdout: str | None = None,
    end: str = "2024-07-01",
) -> Path:
    wf_block = f"walk_forward:\n  {wf}\n" if wf else ""
    ho_block = f"holdout:\n  {holdout}\n" if holdout else ""
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
              end: "{end}"
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
        ).lstrip()
        + wf_block
        + ho_block,
        encoding="utf-8",
    )
    return path


# --- валидация секций конфига ---------------------------------------------


def test_walk_forward_section_parsed(tmp_path: Path) -> None:
    cfg = load_experiment_config(_config(tmp_path / "e.yaml"))
    assert cfg.walk_forward is not None
    assert (cfg.walk_forward.train_days, cfg.walk_forward.test_days) == (30, 10)
    assert cfg.walk_forward.select_by == "total_return_pct"
    assert cfg.walk_forward.anchored is False
    assert cfg.holdout is None


def test_both_sections_optional(tmp_path: Path) -> None:
    """`run-experiment` не требует новых секций — конфиг без них валиден."""
    cfg = load_experiment_config(_config(tmp_path / "e.yaml", wf=None))
    assert cfg.walk_forward is None
    assert cfg.holdout is None


@pytest.mark.parametrize(
    "bad",
    [
        "train_days: 0\n  test_days: 10\n  step_days: 10",
        "train_days: 30\n  test_days: -1\n  step_days: 10",
        "train_days: 30\n  test_days: 10\n  step_days: 0",
    ],
)
def test_non_positive_spans_rejected(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        load_experiment_config(_config(tmp_path / "e.yaml", wf=bad))


def test_unknown_select_by_rejected(tmp_path: Path) -> None:
    """Опечатка в метрике отбора ловится на уровне конфига, а не в середине прогона."""
    bad = "train_days: 30\n  test_days: 10\n  step_days: 10\n  select_by: sharp"
    with pytest.raises(ValidationError, match="numeric Metrics field"):
        load_experiment_config(_config(tmp_path / "e.yaml", wf=bad))


def test_select_by_rejects_boolean_metric(tmp_path: Path) -> None:
    """``reliable`` — булева, отбирать по ней нельзя (как и в WF-2)."""
    bad = "train_days: 30\n  test_days: 10\n  step_days: 10\n  select_by: reliable"
    with pytest.raises(ValidationError, match="numeric Metrics field"):
        load_experiment_config(_config(tmp_path / "e.yaml", wf=bad))


def test_explicit_valid_select_by_accepted(tmp_path: Path) -> None:
    """Явная валидная метрика проходит валидатор и доезжает до прогона.

    Нужен отдельно от дефолта: pydantic не валидирует значения по умолчанию, так что
    без явного значения успешная ветка валидатора не исполняется вовсе.
    """
    good = "train_days: 30\n  test_days: 10\n  step_days: 10\n  select_by: sharpe"
    cfg = load_experiment_config(_config(tmp_path / "e.yaml", wf=good))
    assert cfg.walk_forward is not None
    assert cfg.walk_forward.select_by == "sharpe"


def test_select_by_reaches_holdout_report(tmp_path: Path) -> None:
    """Выбранная метрика — та же, по которой считается деградация на holdout."""
    data_root, results_root = tmp_path / "raw", tmp_path / "res"
    _seed(data_root)
    cfg = _config(
        tmp_path / "e.yaml",
        wf="train_days: 30\n  test_days: 10\n  step_days: 10\n  select_by: max_drawdown_pct",
        holdout='start: "2024-05-01"',
    )
    walk_forward_from_config(cfg, data_root=data_root, results_root=results_root)
    report = json.loads(
        (wf_dir(results_root, EXP, "BTCUSDT") / HOLDOUT_REPORT_FILE).read_text(encoding="utf-8")
    )
    assert report["metric"] == "max_drawdown_pct"


def test_empty_holdout_rejected(tmp_path: Path) -> None:
    """Пустой резерв бессмыслен — нужно start и/или symbols."""
    with pytest.raises(ValidationError, match="requires 'start' and/or"):
        load_experiment_config(_config(tmp_path / "e.yaml", holdout="symbols: []"))


def test_holdout_section_parsed(tmp_path: Path) -> None:
    cfg = load_experiment_config(
        _config(tmp_path / "e.yaml", holdout='start: "2024-05-01"\n  symbols: [ETHUSDT]')
    )
    assert cfg.holdout is not None
    assert cfg.holdout.start == datetime(2024, 5, 1)
    assert cfg.holdout.symbols == ["ETHUSDT"]
    split = build_split(cfg)
    assert split is not None
    assert split.is_holdout(symbol="ETHUSDT", time=T0)
    assert split.is_holdout(symbol="BTCUSDT", time=datetime(2024, 6, 1, tzinfo=UTC))
    assert not split.is_holdout(symbol="BTCUSDT", time=T0)


def test_no_walk_forward_section_raises(tmp_path: Path) -> None:
    data_root = tmp_path / "raw"
    _seed(data_root)
    cfg = _config(tmp_path / "e.yaml", wf=None)
    with pytest.raises(ValueError, match="no 'walk_forward' section"):
        walk_forward_from_config(cfg, data_root=data_root, results_root=tmp_path / "res")


# --- фолды -----------------------------------------------------------------


def test_folds_stop_at_holdout_start(tmp_path: Path) -> None:
    """Фолды строятся до начала резерва, а не до конца данных."""
    full = load_experiment_config(_config(tmp_path / "a.yaml"))
    reserved = load_experiment_config(_config(tmp_path / "b.yaml", holdout='start: "2024-04-01"'))
    split = build_split(reserved)
    assert split is not None and split.holdout_start is not None

    folds_full = build_folds(full, full.data.end)
    folds_cut = build_folds(reserved, split.holdout_start)

    assert len(folds_cut) < len(folds_full)
    # Ни одно test-окно не заходит в резерв.
    assert all(f.test.end <= split.holdout_start for f in folds_cut)


def test_impossible_folds_raise(tmp_path: Path) -> None:
    """Train+test шире диапазона — понятная ошибка, а не пустой результат."""
    cfg = load_experiment_config(
        _config(tmp_path / "e.yaml", wf="train_days: 300\n  test_days: 100\n  step_days: 10")
    )
    with pytest.raises(ValueError, match="no walk-forward folds fit"):
        build_folds(cfg, cfg.data.end)


def test_anchored_folds_expand(tmp_path: Path) -> None:
    cfg = load_experiment_config(
        _config(
            tmp_path / "e.yaml",
            wf="train_days: 30\n  test_days: 10\n  step_days: 10\n  anchored: true",
        )
    )
    folds = build_folds(cfg, cfg.data.end)
    assert len(folds) >= 2
    # Expanding: начало train фиксировано, конец растёт.
    assert folds[0].train.start == folds[-1].train.start
    assert folds[-1].train.end > folds[0].train.end


# --- прогон и артефакты ----------------------------------------------------


def test_walk_forward_writes_artifacts(tmp_path: Path) -> None:
    data_root, results_root = tmp_path / "raw", tmp_path / "res"
    _seed(data_root)
    cfg = _config(tmp_path / "e.yaml")

    results = walk_forward_from_config(cfg, data_root=data_root, results_root=results_root)
    assert set(results) == {"BTCUSDT"}
    result = results["BTCUSDT"]
    assert result.selections, "ожидали хотя бы один фолд с выбором"

    out = wf_dir(results_root, EXP, "BTCUSDT")
    assert (out / OOS_EQUITY_FILE).exists()
    equity = pl.read_parquet(out / OOS_EQUITY_FILE)
    assert equity.height > 0
    assert equity.columns == ["open_time", "equity"]

    payload = json.loads((out / SELECTIONS_FILE).read_text(encoding="utf-8"))
    assert len(payload["selections"]) == len(result.selections)
    assert "oos_metrics" in payload
    assert payload["selections"][0]["params"]
    # Без секции holdout отчёта нет.
    assert not (out / HOLDOUT_REPORT_FILE).exists()


def test_oos_curve_contains_only_test_bars(tmp_path: Path) -> None:
    """В OOS-кривую попадают только бары test-окон — train туда не протекает.

    Это главный инвариант честного out-of-sample на уровне CLI: если бы в кривую
    попал train, OOS-метрики были бы завышены, и притом молча.
    """
    data_root = tmp_path / "raw"
    _seed(data_root)
    cfg_path = _config(tmp_path / "e.yaml")
    results = walk_forward_from_config(cfg_path, data_root=data_root, results_root=tmp_path / "res")
    cfg = load_experiment_config(cfg_path)
    folds = build_folds(cfg, cfg.data.end)

    equity = results["BTCUSDT"].oos_equity
    assert equity.height > 0
    test_windows = [(f.test.start, f.test.end) for f in folds]
    for ts in equity["open_time"].to_list():
        assert any(start <= ts < end for start, end in test_windows), (
            f"бар {ts} вне test-окон: в OOS-кривую протёк train"
        )


def test_multi_symbol_separate_dirs(tmp_path: Path) -> None:
    data_root, results_root = tmp_path / "raw", tmp_path / "res"
    _seed(data_root, symbols=("BTCUSDT", "ETHUSDT"))
    cfg = _config(tmp_path / "e.yaml", symbols="[BTCUSDT, ETHUSDT]")

    results = walk_forward_from_config(cfg, data_root=data_root, results_root=results_root)
    assert set(results) == {"BTCUSDT", "ETHUSDT"}
    for sym in ("BTCUSDT", "ETHUSDT"):
        assert (wf_dir(results_root, EXP, sym) / OOS_EQUITY_FILE).exists()


# --- holdout ---------------------------------------------------------------


def test_holdout_report_written(tmp_path: Path) -> None:
    data_root, results_root = tmp_path / "raw", tmp_path / "res"
    _seed(data_root)
    cfg = _config(tmp_path / "e.yaml", holdout='start: "2024-05-01"')

    results = walk_forward_from_config(cfg, data_root=data_root, results_root=results_root)
    out = wf_dir(results_root, EXP, "BTCUSDT")
    report = json.loads((out / HOLDOUT_REPORT_FILE).read_text(encoding="utf-8"))

    assert report["symbol"] == "BTCUSDT"
    assert report["metric"] == "total_return_pct"
    assert report["params"]
    # Сравнение именно с OOS-оценкой research, а не с train-скором.
    oos = results["BTCUSDT"].oos_metrics.total_return_pct
    assert report["research_value"] == pytest.approx(oos)
    assert report["degradation"] == pytest.approx(
        report["research_value"] - report["holdout_value"]
    )
    assert report["holdout_metrics"]["trades_count"] >= 0


def test_holdout_params_come_from_last_fold(tmp_path: Path) -> None:
    data_root, results_root = tmp_path / "raw", tmp_path / "res"
    _seed(data_root)
    cfg = _config(tmp_path / "e.yaml", holdout='start: "2024-05-01"')

    results = walk_forward_from_config(cfg, data_root=data_root, results_root=results_root)
    report = json.loads(
        (wf_dir(results_root, EXP, "BTCUSDT") / HOLDOUT_REPORT_FILE).read_text(encoding="utf-8")
    )
    assert report["params"] == results["BTCUSDT"].selections[-1].params


def test_reserved_symbol_excluded_from_research(tmp_path: Path) -> None:
    """Символ в ``holdout.symbols`` целиком уходит в резерв — в research его нет."""
    data_root, results_root = tmp_path / "raw", tmp_path / "res"
    _seed(data_root, symbols=("BTCUSDT", "ETHUSDT"))
    cfg = _config(tmp_path / "e.yaml", symbols="[BTCUSDT, ETHUSDT]", holdout="symbols: [ETHUSDT]")

    results = walk_forward_from_config(cfg, data_root=data_root, results_root=results_root)
    assert set(results) == {"BTCUSDT"}
    assert not wf_dir(results_root, EXP, "ETHUSDT").exists()


def test_research_never_contains_holdout_bars(tmp_path: Path) -> None:
    """``assert_research_only`` — страховка; проверяем, что резерв реально отрезан."""
    data_root = tmp_path / "raw"
    _seed(data_root)
    cfg_path = _config(tmp_path / "e.yaml", holdout='start: "2024-05-01"')
    cfg = load_experiment_config(cfg_path)
    split = build_split(cfg)
    assert split is not None

    data = ParquetStore(data_root).read_candles(**_keys())
    research = split.research_data(data, symbol="BTCUSDT")
    assert research.height > 0
    assert research["open_time"].max() < split.holdout_start  # type: ignore[operator]
    # А полный фрейм страховку не проходит.
    from trading_research.experiments.holdout import assert_research_only

    with pytest.raises(ValueError, match="holdout bar"):
        assert_research_only(data, split, symbol="BTCUSDT")


# --- CLI smoke -------------------------------------------------------------


def test_cli_walk_forward_smoke(tmp_path: Path) -> None:
    data_root, results_root = tmp_path / "raw", tmp_path / "res"
    _seed(data_root)
    cfg = _config(tmp_path / "e.yaml", holdout='start: "2024-05-01"')

    result = runner.invoke(
        app,
        [
            "walk-forward",
            "--config",
            str(cfg),
            "--data-root",
            str(data_root),
            "--results-root",
            str(results_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "BTCUSDT" in result.output
    assert "folds" in result.output
    assert "OOS:" in result.output
    assert (wf_dir(results_root, EXP, "BTCUSDT") / HOLDOUT_REPORT_FILE).exists()


def test_cli_no_symbols_message(tmp_path: Path) -> None:
    """Нет данных — внятное сообщение, а не пустой вывод."""
    cfg = _config(tmp_path / "e.yaml")
    result = runner.invoke(
        app,
        [
            "walk-forward",
            "--config",
            str(cfg),
            "--data-root",
            str(tmp_path / "empty"),
            "--results-root",
            str(tmp_path / "res"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no symbols" in result.output
