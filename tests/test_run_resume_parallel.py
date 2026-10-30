"""Tests for RUN-1 — resume/dedupe и параллелизм в CLI-прогоне.

Проверяют, что `run-experiment` действительно использует EXP-3 (``RunLedger``),
EXP-4 (``ResultsStore``) и EXP-5 (пул процессов), а не только строит таблицу AN-1.
"""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from trading_research.cli.run_experiment import (
    LEDGER_DIR,
    group_specs_by_params,
    run_experiment_from_config,
)
from trading_research.data.schema import CANDLE_SCHEMA
from trading_research.data.storage.parquet_store import ParquetStore
from trading_research.experiments.results_store import ResultsStore

T0 = datetime(2024, 1, 1, tzinfo=UTC)
_RUNNER = "trading_research.cli.run_experiment.run_backtest"


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


def _seed(data_root: Path, n: int = 24 * 60) -> None:
    ParquetStore(data_root).write_candles(
        _candles(n), exchange="binance", market="futures", symbol="BTCUSDT", timeframe="1h"
    )


def _config(
    path: Path, *, combos: int = 2, save_artifacts: bool = True, slice_mode: str = "continuous"
) -> Path:
    fast = "[5, 7]" if combos == 2 else "[5]"
    path.write_text(
        textwrap.dedent(
            f"""
            schema_version: 2
            name: run1_exp
            data:
              exchange: binance
              market: futures
              symbols: [BTCUSDT]
              timeframes: [1h]
              start: "2024-01-01"
              end: "2024-03-01"
            time_slices:
              type: month
              min_bars: 50
            strategy:
              name: ma_cross
            slice_mode: {slice_mode}
            params_grid:
              fast_ma: {fast}
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
            output:
              save_trades: {str(save_artifacts).lower()}
              save_equity_curve: {str(save_artifacts).lower()}
            """
        ),
        encoding="utf-8",
    )
    return path


def _setup(tmp_path: Path, **cfg_kw: Any) -> tuple[Path, Path, Path]:
    data_root, results_root = tmp_path / "raw", tmp_path / "results"
    _seed(data_root)
    cfg = _config(tmp_path / "exp.yaml", **cfg_kw)
    return cfg, data_root, results_root


# --- EXP-4: каталог наполняется -------------------------------------------


def test_results_store_is_populated(tmp_path: Path) -> None:
    """Раны попадают в ResultsStore, а не только в performance.parquet."""
    cfg, data_root, results_root = _setup(tmp_path)
    run_experiment_from_config(cfg, data_root=data_root, results_root=results_root, workers=1)

    store = ResultsStore(results_root / "run1_exp")
    # 2 комбинации × 2 месячных слайса.
    assert store.count() == 4
    runs = store.all_runs()
    assert all(r.metrics is not None for r in runs)
    assert {r.slice_id for r in runs} == {"2024-01", "2024-02"}
    assert all(r.dataset_hash for r in runs), "dataset_hash должен быть заполнен (DATA-4)"
    assert all(r.git_commit for r in runs), "git_commit должен быть заполнен (INF-3)"


def test_output_flags_control_artifacts(tmp_path: Path) -> None:
    """``output.save_trades``/``save_equity_curve`` реально управляют артефактами."""
    cfg_on, data_root, results_on = _setup(tmp_path / "on", combos=1, save_artifacts=True)
    run_experiment_from_config(cfg_on, data_root=data_root, results_root=results_on, workers=1)
    store_on = ResultsStore(results_on / "run1_exp")
    hashes_on = [r.run_hash for r in store_on.all_runs()]
    assert any(store_on.load_equity(h).height > 0 for h in hashes_on)

    cfg_off, data_root2, results_off = _setup(tmp_path / "off", combos=1, save_artifacts=False)
    run_experiment_from_config(cfg_off, data_root=data_root2, results_root=results_off, workers=1)
    store_off = ResultsStore(results_off / "run1_exp")
    hashes_off = [r.run_hash for r in store_off.all_runs()]
    assert hashes_off, "раны должны быть записаны и при выключенных артефактах"
    assert all(store_off.load_equity(h).height == 0 for h in hashes_off)
    assert all(store_off.load_trades(h).height == 0 for h in hashes_off)
    # Метрики при этом сохранены — выключается только тяжёлый артефакт.
    assert all(r.metrics is not None for r in store_off.all_runs())


# --- EXP-3: dedupe и resume -----------------------------------------------


def test_second_run_skips_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Повторный прогон не вызывает движок: все раны уже в ledger (dedupe)."""
    cfg, data_root, results_root = _setup(tmp_path)
    run_experiment_from_config(cfg, data_root=data_root, results_root=results_root, workers=1)

    markers = list((results_root / "run1_exp" / LEDGER_DIR).glob("*.json"))
    assert len(markers) == 4

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("движок не должен вызываться на повторном прогоне")

    monkeypatch.setattr(_RUNNER, _boom)
    out = run_experiment_from_config(cfg, data_root=data_root, results_root=results_root, workers=1)
    assert pl.read_parquet(out).height == 4


def test_no_resume_recomputes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``resume=False`` сбрасывает маркеры и считает заново (движок вызывается)."""
    cfg, data_root, results_root = _setup(tmp_path)
    run_experiment_from_config(cfg, data_root=data_root, results_root=results_root, workers=1)
    ledger_dir = results_root / "run1_exp" / LEDGER_DIR
    first = {p.name for p in ledger_dir.glob("*.json")}

    import trading_research.cli.run_experiment as mod

    real = mod.run_backtest
    calls: list[dict[str, Any]] = []

    def _counting(strategy: Any, data: Any, params: Any, execution: Any, **kw: Any) -> Any:
        calls.append(dict(params))
        return real(strategy, data, params, execution, **kw)

    monkeypatch.setattr(_RUNNER, _counting)
    out = run_experiment_from_config(
        cfg, data_root=data_root, results_root=results_root, workers=1, resume=False
    )
    assert pl.read_parquet(out).height == 4
    # Один прогон на комбинацию (CONTINUOUS), а не на каждый слайс.
    assert len(calls) == 2, f"ожидали 2 прогона движка, было {len(calls)}"
    # Маркеры те же — хэши детерминированы и не зависят от того, кэш это или пересчёт.
    assert {p.name for p in ledger_dir.glob("*.json")} == first


def test_interrupted_run_completes_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Группа, упавшая в первом прогоне, доделывается во втором (resume)."""
    cfg, data_root, results_root = _setup(tmp_path)

    import trading_research.cli.run_experiment as mod

    real = mod.run_backtest
    failed_params: dict[str, Any] = {}

    def _fail_on_second(strategy: Any, data: Any, params: Any, execution: Any, **kw: Any) -> Any:
        # Роняем комбинацию fast_ma=7, оставляя fast_ma=5 успешной.
        if params.get("fast_ma") == 7:
            failed_params.update(params)
            raise RuntimeError("simulated worker crash")
        return real(strategy, data, params, execution, **kw)

    monkeypatch.setattr(_RUNNER, _fail_on_second)
    run_experiment_from_config(cfg, data_root=data_root, results_root=results_root, workers=1)

    assert failed_params, "тест должен был уронить одну комбинацию"
    ledger_dir = results_root / "run1_exp" / LEDGER_DIR
    # Упавшая группа не помечена готовой — только 2 слайса успешной комбинации.
    assert len(list(ledger_dir.glob("*.json"))) == 2
    store = ResultsStore(results_root / "run1_exp")
    assert store.count() == 2

    # Второй прогон без сбоя доделывает остаток.
    monkeypatch.setattr(_RUNNER, real)
    out = run_experiment_from_config(cfg, data_root=data_root, results_root=results_root, workers=1)
    assert len(list(ledger_dir.glob("*.json"))) == 4
    assert ResultsStore(results_root / "run1_exp").count() == 4
    assert pl.read_parquet(out).height == 4


# --- EXP-5: параллелизм ---------------------------------------------------


def test_workers_do_not_change_result(tmp_path: Path) -> None:
    """``workers=2`` даёт тот же результат, что ``workers=1`` (детерминизм)."""
    cfg1, data1, res1 = _setup(tmp_path / "seq")
    out_seq = run_experiment_from_config(cfg1, data_root=data1, results_root=res1, workers=1)

    cfg2, data2, res2 = _setup(tmp_path / "par")
    out_par = run_experiment_from_config(cfg2, data_root=data2, results_root=res2, workers=2)

    seq = pl.read_parquet(out_seq).sort(["param_hash", "time_slice_id"])
    par = pl.read_parquet(out_par).sort(["param_hash", "time_slice_id"])
    assert seq.height == par.height == 4
    assert seq.equals(par)


# --- режим INDEPENDENT ----------------------------------------------------


def test_independent_mode_runs_per_slice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """В ``INDEPENDENT`` движок вызывается на каждое окно, а не раз на комбинацию."""
    cfg, data_root, results_root = _setup(tmp_path, combos=1, slice_mode="independent")

    import trading_research.cli.run_experiment as mod

    real = mod.run_backtest
    calls: list[int] = []

    def _counting(strategy: Any, data: Any, params: Any, execution: Any, **kw: Any) -> Any:
        calls.append(data.height)
        return real(strategy, data, params, execution, **kw)

    monkeypatch.setattr(_RUNNER, _counting)
    out = run_experiment_from_config(cfg, data_root=data_root, results_root=results_root, workers=1)

    # 1 комбинация × 2 окна → 2 прогона, каждый по своему (меньшему) срезу данных.
    assert len(calls) == 2
    assert all(h > 0 for h in calls)
    table = pl.read_parquet(out)
    assert table.height == 2
    assert set(table["time_slice_id"].to_list()) == {"2024-01", "2024-02"}
    assert ResultsStore(results_root / "run1_exp").count() == 2


def test_independent_resets_balance_per_window(tmp_path: Path) -> None:
    """Независимые окна считаются от ``initial_balance``, а не с переносом."""
    cfg_c, data_c, res_c = _setup(tmp_path / "cont", combos=1, slice_mode="continuous")
    cont = pl.read_parquet(
        run_experiment_from_config(cfg_c, data_root=data_c, results_root=res_c, workers=1)
    ).sort("time_slice_id")

    cfg_i, data_i, res_i = _setup(tmp_path / "indep", combos=1, slice_mode="independent")
    indep = pl.read_parquet(
        run_experiment_from_config(cfg_i, data_root=data_i, results_root=res_i, workers=1)
    ).sort("time_slice_id")

    # Оба режима дают те же окна, но это разные вычисления — метрики не обязаны
    # совпадать. Проверяем, что независимый режим действительно отработал.
    assert cont["time_slice_id"].to_list() == indep["time_slice_id"].to_list()
    assert indep["trades_count"].sum() > 0


# --- валидация ------------------------------------------------------------


def test_no_slices_raises(tmp_path: Path) -> None:
    """Конфиг, не дающий ни одного окна, падает понятной ошибкой."""
    data_root, results_root = tmp_path / "raw", tmp_path / "results"
    _seed(data_root)
    cfg = tmp_path / "exp.yaml"
    _config(cfg)
    # min_bars заведомо больше, чем баров в месяце.
    cfg.write_text(
        cfg.read_text(encoding="utf-8").replace("min_bars: 50", "min_bars: 100000"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no time slices"):
        run_experiment_from_config(cfg, data_root=data_root, results_root=results_root, workers=1)


# --- группировка ----------------------------------------------------------


def test_group_specs_by_params_groups_slices_together(tmp_path: Path) -> None:
    """Группа — все слайсы одной комбинации (гранулярность CONTINUOUS)."""
    from trading_research.domain import ExecutionModel, TimeSliceKind
    from trading_research.experiments.cache import build_run_specs
    from trading_research.experiments.time_splitters import generate_time_slices

    slices = generate_time_slices(
        TimeSliceKind.MONTH, T0, T0 + timedelta(days=90), timeframe="1h", min_bars=0
    )
    specs = build_run_specs(
        strategy="ma_cross",
        symbol="BTCUSDT",
        timeframe="1h",
        dataset_hash="deadbeef",
        execution=ExecutionModel(
            initial_balance=10_000,
            commission_rate=0.0006,
            slippage_pct=0.0003,
            position_size_value=20,
        ),
        param_combos=[{"fast_ma": 5, "slow_ma": 20}, {"fast_ma": 7, "slow_ma": 20}],
        slices=slices,
        git_commit="abc",
    )
    groups = group_specs_by_params(specs)
    assert len(groups) == 2
    assert all(len(g) == len(slices) for g in groups)
    # Внутри группы параметры одинаковы, слайсы различны.
    for g in groups:
        assert len({str(s.params) for s in g}) == 1
        assert len({s.slice_id for s in g}) == len(slices)
