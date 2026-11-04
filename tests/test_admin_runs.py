"""Tests for ADM-4 — read-model прогонов: откуда взялось число.

Каталог собирается руками (``save_run``), а таблица AN-1 строится из него
``performance_from_store`` — ровно так, как это делает ``run-experiment``.
Поэтому store и таблица согласованы по построению, и проверка «сколько прогонов
за строкой рейтинга» сравнивает два независимых пути к одному числу, а не
пересказ одного и того же списка.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from trading_research.admin import runs as R
from trading_research.analytics.queries import performance_from_store, save_performance_table
from trading_research.backtest.metrics import Metrics
from trading_research.backtest.orders import OrderReason, Side
from trading_research.backtest.portfolio import Trade
from trading_research.cli.analyze_results import experiment_results_path
from trading_research.cli.run_experiment import experiment_store_root, run_experiment_from_config
from trading_research.config import ExperimentConfig, load_experiment_config
from trading_research.domain import ExecutionModel, TimeSlice, TimeSliceKind
from trading_research.experiments.cache import RunSpec
from trading_research.experiments.results_store import ResultsStore, RunStatus
from trading_research.hashing import git_commit_hash

T0 = datetime(2024, 1, 1, tzinfo=UTC)
EXP = "runs_exp"
DATASET = "ded1ca7edda7a5e7"
COMMIT = "c0ffee" * 6 + "cafe"
PAIRS = (("BTCUSDT", "1h"), ("ETHUSDT", "1h"))
COMBOS: tuple[dict[str, Any], ...] = (
    {"fast_ma": 10, "slow_ma": 100},
    {"fast_ma": 20, "slow_ma": 100},
    {"fast_ma": 20, "slow_ma": 200},
)
SLICES = tuple(
    TimeSlice(
        id=f"2024-M{index + 1}",
        start=T0 + timedelta(days=31 * index),
        end=T0 + timedelta(days=31 * (index + 1)),
        label=f"2024-M{index + 1}",
        kind=TimeSliceKind.MONTH,
    )
    for index in range(4)
)

EXECUTION = ExecutionModel(
    initial_balance=10_000,
    commission_rate=0.0006,
    slippage_pct=0.0003,
    leverage=3,
    position_size_value=20,
)


# --- фикстуры ---------------------------------------------------------------


def _metrics(*, trades: int, total_return: float) -> Metrics:
    """Метрики прогона. ``reliable`` — то же правило, что в BT-7: сделок хватает."""
    return Metrics(
        total_return_pct=total_return,
        max_drawdown_pct=5.0,
        sharpe=total_return / 10,
        sortino=total_return / 8,
        profit_factor=1.5,
        win_rate=55.0,
        avg_trade_pct=0.4,
        expectancy=12.0,
        recovery_factor=1.1,
        trades_count=trades,
        annualization_factor=8760.0,
        min_trades=10,
        reliable=trades >= 10,
    )


def _equity(points: int = 600, flat: bool = False) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open_time": [T0 + timedelta(hours=i) for i in range(points)],
            "equity": [10_000.0 if flat else 10_000.0 + i for i in range(points)],
        }
    )


def _trade(index: int) -> Trade:
    return Trade(
        entry_time=T0 + timedelta(hours=index),
        exit_time=T0 + timedelta(hours=index + 1),
        side=Side.LONG,
        entry_price=100.0,
        exit_price=101.0,
        qty=1.0,
        pnl=1.0,
        pnl_pct=1.0,
        fees=0.1,
        reason=OrderReason.EXIT,
    )


def _spec(params: dict[str, Any], symbol: str, timeframe: str, slice_id: str) -> RunSpec:
    return RunSpec(
        strategy="ma_cross",
        params=dict(params),
        symbol=symbol,
        timeframe=timeframe,
        slice_id=slice_id,
        dataset_hash=DATASET,
        execution=EXECUTION,
        git_commit=COMMIT,
    )


def _trade_count(params: dict[str, Any], index: int) -> int:
    """Сколько сделок у прогона.

    Разложено так, чтобы ненадёжность не распределялась поровну: у
    ``slow_ma=200`` ненадёжны три прогона из четырёх в каждой паре, у остальных —
    один. Именно этим рейтинг и опасен: доля зависит от комбинации.
    """
    if params["slow_ma"] == 200:
        return 12 if index == 0 else 3
    return 3 if index == 0 else 12


def seed_experiment(root: Path, *, failed: bool = False) -> ResultsStore:
    """Наполнить каталог прогонами и записать таблицу AN-1 рядом с ним."""
    store = ResultsStore(experiment_store_root(root, EXP))
    for symbol, timeframe in PAIRS:
        for params in COMBOS:
            for index, time_slice in enumerate(SLICES):
                trades = _trade_count(params, index)
                store.save_run(
                    _spec(params, symbol, timeframe, time_slice.id),
                    _metrics(trades=trades, total_return=float(params["fast_ma"] + index)),
                    [_trade(i) for i in range(min(trades, 5))],
                    _equity(),
                )
    if failed:
        store.save_run(
            _spec({"fast_ma": 99, "slow_ma": 100}, "BTCUSDT", "1h", SLICES[0].id),
            None,
            [],
            pl.DataFrame(),
            status=RunStatus.FAILED,
            error="стратегия не нашла колонку close",
        )
    table = performance_from_store(store, list(SLICES), experiment_id=EXP)
    save_performance_table(table, experiment_results_path(root, EXP))
    return store


@pytest.fixture
def results(tmp_path: Path) -> Path:
    seed_experiment(tmp_path / "results")
    return tmp_path / "results"


@pytest.fixture
def table(results: Path) -> pl.DataFrame:
    loaded = R.performance_table(results, EXP)
    assert loaded is not None
    return loaded


@pytest.fixture
def store(results: Path) -> ResultsStore:
    opened = R.open_store(results, EXP)
    assert opened is not None
    return opened


def _config(path: Path) -> ExperimentConfig:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "name": EXP,
                "data": {
                    "exchange": "binance",
                    "market": "futures",
                    "symbols": [pair[0] for pair in PAIRS],
                    "timeframes": ["1h"],
                    "start": "2024-01-01",
                    "end": "2024-05-01",
                },
                "time_slices": {"type": "month", "min_bars": 50},
                "strategy": {"name": "ma_cross"},
                "params_grid": {"fast_ma": [10, 20], "slow_ma": [100, 200]},
                "execution": EXECUTION.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    return load_experiment_config(path)


# --- рейтинг: число и его происхождение -------------------------------------


def test_ranking_row_counts_match_the_runs_behind_it(
    table: pl.DataFrame, store: ResultsStore
) -> None:
    """Главное свойство страницы: «прогонов N» ведёт ровно к N прогонам.

    Два независимых пути — агрегат таблицы AN-1 и обход каталога — должны дать
    одно число. Расхождение значило бы, что ссылка «откуда» ведёт не туда.
    """
    rank = R.ranking(table, experiment=EXP)

    for row in rank.rows:
        behind = R.run_rows(store, param_hash_value=row.param_hash)
        assert row.runs == len(behind)
        assert row.unreliable == sum(1 for run in behind if run.reliable is False)


def test_ranking_marks_rows_held_up_by_unreliable_runs(table: pl.DataFrame) -> None:
    """Строка, где ненадёжных больше половины, помечена отдельно от доли.

    На сегодняшнем store рейтинг по ``sharpe`` возглавляет комбинация с 61
    ненадёжным прогоном из 114 — без пометки такая строка читается как лучшая.
    """
    rank = R.ranking(table, experiment=EXP)
    by_params = {row.param_hash: row for row in rank.rows}

    mostly = [row for row in by_params.values() if row.is_mostly_unreliable]
    assert [row.params["slow_ma"] for row in mostly] == [200]
    assert all(row.unreliable * 2 > row.runs for row in mostly)
    assert all(not row.is_mostly_unreliable for row in by_params.values() if row not in mostly)


def test_filters_narrow_the_ranking_but_not_the_experiment_share(table: pl.DataFrame) -> None:
    """Доля ненадёжных по эксперименту не теряется при фильтре по паре."""
    full = R.ranking(table, experiment=EXP)
    one_pair = R.ranking(table, experiment=EXP, symbol="BTCUSDT")

    assert one_pair.selected_runs == full.selected_runs // 2
    assert one_pair.total_runs == full.total_runs
    assert one_pair.total_unreliable == full.total_unreliable
    assert one_pair.is_filtered and not full.is_filtered


def test_ranking_is_concerning_when_a_quarter_of_runs_is_unreliable(
    table: pl.DataFrame,
) -> None:
    rank = R.ranking(table, experiment=EXP)

    assert rank.selected_share >= R.UNRELIABLE_ALARM
    assert rank.is_concerning


def test_ranking_of_an_empty_selection_is_empty_not_broken(table: pl.DataFrame) -> None:
    rank = R.ranking(table, experiment=EXP, symbol="NOPEUSDT")

    assert rank.is_empty
    assert rank.selected_runs == 0
    assert rank.selected_share == 0.0
    assert rank.total_runs == table.height


# --- устойчивость -----------------------------------------------------------


def test_stability_says_when_it_mixes_pairs(table: pl.DataFrame) -> None:
    """AN-3 требует одной пары; требование было в докстринге и нигде больше."""
    mixed = R.stability(table)
    single = R.stability(table, symbol="BTCUSDT", timeframe="1h")

    assert mixed.pairs == len(PAIRS)
    assert mixed.mixes_pairs
    assert single.pairs == 1
    assert not single.mixes_pairs


def test_stability_rows_count_table_rows_not_slices(table: pl.DataFrame) -> None:
    """``table_rows`` — строки таблицы, и при двух парах это вдвое больше слайсов."""
    single = R.stability(table, symbol="BTCUSDT", timeframe="1h")
    mixed = R.stability(table)

    assert {row.table_rows for row in single.rows} == {len(SLICES)}
    assert {row.table_rows for row in mixed.rows} == {len(SLICES) * len(PAIRS)}


# --- карта параметров -------------------------------------------------------


def test_heatmap_cells_carry_their_place_between_low_and_high(table: pl.DataFrame) -> None:
    card = R.heatmap(table, x="fast_ma", y="slow_ma")
    cells = [cell for row in card.rows for cell in row if not cell.is_empty]

    assert card.low is not None and card.high is not None
    assert min(cell.shade for cell in cells if cell.shade is not None) == 0.0
    assert max(cell.shade for cell in cells if cell.shade is not None) == 1.0
    assert all(0.0 <= cell.shade <= 1.0 for cell in cells if cell.shade is not None)


def test_heatmap_cell_without_data_has_no_value_and_no_shade(table: pl.DataFrame) -> None:
    """Сетка неполная: ``fast_ma=10`` считался только со ``slow_ma=100``."""
    card = R.heatmap(table, x="fast_ma", y="slow_ma")
    empty = [cell for row in card.rows for cell in row if cell.is_empty]

    assert empty, "в сетке фикстуры есть несчитанная комбинация"
    assert all(cell.shade is None for cell in empty)


def test_heatmap_of_one_value_puts_it_in_the_middle(table: pl.DataFrame) -> None:
    """Нулевой разброс не должен давать деления на ноль и «самой яркой» ячейки."""
    one = table.filter(
        (pl.col("param_hash") == R.ranking(table, experiment=EXP).rows[0].param_hash)
        & (pl.col("time_slice_id") == SLICES[0].id)
        & (pl.col("symbol") == "BTCUSDT")
    )

    card = R.heatmap(one, x="fast_ma", y="slow_ma")
    cells = [cell for row in card.rows for cell in row if not cell.is_empty]

    assert len(cells) == 1
    assert cells[0].shade == 0.5


def test_heatmap_labelled_rows_pair_each_row_with_its_axis_value(table: pl.DataFrame) -> None:
    card = R.heatmap(table, x="fast_ma", y="slow_ma")

    assert [label for label, _ in card.labelled_rows] == list(card.ys)
    assert all(all(cell.y == label for cell in cells) for label, cells in card.labelled_rows)


# --- прогоны ----------------------------------------------------------------


def test_run_rows_filter_by_the_values_a_heatmap_cell_knows(store: ResultsStore) -> None:
    """Ячейка карты задаёт две оси сетки, а не полный набор параметров."""
    cell = R.run_rows(store, params={"fast_ma": "20", "slow_ma": "200"})

    assert len(cell) == len(PAIRS) * len(SLICES)
    assert {(row.params["fast_ma"], row.params["slow_ma"]) for row in cell} == {(20, 200)}


def test_run_rows_filters_compose(store: ResultsStore) -> None:
    rows = R.run_rows(store, symbol="BTCUSDT", timeframe="1h", slice_id=SLICES[0].id)

    assert len(rows) == len(COMBOS)
    assert {row.symbol for row in rows} == {"BTCUSDT"}
    assert {row.slice_id for row in rows} == {SLICES[0].id}


def test_run_rows_select_only_unreliable(store: ResultsStore) -> None:
    unreliable = R.run_rows(store, reliable=False)
    reliable = R.run_rows(store, reliable=True)

    assert unreliable and reliable
    assert all(row.reliable is False for row in unreliable)
    assert len(unreliable) + len(reliable) == len(R.run_rows(store))


def test_runs_without_metrics_are_neither_reliable_nor_unreliable(tmp_path: Path) -> None:
    """Надёжность неизвестна — это третье состояние, а не «надёжен по умолчанию».

    Упавший прогон метрик не имеет; попасть в выборку «только надёжные» он не
    должен, иначе фильтр молча подмешивает в неё то, о чём ничего не известно.
    """
    seed_experiment(tmp_path / "results", failed=True)
    opened = R.open_store(tmp_path / "results", EXP)
    assert opened is not None

    reliable = R.run_rows(opened, reliable=True)
    unreliable = R.run_rows(opened, reliable=False)
    everything = R.run_rows(opened)

    assert all(row.metrics is not None for row in reliable)
    assert len(reliable) + len(unreliable) == len(everything) - 1


def test_failed_run_keeps_its_error_and_has_no_metrics(tmp_path: Path) -> None:
    """Прогон со статусом ``failed`` — это строка с объяснением, а не пропуск."""
    seed_experiment(tmp_path / "results", failed=True)
    opened = R.open_store(tmp_path / "results", EXP)
    assert opened is not None

    failed = R.run_rows(opened, status=RunStatus.FAILED.value)

    assert len(failed) == 1
    assert failed[0].failed
    assert failed[0].error == "стратегия не нашла колонку close"
    assert failed[0].metrics is None
    assert failed[0].reliable is None
    assert failed[0].value("sharpe") is None


# --- происхождение ----------------------------------------------------------


def test_identity_shown_on_the_page_reproduces_the_run_hash(
    store: ResultsStore, tmp_path: Path
) -> None:
    """Принцип №5 проверяется пересчётом, а не обещанием.

    Если показанного набора входов не хватает, пересчёт разойдётся — и это
    единственный способ отличить «воспроизводимо» от «записано достаточно полей».
    """
    cfg = _config(tmp_path / "cfg.json")
    row = R.run_rows(store)[0]

    detail = R.run_detail(
        store,
        row.run_hash,
        config_path=tmp_path / "cfg.json",
        data_root=tmp_path / "raw",
        results_root=tmp_path / "results",
        config=cfg,
        head_commit=COMMIT,
    )

    assert detail is not None
    assert detail.provenance.recomputed == row.run_hash
    assert detail.provenance.identity_matches is True


def test_identity_does_not_match_when_the_config_moved_on(
    store: ResultsStore, tmp_path: Path
) -> None:
    """Другая модель исполнения — другой ``run_hash``: конфиг разошёлся с прогоном."""
    cfg = _config(tmp_path / "cfg.json")
    moved = cfg.model_copy(
        update={"execution": EXECUTION.model_copy(update={"commission_rate": 0.001})}
    )
    row = R.run_rows(store)[0]

    detail = R.run_detail(
        store,
        row.run_hash,
        config_path=tmp_path / "cfg.json",
        data_root=tmp_path / "raw",
        results_root=tmp_path / "results",
        config=moved,
        head_commit=COMMIT,
    )

    assert detail is not None
    assert detail.provenance.identity_matches is False


def test_identity_is_not_guessed_without_a_config(store: ResultsStore, tmp_path: Path) -> None:
    """Модель исполнения живёт в конфиге; без него страница не выдумывает ответ."""
    row = R.run_rows(store)[0]

    detail = R.run_detail(
        store,
        row.run_hash,
        config_path=tmp_path / "cfg.json",
        data_root=tmp_path / "raw",
        results_root=tmp_path / "results",
        config=None,
        head_commit=COMMIT,
    )

    assert detail is not None
    assert detail.provenance.recomputed is None
    assert detail.provenance.identity_matches is None


def test_command_returns_to_the_commit_of_the_run(store: ResultsStore, tmp_path: Path) -> None:
    """``git_commit`` входит в ``run_hash``: без checkout получится другой ран."""
    row = R.run_rows(store)[0]
    kwargs = {
        "config_path": "cfg.yaml",
        "data_root": "data/raw",
        "results_root": "data/results",
    }

    changed = R.run_detail(store, row.run_hash, head_commit="0" * 40, **kwargs)
    same = R.run_detail(store, row.run_hash, head_commit=COMMIT, **kwargs)

    assert changed is not None and same is not None
    assert changed.provenance.code_changed
    assert changed.provenance.command[0] == f"git checkout {COMMIT}"
    assert not same.provenance.code_changed
    assert not any(line.startswith("git checkout") for line in same.provenance.command)
    assert same.provenance.command[-1].startswith("trading-research run-experiment --config")


def test_unknown_run_is_absent_not_empty(store: ResultsStore) -> None:
    assert (
        R.run_detail(
            store,
            "0" * 16,
            config_path="cfg.yaml",
            data_root="data/raw",
            results_root="data/results",
        )
        is None
    )


# --- артефакты прогона ------------------------------------------------------


def test_equity_is_thinned_but_keeps_its_ends(store: ResultsStore, tmp_path: Path) -> None:
    """Прореживание меняет размер разметки, а не форму кривой."""
    row = R.run_rows(store)[0]
    detail = R.run_detail(
        store,
        row.run_hash,
        config_path="cfg.yaml",
        data_root="data/raw",
        results_root="data/results",
    )
    assert detail is not None
    curve = detail.equity
    full = store.load_equity(row.run_hash)

    assert curve.total_points == full.height
    assert len(curve.points) <= R.EQUITY_POINTS + 1
    assert curve.points[0][1] == full.item(0, "equity")
    assert curve.points[-1][1] == full.item(-1, "equity")
    assert curve.low == full.get_column("equity").min()
    assert curve.high == full.get_column("equity").max()


def test_equity_shape_is_normalised_for_markup(store: ResultsStore) -> None:
    row = R.run_rows(store)[0]
    detail = R.run_detail(
        store,
        row.run_hash,
        config_path="cfg.yaml",
        data_root="data/raw",
        results_root="data/results",
    )
    assert detail is not None

    shape = detail.equity.shape()

    assert shape[0] == (0.0, 0.0)
    assert shape[-1] == (1.0, 1.0)
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in shape)


def test_flat_equity_does_not_divide_by_zero(tmp_path: Path) -> None:
    """Плоская кривая — не ошибка: она рисуется посередине."""
    curve = R.EquityCurve(
        points=tuple((T0 + timedelta(hours=i), 10_000.0) for i in range(5)),
        total_points=5,
        low=10_000.0,
        high=10_000.0,
    )

    assert [y for _, y in curve.shape()] == [0.5] * 5


def test_trades_table_is_capped_and_says_how_many_are_hidden(
    store: ResultsStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(R, "TRADES_SHOWN", 2)
    row = R.run_rows(store, reliable=True)[0]

    detail = R.run_detail(
        store,
        row.run_hash,
        config_path="cfg.yaml",
        data_root="data/raw",
        results_root="data/results",
    )

    assert detail is not None
    assert detail.trades is not None
    assert detail.trades.height == 2
    assert detail.trades_total == 5
    assert detail.trades_hidden == 3


# --- те ли сегодня данные ---------------------------------------------------


def _seed_candles(root: Path, *, closes: list[float]) -> None:
    from trading_research.data.schema import CANDLE_SCHEMA
    from trading_research.data.storage.parquet_store import ParquetStore

    times = [T0 + timedelta(hours=i) for i in range(len(closes))]
    ParquetStore(root).write_candles(
        pl.DataFrame(
            {
                "open_time": times,
                "open": closes,
                "high": [c + 1 for c in closes],
                "low": [c - 1 for c in closes],
                "close": closes,
                "volume": [10.0] * len(closes),
                "close_time": [t + timedelta(hours=1) for t in times],
                "quote_volume": [1000.0] * len(closes),
                "trades": [5] * len(closes),
                "taker_buy_base": [4.0] * len(closes),
                "taker_buy_quote": [400.0] * len(closes),
            },
            schema=CANDLE_SCHEMA,
        ),
        exchange="binance",
        market="futures",
        symbol="BTCUSDT",
        timeframe="1h",
    )


def test_data_check_confirms_the_candles_are_the_same(tmp_path: Path) -> None:
    """``dataset_hash`` входит в ``run_hash``: совпал — прогон воспроизводим."""
    from trading_research.data.catalog import dataset_hash
    from trading_research.data.storage.duckdb_store import DuckDBStore

    cfg = _config(tmp_path / "cfg.json")
    _seed_candles(tmp_path / "raw", closes=[100.0 + i % 5 for i in range(400)])
    stored = dataset_hash(
        DuckDBStore(tmp_path / "raw").read_window(
            exchange="binance",
            market="futures",
            symbol="BTCUSDT",
            timeframe="1h",
            start=cfg.data.start,
            end=cfg.data.end,
        )
    )

    check = R.check_data(cfg, stored, symbol="BTCUSDT", timeframe="1h", data_root=tmp_path / "raw")

    assert check.matches is True
    assert check.rows == 400


def test_data_check_catches_candles_that_changed_under_the_run(tmp_path: Path) -> None:
    """Сырьё обещано неизменным; расхождение здесь — повод разбираться."""
    cfg = _config(tmp_path / "cfg.json")
    _seed_candles(tmp_path / "raw", closes=[100.0 + i % 5 for i in range(400)])

    check = R.check_data(
        cfg, "не тот хэш", symbol="BTCUSDT", timeframe="1h", data_root=tmp_path / "raw"
    )

    assert check.matches is False
    assert check.today is not None


def test_data_check_without_candles_says_so_instead_of_failing(tmp_path: Path) -> None:
    cfg = _config(tmp_path / "cfg.json")

    check = R.check_data(cfg, DATASET, symbol="BTCUSDT", timeframe="1h", data_root=tmp_path / "raw")

    assert check.matches is None
    assert check.error


# --- границы read-model -----------------------------------------------------


def test_missing_table_is_none_not_an_exception(tmp_path: Path) -> None:
    """Каталог есть, таблицы AN-1 нет — страница должна объяснить, а не упасть."""
    ResultsStore(experiment_store_root(tmp_path / "results", EXP))

    assert R.performance_table(tmp_path / "results", EXP) is None
    assert R.open_store(tmp_path / "results", EXP) is not None


def test_opening_an_absent_experiment_creates_nothing(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()

    assert R.open_store(root, "нет-такого") is None
    assert list(root.iterdir()) == []


def test_param_names_come_from_the_table_not_the_config(table: pl.DataFrame) -> None:
    """Конфиг мог измениться; оси карты обязаны соответствовать посчитанному."""
    assert R.param_names(table) == ("fast_ma", "slow_ma")


def test_metric_names_are_the_queryable_whitelist() -> None:
    names = R.metric_names()

    assert "sharpe" in names
    assert "reliable" not in names, "булево поле агрегировать нечем"
    assert list(names) == sorted(names)


# --- команда воспроизведения: проверено запуском ----------------------------


def _pipeline_config(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            """
            schema_version: 2
            name: repro_exp
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
            output:
              save_trades: true
              save_equity_curve: true
            """
        ),
        encoding="utf-8",
    )
    return path


def test_printed_command_really_reproduces_the_run(tmp_path: Path) -> None:
    """Команда со страницы запускается и даёт тот же ``run_hash``.

    Проверяется запуском настоящего CLI: напечатанная строка — обещание, и
    единственный способ его проверить — выполнить ровно её. Второй прогон идёт в
    чистый ``--results-root``, иначе dedupe пропустил бы работу и проверка стала
    бы проверкой кэша.
    """
    binary = Path(sys.executable).parent / "trading-research"
    if not binary.exists():  # pragma: no cover — установка без console-script
        pytest.skip("консольная команда trading-research не установлена")

    cfg = _pipeline_config(tmp_path / "cfg.yaml")
    _seed_candles(tmp_path / "raw", closes=[100.0 + 10.0 * ((i // 12) % 2) for i in range(1500)])
    first = tmp_path / "results-1"
    run_experiment_from_config(cfg, data_root=tmp_path / "raw", results_root=first, workers=1)

    opened = R.open_store(first, "repro_exp")
    assert opened is not None
    row = R.run_rows(opened)[0]
    detail = R.run_detail(
        opened,
        row.run_hash,
        config_path=cfg,
        data_root=tmp_path / "raw",
        results_root=tmp_path / "results-2",
        config=load_experiment_config(cfg),
        head_commit=git_commit_hash(),
    )
    assert detail is not None
    assert detail.provenance.identity_matches is True

    command = shlex.split(detail.provenance.command[-1])
    assert command[0] == "trading-research"
    proc = subprocess.run([str(binary), *command[1:]], capture_output=True, text=True, check=False)

    assert proc.returncode == 0, proc.stderr
    again = R.open_store(tmp_path / "results-2", "repro_exp")
    assert again is not None
    assert row.run_hash in {run.run_hash for run in R.run_rows(again)}
