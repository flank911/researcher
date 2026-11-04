"""Tests for ADM-1 — read-model состояния платформы.

Всё на временном store и без сети. Отдельно закреплены четыре места, где состояние
легко соврало бы: часы (должны приходить снаружи), отсутствие данных (это состояние,
а не исключение), время проверки в отчёте (его отсутствие ≠ «проверено сейчас») и
запись на диск (read-model не имеет права ничего создавать).
"""

from __future__ import annotations

import json
import re
import textwrap
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

from trading_research.admin import runs as admin_runs
from trading_research.admin import state as admin_state
from trading_research.admin import web as admin_web
from trading_research.admin.state import (
    DatasetState,
    collect_state,
    dataset_states,
    experiment_names,
    experiment_state,
    schedule_state,
    validation_states,
    walk_forward_states,
)
from trading_research.backtest.metrics import compute_metrics
from trading_research.backtest.orders import OrderReason, Side
from trading_research.backtest.portfolio import Trade
from trading_research.cli.run_experiment import experiment_store_root
from trading_research.cli.validate_data import report_path, validate_data_from_config
from trading_research.cli.walk_forward import (
    HOLDOUT_REPORT_FILE,
    OOS_EQUITY_FILE,
    SELECTIONS_FILE,
    wf_dir,
)
from trading_research.config import load_experiment_config
from trading_research.data.ingestion.market_data_client import SAFE_RETENTION
from trading_research.data.schema import (
    CANDLE_SCHEMA,
    FUNDING_SCHEMA,
    LONG_SHORT_RATIO_SCHEMA,
    OPEN_INTEREST_SCHEMA,
)
from trading_research.data.storage.parquet_store import (
    CANDLES,
    FUNDING,
    LONG_SHORT_RATIO,
    OPEN_INTEREST,
    ParquetStore,
)
from trading_research.data.validators import load_report, save_report, validate_candles
from trading_research.domain import ExecutionModel, PositionSizeType
from trading_research.experiments.cache import RunSpec
from trading_research.experiments.results_store import ResultsStore, RunStatus

T0 = datetime(2024, 1, 1, tzinfo=UTC)
HOURS = 24 * 40
END = T0 + timedelta(hours=HOURS)
EXP = "admin_exp"

KEYS = {"exchange": "binance", "market": "futures", "symbol": "BTCUSDT", "timeframe": "1h"}


# --- фикстуры ---------------------------------------------------------------


def _candles(times: list[datetime]) -> pl.DataFrame:
    close = [100.0 + i % 7 for i in range(len(times))]
    return pl.DataFrame(
        {
            "open_time": times,
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [10.0] * len(times),
            "close_time": [t + timedelta(hours=1) for t in times],
            "quote_volume": [1000.0] * len(times),
            "trades": [5] * len(times),
            "taker_buy_base": [4.0] * len(times),
            "taker_buy_quote": [400.0] * len(times),
        },
        schema=CANDLE_SCHEMA,
    )


def _hours(n: int = HOURS) -> list[datetime]:
    return [T0 + timedelta(hours=i) for i in range(n)]


def _seed_data(root: Path, *, hours: int = HOURS) -> None:
    store = ParquetStore(root)
    times = _hours(hours)
    store.write_candles(_candles(times), **KEYS)
    funding_times = times[::8]
    store.write_series(
        FUNDING,
        pl.DataFrame(
            {"open_time": funding_times, "funding_rate": [-0.0002] * len(funding_times)},
            schema=FUNDING_SCHEMA,
        ),
        **KEYS,
    )
    store.write_series(
        OPEN_INTEREST,
        pl.DataFrame(
            {
                "open_time": times,
                "open_interest": [1000.0 + i for i in range(len(times))],
                "open_interest_value": [5e6] * len(times),
            },
            schema=OPEN_INTEREST_SCHEMA,
        ),
        **KEYS,
    )
    store.write_series(
        LONG_SHORT_RATIO,
        pl.DataFrame(
            {
                "open_time": times,
                "long_short_ratio": [1.5] * len(times),
                "long_account": [0.6] * len(times),
                "short_account": [0.4] * len(times),
            },
            schema=LONG_SHORT_RATIO_SCHEMA,
        ),
        **KEYS,
    )


def _config(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            f"""
            schema_version: 2
            name: {EXP}
            data:
              exchange: binance
              market: futures
              symbols: [BTCUSDT]
              timeframes: [1h]
              start: "{T0:%Y-%m-%d}"
              end: "{END:%Y-%m-%d}"
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


def _state(tmp_path: Path, *, now: datetime = END, log: Path | None = None) -> object:
    return collect_state(
        _config(tmp_path / "cfg.yaml"),
        data_root=tmp_path / "raw",
        results_root=tmp_path / "results",
        reports_root=tmp_path / "reports",
        log_path=log,
        now=now,
    )


def _dataset(state: object, dataset: str) -> DatasetState:
    return next(d for d in state.datasets if d.dataset == dataset)  # type: ignore[attr-defined]


# --- данные: отставание и запас --------------------------------------------


def test_lag_counted_from_passed_now(tmp_path: Path) -> None:
    """Часы приходят снаружи: тот же store, разный ``now`` — разное отставание."""
    _seed_data(tmp_path / "raw")
    cfg = load_experiment_config(_config(tmp_path / "cfg.yaml"))
    last = _hours()[-1]

    at_last = dataset_states(cfg, data_root=tmp_path / "raw", now=last)
    later = dataset_states(cfg, data_root=tmp_path / "raw", now=last + timedelta(hours=5))

    candles_now = next(d for d in at_last if d.dataset == CANDLES)
    candles_later = next(d for d in later if d.dataset == CANDLES)
    assert candles_now.lag == timedelta(0)
    assert candles_now.lag_periods == 0
    assert candles_later.lag == timedelta(hours=5)
    assert candles_later.lag_periods == 5


def test_reserve_only_for_series_with_retention(tmp_path: Path) -> None:
    """Запас до невосполнимой потери есть только у OI/LS."""
    _seed_data(tmp_path / "raw")
    state = _state(tmp_path, now=_hours()[-1])

    assert _dataset(state, CANDLES).reserve is None
    assert _dataset(state, FUNDING).reserve is None
    assert _dataset(state, OPEN_INTEREST).reserve == SAFE_RETENTION
    assert {d.dataset for d in state.perishable} == {OPEN_INTEREST, LONG_SHORT_RATIO}  # type: ignore[attr-defined]


def test_reserve_runs_out_and_is_named(tmp_path: Path) -> None:
    """Молчание дольше окна retention — потеря, а не отставание."""
    _seed_data(tmp_path / "raw")
    silent_for_a_month = _hours()[-1] + SAFE_RETENTION + timedelta(hours=1)
    state = _state(tmp_path, now=silent_for_a_month)

    oi = _dataset(state, OPEN_INTEREST)
    assert oi.reserve is not None and oi.reserve < timedelta(0)
    assert oi.is_exhausted is True
    assert _dataset(state, CANDLES).is_exhausted is False  # свечи догружаются всегда


def test_funding_period_comes_from_exchange(tmp_path: Path) -> None:
    """У funding шаг задаёт биржа, поэтому отставание в периодах не считается."""
    _seed_data(tmp_path / "raw")
    funding = _dataset(_state(tmp_path), FUNDING)

    assert funding.period is None
    assert funding.lag_periods is None
    assert funding.lag is not None


def test_empty_store_is_a_state_not_an_error(tmp_path: Path) -> None:
    state = _state(tmp_path)

    for dataset in (CANDLES, FUNDING, OPEN_INTEREST, LONG_SHORT_RATIO):
        empty = _dataset(state, dataset)
        assert empty.is_empty is True
        assert empty.rows == 0
        assert empty.last_point is None
        assert empty.lag is None
        assert empty.reserve is None
        assert empty.is_exhausted is False


# --- валидация: вердикт без даты -------------------------------------------


def test_report_is_dated_on_save(tmp_path: Path) -> None:
    """``save_report`` ставит время проверки, и оно читается обратно."""
    frame = _candles(_hours(50))
    report = validate_candles(frame, timeframe="1h")
    path = tmp_path / "report.json"
    save_report(report, path, checked_at=END)

    assert load_report(path).checked_at == END


def test_legacy_report_reads_as_undated(tmp_path: Path) -> None:
    """Отчёт, записанный до ADM-1, — «время неизвестно», а не «проверено сейчас»."""
    frame = _candles(_hours(50))
    legacy = validate_candles(frame, timeframe="1h").model_dump(mode="json")
    legacy.pop("checked_at")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    report = load_report(path)
    assert report.checked_at is None

    _seed_data(tmp_path / "raw")
    target = report_path(tmp_path / "reports" / EXP, "BTCUSDT", "1h", CANDLES)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(legacy), encoding="utf-8")

    candles = next(v for v in validation_states_of(tmp_path) if v.dataset == CANDLES)
    assert candles.exists is True
    assert candles.is_dated is False
    assert candles.age(END) is None


def validation_states_of(tmp_path: Path) -> list[object]:
    cfg = load_experiment_config(_config(tmp_path / "cfg.yaml"))
    return list(validation_states(cfg, reports_root=tmp_path / "reports"))


def test_validation_state_follows_validate_data(tmp_path: Path) -> None:
    """Read-model читает ровно те отчёты, которые пишет DATA-8."""
    _seed_data(tmp_path / "raw")
    validate_data_from_config(
        _config(tmp_path / "cfg.yaml"),
        data_root=tmp_path / "raw",
        reports_root=tmp_path / "reports",
        now=END,
    )
    state = _state(tmp_path)

    candles = next(v for v in state.validations if v.dataset == CANDLES)  # type: ignore[attr-defined]
    assert candles.exists is True
    assert candles.checked_at == END
    assert candles.is_clean is True
    assert candles.rows == HOURS
    assert candles.age(END + timedelta(days=3)) == timedelta(days=3)
    assert state.defects == ()  # type: ignore[attr-defined]


def test_missing_report_is_a_state(tmp_path: Path) -> None:
    _seed_data(tmp_path / "raw")
    state = _state(tmp_path)

    candles = next(v for v in state.validations if v.dataset == CANDLES)  # type: ignore[attr-defined]
    assert candles.exists is False
    assert candles.is_clean is None
    assert candles.checked_at is None


def test_unreadable_report_is_a_state(tmp_path: Path) -> None:
    _seed_data(tmp_path / "raw")
    target = report_path(tmp_path / "reports" / EXP, "BTCUSDT", "1h", CANDLES)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"nonsense": 1}', encoding="utf-8")

    candles = next(v for v in validation_states_of(tmp_path) if v.dataset == CANDLES)
    assert candles.exists is True
    assert candles.error is not None
    assert candles.is_clean is None


# --- расписание -------------------------------------------------------------


def _log(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    return path


def test_schedule_reads_last_iteration(tmp_path: Path) -> None:
    log = _log(
        tmp_path / "update.log",
        """
        === 2026-11-02 06:00:01 UTC — cfg.yaml
        итого: 0 точек по 18 рядам, пропусков нет
        --- exit=0
        === 2026-11-02 12:00:02 UTC — cfg.yaml
        итого: 48 точек по 18 рядам, пропусков нет
        --- exit=1
        """,
    )
    state = schedule_state(log, now=datetime(2026, 11, 2, 18, 0, tzinfo=UTC))

    assert state.exists is True
    assert state.last_iteration_at == datetime(2026, 11, 2, 12, 0, 2, tzinfo=UTC)
    assert state.last_exit_status == 1
    assert state.last_iteration_failed is True
    assert state.silence == timedelta(hours=5, minutes=59, seconds=58)


def test_iteration_in_progress_has_no_exit_status(tmp_path: Path) -> None:
    """Итерация началась и не кончилась — это не «успешно» и не прошлый код."""
    log = _log(
        tmp_path / "update.log",
        """
        === 2026-11-02 06:00:01 UTC — cfg.yaml
        --- exit=0
        === 2026-11-02 12:00:02 UTC — cfg.yaml
        """,
    )
    state = schedule_state(log, now=datetime(2026, 11, 2, 12, 1, tzinfo=UTC))

    assert state.last_iteration_at == datetime(2026, 11, 2, 12, 0, 2, tzinfo=UTC)
    assert state.last_exit_status is None
    assert state.last_iteration_failed is False


def test_missing_and_absent_log(tmp_path: Path) -> None:
    now = datetime(2026, 11, 2, tzinfo=UTC)
    missing = schedule_state(tmp_path / "nope.log", now=now)
    none = schedule_state(None, now=now)

    assert missing.exists is False
    assert missing.silence is None
    assert none.exists is False
    assert none.log_path is None


def test_garbage_log_lines_are_ignored(tmp_path: Path) -> None:
    log = _log(
        tmp_path / "update.log",
        """
        === не время вовсе — cfg.yaml
        --- exit=не число
        """,
    )
    state = schedule_state(log, now=datetime(2026, 11, 2, tzinfo=UTC))

    assert state.exists is True
    assert state.last_iteration_at is None
    assert state.last_exit_status is None


def test_only_log_tail_is_read(tmp_path: Path) -> None:
    """Лог append-only: читается хвост, а не весь файл."""
    filler = "итого: 48 точек по 18 рядам, пропусков нет\n" * 4000
    log = tmp_path / "update.log"
    log.write_text(
        filler + "=== 2026-11-02 12:00:02 UTC — cfg.yaml\n--- exit=0\n", encoding="utf-8"
    )
    assert log.stat().st_size > 65_536

    state = schedule_state(log, now=datetime(2026, 11, 2, 13, 0, tzinfo=UTC))
    assert state.last_iteration_at == datetime(2026, 11, 2, 12, 0, 2, tzinfo=UTC)


# --- результаты -------------------------------------------------------------


def _execution() -> ExecutionModel:
    return ExecutionModel(
        initial_balance=10_000.0,
        commission_rate=0.0006,
        slippage_pct=0.0003,
        leverage=3,
        position_size_type=PositionSizeType.PERCENT_BALANCE,
        position_size_value=20.0,
    )


def _spec(slice_id: str, **overrides: object) -> RunSpec:
    base: dict[str, object] = {
        "strategy": "ma_cross",
        "params": {"fast_ma": 20, "slow_ma": 100},
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "slice_id": slice_id,
        "dataset_hash": "abc123",
        "execution": _execution(),
        "git_commit": "deadbeef",
    }
    base.update(overrides)
    return RunSpec(**base)  # type: ignore[arg-type]


def _trade(n: int) -> Trade:
    return Trade(
        entry_time=T0 + timedelta(hours=n),
        exit_time=T0 + timedelta(hours=n + 1),
        side=Side.LONG,
        entry_price=100.0,
        exit_price=101.0,
        qty=1.0,
        pnl=1.0,
        pnl_pct=1.0,
        fees=0.1,
        reason=OrderReason.REVERSE,
    )


def _equity() -> pl.DataFrame:
    times = [T0 + timedelta(hours=i) for i in range(5)]
    return pl.DataFrame(
        {"open_time": times, "equity": [10000.0, 10010.0, 10015.0, 10012.0, 10022.5]}
    )


def _seed_results(root: Path, *, trades_per_run: tuple[int, ...]) -> ResultsStore:
    store = ResultsStore(experiment_store_root(root, EXP))
    for i, count in enumerate(trades_per_run):
        trades = [_trade(n) for n in range(count)]
        metrics = compute_metrics(_equity(), trades, "1h", min_trades=3)
        store.save_run(_spec(f"2024-{i + 1:02d}"), metrics, trades, _equity())
    return store


def test_experiment_state_counts_unreliable_runs(tmp_path: Path) -> None:
    """Доля ненадёжных — от всех прогонов, а не от удачных."""
    _seed_results(tmp_path / "results", trades_per_run=(5, 1, 1))
    state = experiment_state(tmp_path / "results", EXP)

    assert state is not None
    assert state.runs == 3
    assert state.by_status == {RunStatus.COMPLETED.value: 3}
    assert state.unreliable == 2
    assert state.unreliable_share == pytest.approx(2 / 3)
    assert state.without_metrics == 0
    assert state.strategies == ("ma_cross",)
    assert state.pairs == (("BTCUSDT", "1h"),)
    assert state.slices == 3
    assert state.param_sets == 1
    assert state.git_commits == ("deadbeef",)
    assert state.dataset_hashes == ("abc123",)


def test_experiment_state_knows_when_runs_happened(tmp_path: Path) -> None:
    """«Последний прогон был давно» — это ответ, которого раньше не было."""
    _seed_results(tmp_path / "results", trades_per_run=(5,))
    state = experiment_state(tmp_path / "results", EXP)

    assert state is not None
    assert state.first_run_at is not None
    assert state.last_run_at is not None
    assert state.last_run_at >= state.first_run_at


def test_unreliable_share_counts_runs_without_metrics(tmp_path: Path) -> None:
    """Знаменатель — все прогоны. Иначе упавший ран улучшал бы долю надёжных."""
    store = _seed_results(tmp_path / "results", trades_per_run=(1, 1))
    store.save_run(_spec("2024-09"), None, [], _equity(), status=RunStatus.FAILED, error="boom")
    state = experiment_state(tmp_path / "results", EXP)

    assert state is not None
    assert state.runs == 3
    assert state.unreliable == 2
    assert state.without_metrics == 1
    assert state.unreliable_share == pytest.approx(2 / 3)


def test_failed_run_is_visible(tmp_path: Path) -> None:
    store = ResultsStore(experiment_store_root(tmp_path / "results", EXP))
    store.save_run(
        _spec("2024-01"),
        None,
        [],
        _equity(),
        status=RunStatus.FAILED,
        error="boom",
    )
    state = experiment_state(tmp_path / "results", EXP)

    assert state is not None
    assert state.by_status == {RunStatus.FAILED.value: 1}
    assert state.without_metrics == 1
    assert state.unreliable == 0  # метрик нет — и ненадёжным такой ран не считается


def test_missing_experiment_is_none_and_creates_nothing(tmp_path: Path) -> None:
    """Взгляд на эксперимент, которого нет, не создаёт пустой эксперимент."""
    results = tmp_path / "results"
    results.mkdir()

    assert experiment_state(results, "nope") is None
    assert list(results.iterdir()) == []
    assert experiment_names(results) == []


def test_experiment_names_ignores_folders_without_catalog(tmp_path: Path) -> None:
    results = tmp_path / "results"
    (results / "just_a_folder").mkdir(parents=True)
    _seed_results(results, trades_per_run=(5,))

    assert experiment_names(results) == [EXP]


# --- walk-forward -----------------------------------------------------------


def test_walk_forward_state_counts_folds_without_reading_holdout(tmp_path: Path) -> None:
    """Holdout считается существующим, но не читается: раскрытие резерва — ADM-5."""
    out = wf_dir(tmp_path / "results", EXP, "BTCUSDT", "1h")
    out.mkdir(parents=True)
    (out / SELECTIONS_FILE).write_text(
        json.dumps({"selections": [{"fold": 0}, {"fold": 1}], "oos_metrics": {}}),
        encoding="utf-8",
    )
    (out / OOS_EQUITY_FILE).write_text("не parquet, но файл есть", encoding="utf-8")
    (out / HOLDOUT_REPORT_FILE).write_text(json.dumps({"holdout_value": -1.75}), encoding="utf-8")

    cfg = load_experiment_config(_config(tmp_path / "cfg.yaml"))
    state = walk_forward_states(cfg, results_root=tmp_path / "results")[0]

    assert state.exists is True
    assert state.folds == 2
    assert state.has_oos_equity is True
    assert state.has_holdout is True
    assert "holdout_value" not in str(state)  # значения резерва в состоянии нет


def test_walk_forward_absent_is_a_state(tmp_path: Path) -> None:
    cfg = load_experiment_config(_config(tmp_path / "cfg.yaml"))
    state = walk_forward_states(cfg, results_root=tmp_path / "results")[0]

    assert state.exists is False
    assert state.folds == 0
    assert state.has_holdout is False


def test_broken_selections_file_is_a_state(tmp_path: Path) -> None:
    out = wf_dir(tmp_path / "results", EXP, "BTCUSDT", "1h")
    out.mkdir(parents=True)
    (out / SELECTIONS_FILE).write_text("{ не json", encoding="utf-8")

    cfg = load_experiment_config(_config(tmp_path / "cfg.yaml"))
    state = walk_forward_states(cfg, results_root=tmp_path / "results")[0]

    assert state.exists is True
    assert state.folds == 0
    assert state.error is not None


# --- границы read-model -----------------------------------------------------


def test_collect_state_never_touches_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Состояние читается с диска: ни один запрос в сеть не допустим."""

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("read-model не имеет права ходить в сеть")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    _seed_data(tmp_path / "raw")
    _seed_results(tmp_path / "results", trades_per_run=(5,))

    state = _state(tmp_path)
    assert state.experiment == EXP  # type: ignore[attr-defined]


def test_collect_state_writes_nothing(tmp_path: Path) -> None:
    """Read-model ничего не создаёт — ни каталогов, ни баз."""
    _seed_data(tmp_path / "raw")
    _config(tmp_path / "cfg.yaml")
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    _state(tmp_path)

    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("module", [admin_state, admin_runs, admin_web])
def test_admin_reads_only_through_stores(module: ModuleType) -> None:
    """В пакете нет своего доступа к данным: ни SQL, ни чтения parquet, ни путей.

    Это и есть защита от третьей реализации чтения: расхождение «в панели одно
    число, в CLI другое» иначе проявилось бы молча. Запрещены именно **свои**
    движки и свои формулы путей — ``DuckDBStore`` и ``ResultsStore`` для того и
    существуют, чтобы через них читали все.

    Проверка распространена на весь пакет в ADM-4: страницы прогонов — ровно то
    место, где соблазн сходить в каталог своим SQL сильнее всего.
    """
    source = Path(module.__file__ or "").read_text(encoding="utf-8")
    body = source.split('"""', 2)[2]  # без докстринга модуля, где эти слова упомянуты

    engines = re.findall(r"(?m)^\s*(?:import|from)\s+(duckdb|sqlite3|pyarrow)\b", body)
    assert engines == [], f"{module.__name__} открывает хранилище сам: {engines}"
    for forbidden in (
        "read_parquet",
        "catalog.sqlite",
        "performance.parquet",
        "trades.parquet",
        "equity.parquet",
    ):
        assert forbidden not in body, f"{module.__name__} знает свой путь к данным: {forbidden}"


def test_results_store_read_only_mode_creates_nothing(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        ResultsStore(missing, create=False)

    assert not missing.exists()


def test_read_only_store_touches_nothing_on_existing_root(tmp_path: Path) -> None:
    """На существующем каталоге ``create=False`` тоже ничего не создаёт.

    Иначе read-model переписывал бы схему чужой базы на каждый взгляд.
    """
    root = tmp_path / "exp"
    root.mkdir()
    (root / "catalog.sqlite").write_text("", encoding="utf-8")

    ResultsStore(root, create=False)

    assert sorted(p.name for p in root.iterdir()) == ["catalog.sqlite"]
