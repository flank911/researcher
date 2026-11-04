"""Tests for ADM-2 — команда ``status``.

Состояние читает ADM-1, поэтому здесь проверяется ровно то, что добавляет эта
команда: пороги, по которым состояние превращается в поводы вмешаться, и код
возврата. Всё на временном store и без сети.
"""

from __future__ import annotations

import json
import textwrap
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from trading_research.backtest.metrics import compute_metrics
from trading_research.backtest.orders import OrderReason, Side
from trading_research.backtest.portfolio import Trade
from trading_research.cli.main import app
from trading_research.cli.run_experiment import experiment_store_root
from trading_research.cli.status import (
    RESERVE_FLOOR,
    STALE_PERIODS,
    StatusReport,
    describe_report,
    status_from_config,
    status_payload,
)
from trading_research.cli.validate_data import validate_data_from_config
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
from trading_research.domain import ExecutionModel, PositionSizeType
from trading_research.experiments.cache import RunSpec
from trading_research.experiments.results_store import ResultsStore, RunStatus

runner = CliRunner()

T0 = datetime(2024, 1, 1, tzinfo=UTC)
HOURS = 24 * 40
END = T0 + timedelta(hours=HOURS)
LAST_BAR = T0 + timedelta(hours=HOURS - 1)
EXP = "status_exp"

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


def _seed(
    root: Path,
    *,
    candles: bool = True,
    funding: bool = True,
    periodic: bool = True,
    periodic_until: datetime | None = None,
) -> None:
    """Наполнить store. ``periodic_until`` — где обрываются OI/LS (окно retention)."""
    store = ParquetStore(root)
    times = _hours()
    if candles:
        store.write_candles(_candles(times), **KEYS)
    if funding:
        funding_times = times[::8]
        store.write_series(
            FUNDING,
            pl.DataFrame(
                {"open_time": funding_times, "funding_rate": [-0.0002] * len(funding_times)},
                schema=FUNDING_SCHEMA,
            ),
            **KEYS,
        )
    if periodic:
        points = [t for t in times if periodic_until is None or t <= periodic_until]
        store.write_series(
            OPEN_INTEREST,
            pl.DataFrame(
                {
                    "open_time": points,
                    "open_interest": [1000.0 + i for i in range(len(points))],
                    "open_interest_value": [5e6] * len(points),
                },
                schema=OPEN_INTEREST_SCHEMA,
            ),
            **KEYS,
        )
        store.write_series(
            LONG_SHORT_RATIO,
            pl.DataFrame(
                {
                    "open_time": points,
                    "long_short_ratio": [1.5] * len(points),
                    "long_account": [0.6] * len(points),
                    "short_account": [0.4] * len(points),
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


def _report(tmp_path: Path, *, now: datetime = LAST_BAR, log: Path | None = None) -> StatusReport:
    return status_from_config(
        _config(tmp_path / "cfg.yaml"),
        data_root=tmp_path / "raw",
        results_root=tmp_path / "results",
        reports_root=tmp_path / "reports",
        log_path=log,
        now=now,
    )


def _validate(tmp_path: Path, *, now: datetime = LAST_BAR) -> None:
    validate_data_from_config(
        _config(tmp_path / "cfg.yaml"),
        data_root=tmp_path / "raw",
        reports_root=tmp_path / "reports",
        now=now,
    )


def _messages(report: StatusReport, block: str) -> list[str]:
    return [a.message for a in report.alerts if a.block == block]


# --- чистое состояние -------------------------------------------------------


def test_fresh_store_needs_no_attention(tmp_path: Path) -> None:
    _seed(tmp_path / "raw")
    _validate(tmp_path)
    report = _report(tmp_path)

    assert report.alerts == ()
    assert report.needs_attention is False
    assert describe_report(report)[-1] == "итого: вмешательство не требуется"


# --- отставание -------------------------------------------------------------


def test_candles_lag_becomes_an_alert_past_the_threshold(tmp_path: Path) -> None:
    """Один-два периода — норма (бар мог не закрыться), выше порога — повод."""
    _seed(tmp_path / "raw")
    _validate(tmp_path)

    at_threshold = _report(tmp_path, now=LAST_BAR + STALE_PERIODS * timedelta(hours=1))
    past_threshold = _report(tmp_path, now=LAST_BAR + (STALE_PERIODS + 1) * timedelta(hours=1))

    assert _messages(at_threshold, "данные") == []
    assert any("candles — отстаёт" in m for m in _messages(past_threshold, "данные"))


def test_periodic_series_lag_is_not_an_alert(tmp_path: Path) -> None:
    """OI/LS собираются раз в шесть часов и отстают по построению — это не повод.

    Замер на живом store: 7 периодов отставания при живом сборе и ``exit=0``.
    """
    _seed(tmp_path / "raw")
    _validate(tmp_path)
    report = _report(tmp_path, now=LAST_BAR + timedelta(hours=10))

    lag_alerts = [m for m in _messages(report, "данные") if "отстаёт" in m]
    assert all(CANDLES in m for m in lag_alerts)
    assert not any(OPEN_INTEREST in m or LONG_SHORT_RATIO in m for m in lag_alerts)


def test_funding_silence_is_measured_in_time(tmp_path: Path) -> None:
    """У funding период задаёт биржа, поэтому порог — сутки, а не периоды."""
    _seed(tmp_path / "raw", candles=False, periodic=False)
    last_funding = _hours()[::8][-1]

    quiet = _report(tmp_path, now=last_funding + timedelta(hours=20))
    silent = _report(tmp_path, now=last_funding + timedelta(hours=30))

    assert not any(FUNDING in m for m in _messages(quiet, "данные"))
    assert any(f"{FUNDING} — нет новых точек" in m for m in _messages(silent, "данные"))


# --- запас по невосполнимым рядам ------------------------------------------


def test_reserve_below_floor_is_an_alert(tmp_path: Path) -> None:
    _seed(tmp_path / "raw")
    last = _hours()[-1]
    report = _report(tmp_path, now=last + RESERVE_FLOOR + timedelta(days=16))

    messages = _messages(report, "данные")
    assert any("запас до потери истории" in m and OPEN_INTEREST in m for m in messages)
    assert any("запас до потери истории" in m and LONG_SHORT_RATIO in m for m in messages)


def test_exhausted_reserve_is_named_irreplaceable(tmp_path: Path) -> None:
    """Потерянную историю OI/LS не восстановить — она не «отстаёт», она потеряна."""
    _seed(tmp_path / "raw")
    report = _report(tmp_path, now=_hours()[-1] + timedelta(days=40))

    assert any("НЕВОСПОЛНИМЫЙ ПРОПУСК" in m for m in _messages(report, "данные"))


# --- отсутствие данных ------------------------------------------------------


def test_missing_candles_are_an_alert(tmp_path: Path) -> None:
    _seed(tmp_path / "raw", candles=False)
    report = _report(tmp_path)

    assert any("свечей нет" in m for m in _messages(report, "данные"))


def test_missing_optional_series_are_not_an_alert(tmp_path: Path) -> None:
    """funding/OI/LS опциональны — так же, как на них смотрит validate-data."""
    _seed(tmp_path / "raw", funding=False, periodic=False)
    _validate(tmp_path)
    report = _report(tmp_path)

    assert _messages(report, "данные") == []


# --- валидация --------------------------------------------------------------


def test_absent_reports_are_an_alert(tmp_path: Path) -> None:
    """Нет вердикта о качестве — это не то же самое, что «дефектов нет»."""
    _seed(tmp_path / "raw")
    report = _report(tmp_path)

    assert report.validation.missing == 4
    assert any("нет вердикта о качестве" in m for m in _messages(report, "валидация"))


def test_undated_report_is_an_alert(tmp_path: Path) -> None:
    """Вердикт без даты не позволяет сказать, покрывает ли он текущие данные."""
    _seed(tmp_path / "raw")
    _validate(tmp_path)
    for path in (tmp_path / "reports" / EXP).glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("checked_at")
        path.write_text(json.dumps(payload), encoding="utf-8")

    report = _report(tmp_path)
    assert report.validation.undated == 4
    assert any("вердикт без даты" in m for m in _messages(report, "валидация"))


def test_report_older_than_data_is_an_alert(tmp_path: Path) -> None:
    """Проверка, сделанная до последней точки ряда, этой точки не видела."""
    _seed(tmp_path / "raw", periodic=False, funding=False)
    _validate(tmp_path, now=T0 + timedelta(hours=10))

    report = _report(tmp_path)
    assert report.validation.stale == 1
    assert any("вердикт старше данных" in m for m in _messages(report, "валидация"))


def test_defects_are_an_alert(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "raw")
    with_gap = [t for i, t in enumerate(_hours()) if i not in range(10, 20)]
    store.write_candles(_candles(with_gap), **KEYS)
    _validate(tmp_path)

    report = _report(tmp_path)
    assert report.validation.defects == 1
    assert any("дефекты в данных" in m for m in _messages(report, "валидация"))


# --- расписание -------------------------------------------------------------


def _log(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    return path


def test_failed_iteration_is_an_alert(tmp_path: Path) -> None:
    log = _log(
        tmp_path / "update.log",
        """
        === 2024-02-09 06:00:01 UTC — cfg.yaml
        --- exit=1
        """,
    )
    _seed(tmp_path / "raw")
    _validate(tmp_path)
    report = _report(tmp_path, now=datetime(2024, 2, 9, 7, tzinfo=UTC), log=log)

    assert any("кончилась кодом 1" in m for m in _messages(report, "расписание"))


def test_silent_schedule_is_an_alert(tmp_path: Path) -> None:
    log = _log(
        tmp_path / "update.log",
        """
        === 2024-02-09 06:00:01 UTC — cfg.yaml
        --- exit=0
        """,
    )
    _seed(tmp_path / "raw")
    report = _report(tmp_path, now=datetime(2024, 2, 11, 6, tzinfo=UTC), log=log)

    assert any("сбор молчит" in m for m in _messages(report, "расписание"))


def test_absent_log_is_not_an_alert(tmp_path: Path) -> None:
    """Расписание может быть не поднято на этой машине — это решение владельца."""
    _seed(tmp_path / "raw")
    _validate(tmp_path)
    report = _report(tmp_path, log=tmp_path / "nope.log")

    assert _messages(report, "расписание") == []
    assert "расписание на этой машине не поднято" in "\n".join(describe_report(report))


# --- эксперименты -----------------------------------------------------------


def _execution() -> ExecutionModel:
    return ExecutionModel(
        initial_balance=10_000.0,
        commission_rate=0.0006,
        slippage_pct=0.0003,
        leverage=3,
        position_size_type=PositionSizeType.PERCENT_BALANCE,
        position_size_value=20.0,
    )


def _spec(slice_id: str) -> RunSpec:
    return RunSpec(
        strategy="ma_cross",
        params={"fast_ma": 5, "slow_ma": 20},
        symbol="BTCUSDT",
        timeframe="1h",
        slice_id=slice_id,
        dataset_hash="abc123",
        execution=_execution(),
        git_commit="deadbeef",
    )


def _equity() -> pl.DataFrame:
    times = [T0 + timedelta(hours=i) for i in range(5)]
    return pl.DataFrame(
        {"open_time": times, "equity": [10000.0, 10010.0, 10015.0, 10012.0, 10022.5]}
    )


def _trades(count: int) -> list[Trade]:
    return [
        Trade(
            entry_time=T0 + timedelta(hours=i),
            exit_time=T0 + timedelta(hours=i + 1),
            side=Side.LONG,
            entry_price=100.0,
            exit_price=101.0,
            qty=1.0,
            pnl=1.0,
            pnl_pct=1.0,
            fees=0.1,
            reason=OrderReason.REVERSE,
        )
        for i in range(count)
    ]


def test_unfinished_runs_are_an_alert(tmp_path: Path) -> None:
    store = ResultsStore(experiment_store_root(tmp_path / "results", EXP))
    store.save_run(_spec("2024-01"), None, [], _equity(), status=RunStatus.FAILED, error="boom")
    _seed(tmp_path / "raw")
    _validate(tmp_path)

    report = _report(tmp_path)
    assert _messages(report, "эксперименты") == ["прогонов в статусе failed: 1"]


def test_absent_experiment_is_not_an_alert(tmp_path: Path) -> None:
    """«Прогонов ещё не было» — состояние платформы, а не её неисправность."""
    _seed(tmp_path / "raw")
    _validate(tmp_path)
    report = _report(tmp_path)

    assert _messages(report, "эксперименты") == []
    assert "прогонов ещё не было" in "\n".join(describe_report(report))


def test_unreliable_share_is_printed(tmp_path: Path) -> None:
    """Доля ненадёжных прогонов видна там же, где их число (AN-2 её не показывает)."""
    store = ResultsStore(experiment_store_root(tmp_path / "results", EXP))
    for i, count in enumerate((5, 1)):
        trades = _trades(count)
        store.save_run(
            _spec(f"2024-{i + 1:02d}"),
            compute_metrics(_equity(), trades, "1h", min_trades=3),
            trades,
            _equity(),
        )
    _seed(tmp_path / "raw")

    text = "\n".join(describe_report(_report(tmp_path)))
    assert "ненадёжных по числу сделок 1 (50%)" in text


# --- вывод и код возврата ---------------------------------------------------


def test_json_carries_the_same_numbers_as_text(tmp_path: Path) -> None:
    """Одна правда, два вида: --json не считает пороги заново."""
    _seed(tmp_path / "raw")
    report = _report(tmp_path, now=LAST_BAR + timedelta(days=4))
    payload = status_payload(report)

    assert payload["needs_attention"] is report.needs_attention
    assert payload["experiment"] == EXP
    assert len(payload["alerts"]) == len(report.alerts)  # type: ignore[arg-type]
    assert len(payload["datasets"]) == len(report.state.datasets)  # type: ignore[arg-type]
    assert payload["validation"] == {  # type: ignore[comparison-overlap]
        "total": 4,
        "present": 0,
        "missing": 4,
        "defects": 0,
        "undated": 0,
        "stale": 0,
    }
    assert json.loads(json.dumps(payload, ensure_ascii=False))["experiment"] == EXP


def test_cli_strict_exit_code(tmp_path: Path) -> None:
    """Без --strict — диагностика (код 0), с ним команду можно ставить в пайплайн."""
    _seed(tmp_path / "raw")
    args = [
        "status",
        "--config",
        str(_config(tmp_path / "cfg.yaml")),
        "--data-root",
        str(tmp_path / "raw"),
        "--results-root",
        str(tmp_path / "results"),
        "--reports-root",
        str(tmp_path / "reports"),
        "--log",
        str(tmp_path / "nope.log"),
    ]

    plain = runner.invoke(app, args)
    strict = runner.invoke(app, [*args, "--strict"])

    assert plain.exit_code == 0
    assert "Внимание" in plain.stdout
    assert strict.exit_code == 1


def test_cli_json_is_machine_readable(tmp_path: Path) -> None:
    _seed(tmp_path / "raw")
    result = runner.invoke(
        app,
        [
            "status",
            "--config",
            str(_config(tmp_path / "cfg.yaml")),
            "--data-root",
            str(tmp_path / "raw"),
            "--results-root",
            str(tmp_path / "results"),
            "--reports-root",
            str(tmp_path / "reports"),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["experiment"] == EXP
    assert payload["needs_attention"] is True


def test_status_never_touches_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("status не имеет права ходить в сеть")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    _seed(tmp_path / "raw")

    assert _report(tmp_path).state.experiment == EXP
