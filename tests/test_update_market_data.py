"""Tests for OPS-1 — инкрементальная догрузка рыночных рядов (без сети).

Главное, что здесь проверяется, — не «догрузилось», а **видно ли потерянное**.
Окно доступности OI/LS у Binance ~30 дней, поэтому пропущенная итерация сбора
означает дыру, которую уже нечем закрыть; тихий успех в этом месте обнаружился бы
через месяц, когда данные потеряны.

Время задаётся через ``now=``, а не берётся от часов: тест про окно retention,
привязанный к реальному «сейчас», начал бы падать сам по себе.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from trading_research.cli.fetch_market_data import CoverageStatus, SeriesKind
from trading_research.cli.main import app
from trading_research.cli.update_market_data import (
    SAFE_RETENTION,
    SeriesUpdate,
    describe_total,
    describe_update,
    has_defects,
    update_market_data_from_config,
)
from trading_research.data.schema import (
    LONG_SHORT_RATIO_SCHEMA,
    OPEN_INTEREST_SCHEMA,
)
from trading_research.data.storage.parquet_store import (
    FUNDING,
    LONG_SHORT_RATIO,
    OPEN_INTEREST,
    ParquetStore,
)

runner = CliRunner()

NOW = datetime(2026, 10, 31, 12, 30, tzinfo=UTC)
# Начало текущего часа: правая граница окна и точка отсчёта для сидов. NOW
# намеренно не выровнен — на нём проверяется, что граница округляется вниз.
TOP = NOW.replace(minute=0)
CONFIG_START = datetime(2022, 1, 1, tzinfo=UTC)
H = timedelta(hours=1)
_KEYS = {"exchange": "binance", "market": "futures", "symbol": "BTCUSDT", "timeframe": "1h"}


def _config(tmp_path: Path, *, symbols: list[str] | None = None, timeframes: list[str]) -> Path:
    """Минимальный валидный конфиг эксперимента — источник набора пар для сбора."""
    payload = {
        "schema_version": 2,
        "name": "ops1",
        "data": {
            "exchange": "binance",
            "market": "futures",
            "symbols": symbols or ["BTCUSDT"],
            "timeframes": timeframes,
            "start": CONFIG_START.strftime("%Y-%m-%d"),
            "end": "2026-10-01",
        },
        "time_slices": {"type": "quarter"},
        "strategy": {"name": "ma_cross"},
        "params_grid": {"fast_ma": [10], "slow_ma": [100]},
        "execution": {
            "initial_balance": 10000,
            "commission_rate": 0.0006,
            "slippage_pct": 0.0003,
            "leverage": 3,
            "position_size_type": "percent_balance",
            "position_size_value": 20,
            "signal_lag": 1,
            "fill_on": "next_open",
        },
        "metrics": {"primary": "sharpe"},
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


class _RangeFetcher:
    """Источник, честно отдающий точки запрошенного окна ``[start, end)``.

    Запоминает окна: по ним видно, от какой точки догрузка продолжилась.
    """

    def __init__(self, period: timedelta = H) -> None:
        self.period = period
        self.windows: list[tuple[str, datetime, datetime]] = []

    def _points(self, start: datetime, end: datetime, step: timedelta) -> list[datetime]:
        step_ms = int(step.total_seconds() * 1000)
        first_ms = -(-int(start.timestamp() * 1000) // step_ms) * step_ms
        first = datetime.fromtimestamp(first_ms / 1000, UTC)
        out = []
        moment = first
        while moment < end:
            out.append(moment)
            moment += step
        return out

    def fetch_funding_rate(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        self.windows.append((FUNDING, start, end))
        times = self._points(start, end, 8 * H)
        return pl.DataFrame(
            {"open_time": times, "funding_rate": [0.0001] * len(times)},
            schema={"open_time": OPEN_INTEREST_SCHEMA["open_time"], "funding_rate": pl.Float64},
        )

    def fetch_open_interest(
        self, symbol: str, start: datetime, end: datetime, *, period: str
    ) -> pl.DataFrame:
        self.windows.append((OPEN_INTEREST, start, end))
        times = self._points(start, end, self.period)
        n = len(times)
        return pl.DataFrame(
            {
                "open_time": times,
                "open_interest": [1000.0] * n,
                "open_interest_value": [5e6] * n,
            },
            schema=OPEN_INTEREST_SCHEMA,
        )

    def fetch_long_short_ratio(
        self, symbol: str, start: datetime, end: datetime, *, period: str
    ) -> pl.DataFrame:
        self.windows.append((LONG_SHORT_RATIO, start, end))
        times = self._points(start, end, self.period)
        n = len(times)
        return pl.DataFrame(
            {
                "open_time": times,
                "long_short_ratio": [1.5] * n,
                "long_account": [0.6] * n,
                "short_account": [0.4] * n,
            },
            schema=LONG_SHORT_RATIO_SCHEMA,
        )


def _seed_oi(root: Path, *, until: datetime, hours: int) -> None:
    """Положить в store ``hours`` часовых точек, последняя — ``until``."""
    times = [until - i * H for i in range(hours)][::-1]
    frame = pl.DataFrame(
        {
            "open_time": times,
            "open_interest": [1000.0] * hours,
            "open_interest_value": [5e6] * hours,
        },
        schema=OPEN_INTEREST_SCHEMA,
    )
    ParquetStore(root).write_series(OPEN_INTEREST, frame, **_KEYS)


def _run(
    tmp_path: Path, *, timeframes: list[str] | None = None, **kw: object
) -> list[SeriesUpdate]:
    config = _config(tmp_path, timeframes=timeframes or ["1h"])
    params: dict[str, object] = {
        "data_root": tmp_path / "raw",
        "series": SeriesKind.OPEN_INTEREST,
        "now": NOW,
        "fetcher": _RangeFetcher(),
    }
    params.update(kw)
    return update_market_data_from_config(config, **params)  # type: ignore[arg-type]


# --- догрузка от последней точки --------------------------------------------


def test_resumes_from_last_stored_point(tmp_path: Path) -> None:
    """Окно начинается за последней сохранённой точкой, а не от начала конфига."""
    root = tmp_path / "raw"
    _seed_oi(root, until=NOW - 10 * H, hours=48)
    fetcher = _RangeFetcher()

    updates = _run(tmp_path, fetcher=fetcher)

    assert len(updates) == 1
    update = updates[0]
    assert update.stored_until == NOW - 10 * H
    assert update.requested_from == NOW - 9 * H, "продолжаем на период за последней точкой"
    assert fetcher.windows[0][1] == NOW - 9 * H
    assert update.lost is None
    assert update.coverage.status is CoverageStatus.COMPLETE


def test_right_edge_is_previous_complete_period(tmp_path: Path) -> None:
    """Правая граница — начало текущего часа, точка текущего периода не ждётся."""
    _seed_oi(tmp_path / "raw", until=NOW - 5 * H, hours=10)
    update = _run(tmp_path)[0]

    assert update.requested_to == TOP, "12:30 → граница 12:00"
    assert update.coverage.last == TOP - H


def test_up_to_date_series_still_appears_in_report(tmp_path: Path) -> None:
    """Актуальный ряд догружать нечего, но строка про него есть.

    Иначе «ряд актуален» и «ряд выпал из сбора» выглядят одинаково — тишиной,
    а отличить их нужно именно по выводу расписания.
    """
    _seed_oi(tmp_path / "raw", until=TOP - H, hours=24)
    updates = _run(tmp_path)

    assert len(updates) == 1
    update = updates[0]
    assert update.is_up_to_date
    assert update.rows == 0
    assert update.lost is None
    assert not has_defects(updates)
    assert "актуален до" in describe_update(update)
    assert "уже актуальны: 1" in describe_total(updates)


def test_rerun_is_idempotent(tmp_path: Path) -> None:
    """Два прогона подряд не создают дублей и не теряют точек."""
    root = tmp_path / "raw"
    _seed_oi(root, until=TOP - 24 * H, hours=24)
    _run(tmp_path)
    first = ParquetStore(root).read_series(OPEN_INTEREST, **_KEYS)
    _run(tmp_path)
    second = ParquetStore(root).read_series(OPEN_INTEREST, **_KEYS)

    assert second.height == first.height
    assert second["open_time"].n_unique() == second.height
    assert second["open_time"].is_sorted()


# --- пустой store ------------------------------------------------------------


def test_empty_store_loads_available_window_not_config_start(tmp_path: Path) -> None:
    """На пустом store окно урезается доступностью, а не берётся от 2022 года.

    Без клампа запрос ушёл бы с ``startTime`` четырёхлетней давности и получил
    сырой ``-1130`` — ровно то, на чём падал прогон RUN-7.
    """
    update = _run(tmp_path)[0]

    assert update.is_first_load
    assert update.resume_from == CONFIG_START
    assert update.requested_from == NOW - SAFE_RETENTION
    assert update.lost is not None, "история до окна retention недоступна — это пропуск"
    assert update.lost > timedelta(days=4 * 365)


def test_funding_on_empty_store_goes_back_to_config_start(tmp_path: Path) -> None:
    """У funding окна retention нет — доступна вся история, клампа быть не должно."""
    update = _run(tmp_path, series=SeriesKind.FUNDING)[0]

    assert update.requested_from == CONFIG_START
    assert update.lost is None
    assert update.coverage.status is CoverageStatus.UNVERIFIED


# --- пропущенная итерация ----------------------------------------------------


def test_missed_iterations_are_reported_as_unrecoverable(tmp_path: Path) -> None:
    """Сбор молчал два месяца: разрыв виден и назван невосполнимым."""
    _seed_oi(tmp_path / "raw", until=NOW - timedelta(days=60), hours=24)
    update = _run(tmp_path)[0]

    assert update.requested_from == NOW - SAFE_RETENTION
    assert update.lost is not None
    expected_lost = (NOW - SAFE_RETENTION) - (NOW - timedelta(days=60) + H)
    assert update.lost == expected_lost
    assert "НЕВОСПОЛНИМЫЙ ПРОПУСК" in describe_update(update)


def test_daily_cadence_loses_nothing(tmp_path: Path) -> None:
    """Сбор раз в сутки внутри окна retention не теряет ни точки."""
    _seed_oi(tmp_path / "raw", until=TOP - 25 * H, hours=24)
    update = _run(tmp_path)[0]

    assert update.lost is None
    assert update.coverage.status is CoverageStatus.COMPLETE
    assert update.requested_from == TOP - 24 * H
    assert update.rows == 24, "сутки часовых точек до границы окна"


def test_defects_flagged_for_the_whole_run(tmp_path: Path) -> None:
    """``has_defects`` — то, по чему расписание отличает успех от тишины."""
    _seed_oi(tmp_path / "raw", until=TOP - 24 * H, hours=24)
    clean = _run(tmp_path)
    assert not has_defects(clean)
    assert "пропусков нет" in describe_total(clean)

    lost = _run(tmp_path / "other")
    assert has_defects(lost)
    assert "невосполнимых пропусков: 1" in describe_total(lost)


# --- набор пар из конфига ----------------------------------------------------


def test_pairs_come_from_config(tmp_path: Path) -> None:
    """Состав сбора меняется правкой конфига, а не правкой расписания."""
    config = _config(tmp_path, symbols=["BTCUSDT", "ETHUSDT"], timeframes=["1h", "4h"])
    updates = update_market_data_from_config(
        config,
        data_root=tmp_path / "raw",
        series=SeriesKind.OPEN_INTEREST,
        now=NOW,
        fetcher=_RangeFetcher(),
    )
    assert {(u.symbol, u.timeframe) for u in updates} == {
        ("BTCUSDT", "1h"),
        ("BTCUSDT", "4h"),
        ("ETHUSDT", "1h"),
        ("ETHUSDT", "4h"),
    }


def test_all_series_are_collected_together(tmp_path: Path) -> None:
    updates = _run(tmp_path, series=SeriesKind.ALL)
    assert {u.series for u in updates} == {FUNDING, OPEN_INTEREST, LONG_SHORT_RATIO}


def test_timeframe_that_binance_cannot_aggregate_is_an_error(tmp_path: Path) -> None:
    """Таймфрейм не из набора Binance — ошибка, а не молчаливый пропуск ряда."""
    with pytest.raises(ValueError, match="not a valid Binance period"):
        _run(tmp_path, timeframes=["7h"])


def test_empty_update_list_is_reported(tmp_path: Path) -> None:
    assert "нечего догружать" in describe_total([])
    assert not has_defects([])


@pytest.mark.parametrize(
    ("count", "expected"),
    [(1, "1 точка"), (3, "3 точки"), (18, "18 точек"), (11, "11 точек"), (21, "21 точка")],
)
def test_point_count_agrees_with_numeral(tmp_path: Path, count: int, expected: str) -> None:
    """Строку читают люди: «3 точек» в отчёте расписания быть не должно."""
    _seed_oi(tmp_path / "raw", until=TOP - (count + 1) * H, hours=24)
    update = _run(tmp_path)[0]
    assert update.rows == count
    assert expected in describe_update(update)


# --- CLI ---------------------------------------------------------------------


def test_cli_reports_and_exits_nonzero_on_unrecoverable_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Невосполнимый пропуск — код 1: иначе провал сбора выглядит успехом."""
    monkeypatch.setattr(
        "trading_research.cli.update_market_data.BinanceMarketDataClient",
        lambda: _RangeFetcher(),
    )
    monkeypatch.setattr("trading_research.cli.update_market_data._now", lambda: NOW)
    config = _config(tmp_path, timeframes=["1h"])
    result = runner.invoke(
        app,
        [
            "update-market-data",
            "--config",
            str(config),
            "--series",
            "open_interest",
            "--data-root",
            str(tmp_path / "raw"),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "НЕВОСПОЛНИМЫЙ ПРОПУСК" in result.output
    assert "итого:" in result.output


def test_cli_exits_zero_when_nothing_lost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "trading_research.cli.update_market_data.BinanceMarketDataClient",
        lambda: _RangeFetcher(),
    )
    monkeypatch.setattr("trading_research.cli.update_market_data._now", lambda: NOW)
    root = tmp_path / "raw"
    _seed_oi(root, until=TOP - 24 * H, hours=24)
    config = _config(tmp_path, timeframes=["1h"])
    result = runner.invoke(
        app,
        [
            "update-market-data",
            "--config",
            str(config),
            "--series",
            "open_interest",
            "--data-root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "пропусков нет" in result.output
