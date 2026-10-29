"""Tests for RUN-2 — CLI загрузки funding / open interest / long-short ratio.

Без сети: источник подставляется через ``MarketDataFetcher``, как ``KlineFetcher``
в тестах EXP-6.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from trading_research.cli.fetch_market_data import (
    SERIES_PERIODS,
    SeriesKind,
    expand_series,
    fetch_market_data,
)
from trading_research.cli.main import app
from trading_research.data.schema import (
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

runner = CliRunner()
T0 = datetime(2024, 1, 1, tzinfo=UTC)
_KEYS = {"exchange": "binance", "market": "futures", "symbol": "BTCUSDT", "timeframe": "1h"}


def _times(n: int, step_h: int = 1, offset_h: int = 0) -> list[datetime]:
    return [T0 + timedelta(hours=offset_h + i * step_h) for i in range(n)]


def _funding(n: int, offset_h: int = 0, rate: float = 0.0001) -> pl.DataFrame:
    return pl.DataFrame(
        {"open_time": _times(n, 8, offset_h), "funding_rate": [rate] * n},
        schema=FUNDING_SCHEMA,
    )


def _oi(n: int, offset_h: int = 0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open_time": _times(n, 1, offset_h),
            "open_interest": [1000.0 + i for i in range(n)],
            "open_interest_value": [5_000_000.0 + i for i in range(n)],
        },
        schema=OPEN_INTEREST_SCHEMA,
    )


def _ls(n: int, offset_h: int = 0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open_time": _times(n, 1, offset_h),
            "long_short_ratio": [1.5] * n,
            "long_account": [0.6] * n,
            "short_account": [0.4] * n,
        },
        schema=LONG_SHORT_RATIO_SCHEMA,
    )


class _StubFetcher:
    """Подставной источник рядов; запоминает переданные period."""

    def __init__(
        self,
        funding: pl.DataFrame | None = None,
        oi: pl.DataFrame | None = None,
        ls: pl.DataFrame | None = None,
    ) -> None:
        self.funding = funding if funding is not None else _funding(6)
        self.oi = oi if oi is not None else _oi(24)
        self.ls = ls if ls is not None else _ls(24)
        self.periods: list[str] = []
        self.calls: list[str] = []

    def fetch_funding_rate(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        self.calls.append(FUNDING)
        return self.funding

    def fetch_open_interest(
        self, symbol: str, start: datetime, end: datetime, *, period: str
    ) -> pl.DataFrame:
        self.calls.append(OPEN_INTEREST)
        self.periods.append(period)
        return self.oi

    def fetch_long_short_ratio(
        self, symbol: str, start: datetime, end: datetime, *, period: str
    ) -> pl.DataFrame:
        self.calls.append(LONG_SHORT_RATIO)
        self.periods.append(period)
        return self.ls


def _run(root: Path, fetcher: _StubFetcher, **kw: object) -> dict[str, int]:
    params: dict[str, object] = {
        **_KEYS,
        "start": T0,
        "end": T0 + timedelta(days=1),
        "data_root": root,
        "fetcher": fetcher,
    }
    params.update(kw)
    return fetch_market_data(**params)  # type: ignore[arg-type]


# --- три ряда пишутся и читаются обратно ----------------------------------


def test_all_three_series_round_trip(tmp_path: Path) -> None:
    """``--series all`` грузит три ряда, каждый читается через read_series."""
    fetcher = _StubFetcher()
    written = _run(tmp_path, fetcher)

    assert set(written) == {FUNDING, OPEN_INTEREST, LONG_SHORT_RATIO}
    assert written[FUNDING] == 6
    assert written[OPEN_INTEREST] == 24
    assert written[LONG_SHORT_RATIO] == 24

    store = ParquetStore(tmp_path)
    back_funding = store.read_series(FUNDING, **_KEYS)
    assert back_funding.height == 6
    assert "funding_rate" in back_funding.columns

    back_oi = store.read_series(OPEN_INTEREST, **_KEYS)
    assert back_oi.height == 24
    assert {"open_interest", "open_interest_value"} <= set(back_oi.columns)

    back_ls = store.read_series(LONG_SHORT_RATIO, **_KEYS)
    assert back_ls.height == 24
    assert {"long_short_ratio", "long_account", "short_account"} <= set(back_ls.columns)


@pytest.mark.parametrize(
    ("kind", "expected_rows"),
    [(SeriesKind.FUNDING, 6), (SeriesKind.OPEN_INTEREST, 24), (SeriesKind.LONG_SHORT_RATIO, 24)],
)
def test_single_series_only_fetches_that_one(
    tmp_path: Path, kind: SeriesKind, expected_rows: int
) -> None:
    fetcher = _StubFetcher()
    written = _run(tmp_path, fetcher, series=kind)

    assert set(written) == {kind.value}
    assert written[kind.value] == expected_rows
    assert fetcher.calls == [kind.value]


def test_expand_series() -> None:
    assert expand_series(SeriesKind.ALL) == [
        SeriesKind.FUNDING,
        SeriesKind.OPEN_INTEREST,
        SeriesKind.LONG_SHORT_RATIO,
    ]
    assert expand_series(SeriesKind.FUNDING) == [SeriesKind.FUNDING]


# --- идемпотентность ------------------------------------------------------


def test_rerun_is_idempotent(tmp_path: Path) -> None:
    """Повторная загрузка того же диапазона не создаёт дублей (upsert)."""
    fetcher = _StubFetcher()
    _run(tmp_path, fetcher)
    _run(tmp_path, fetcher)

    store = ParquetStore(tmp_path)
    for dataset, rows in ((FUNDING, 6), (OPEN_INTEREST, 24), (LONG_SHORT_RATIO, 24)):
        back = store.read_series(dataset, **_KEYS)
        assert back.height == rows, f"{dataset}: ожидали {rows}, было {back.height}"
        assert back["open_time"].n_unique() == rows


def test_overlapping_window_merges(tmp_path: Path) -> None:
    """Пересекающиеся загрузки сливаются по ``open_time``, а не дублируются."""
    _run(tmp_path, _StubFetcher(oi=_oi(12)))
    # Вторая порция перекрывает первую на 6 часов и добавляет 6 новых.
    _run(tmp_path, _StubFetcher(oi=_oi(12, offset_h=6)))

    back = ParquetStore(tmp_path).read_series(OPEN_INTEREST, **_KEYS)
    assert back.height == 18  # 12 + 6 новых
    assert back["open_time"].n_unique() == 18
    assert back["open_time"].is_sorted()


# --- период для OI/LS ------------------------------------------------------


def test_timeframe_is_passed_as_period(tmp_path: Path) -> None:
    """``timeframe`` уходит в Binance как ``period`` для OI и LS."""
    fetcher = _StubFetcher()
    _run(tmp_path, fetcher, timeframe="4h")
    assert fetcher.periods == ["4h", "4h"]


@pytest.mark.parametrize("bad", ["1m", "3m", "3d", "1w", "7h"])
def test_unsupported_period_raises_for_periodic_series(tmp_path: Path, bad: str) -> None:
    """Интервал, которого нет у Binance, — ошибка, а не пустой результат."""
    with pytest.raises(ValueError, match="not a valid Binance period"):
        _run(tmp_path, _StubFetcher(), timeframe=bad, series=SeriesKind.OPEN_INTEREST)
    with pytest.raises(ValueError, match="not a valid Binance period"):
        _run(tmp_path, _StubFetcher(), timeframe=bad, series=SeriesKind.ALL)


@pytest.mark.parametrize("bad", ["1m", "3d"])
def test_funding_ignores_unsupported_period(tmp_path: Path, bad: str) -> None:
    """У funding своего интервала нет — ограничение на него не распространяется."""
    written = _run(tmp_path, _StubFetcher(), timeframe=bad, series=SeriesKind.FUNDING)
    assert written[FUNDING] == 6
    back = ParquetStore(tmp_path).read_series(FUNDING, **{**_KEYS, "timeframe": bad})
    assert back.height == 6


@pytest.mark.parametrize("good", SERIES_PERIODS)
def test_all_documented_periods_accepted(tmp_path: Path, good: str) -> None:
    written = _run(tmp_path / good, _StubFetcher(), timeframe=good, series=SeriesKind.OPEN_INTEREST)
    assert written[OPEN_INTEREST] == 24


# --- пустой ответ ---------------------------------------------------------


def test_empty_series_is_written_without_error(tmp_path: Path) -> None:
    """Пустой ответ (нет данных за период) не ломает запись."""
    fetcher = _StubFetcher(funding=pl.DataFrame(schema=FUNDING_SCHEMA))
    written = _run(tmp_path, fetcher, series=SeriesKind.FUNDING)
    assert written[FUNDING] == 0


# --- CLI smoke ------------------------------------------------------------


def test_cli_fetch_market_data_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "trading_research.cli.fetch_market_data.BinanceMarketDataClient",
        lambda: _StubFetcher(),
    )
    root = tmp_path / "raw"
    result = runner.invoke(
        app,
        [
            "fetch-market-data",
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
    assert "wrote 6 rows of funding" in result.output
    assert "wrote 24 rows of open_interest" in result.output
    assert "wrote 24 rows of long_short_ratio" in result.output
    assert any(root.glob("funding/**/*.parquet"))
    assert any(root.glob("open_interest/**/*.parquet"))
    assert any(root.glob("long_short_ratio/**/*.parquet"))


def test_cli_single_series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "trading_research.cli.fetch_market_data.BinanceMarketDataClient",
        lambda: _StubFetcher(),
    )
    root = tmp_path / "raw"
    result = runner.invoke(
        app,
        [
            "fetch-market-data",
            "--symbol",
            "BTCUSDT",
            "--timeframe",
            "1h",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-02",
            "--series",
            "funding",
            "--data-root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "funding" in result.output
    assert "open_interest" not in result.output


def test_cli_bad_period_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "trading_research.cli.fetch_market_data.BinanceMarketDataClient",
        lambda: _StubFetcher(),
    )
    result = runner.invoke(
        app,
        [
            "fetch-market-data",
            "--symbol",
            "BTCUSDT",
            "--timeframe",
            "3d",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-02",
            "--data-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
