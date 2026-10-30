"""Tests for REG-4 — окно retention и честность про пустые фичи (без сети).

Три разных тихих отказа, с одним корнем: Binance отдаёт OI/LS только за 30 дней.

1. Загрузка исторического диапазона падала сырым `-1130`, причём `--series all`
   падал целиком — funding успевал записаться, OI и LS нет.
2. `build-features` отрабатывал с кодом 0, когда фича выходила пустой целиком.
3. `analyze-regimes` печатал по пустой фиче `null` в одном ряду с настоящими
   корреляциями, то есть «данных нет» выглядело как «связи нет».
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from trading_research.analytics.correlations import (
    describe_availability,
    regime_feature_availability,
)
from trading_research.cli.build_features import (
    describe_feature_coverage,
    feature_coverage,
)
from trading_research.cli.fetch_market_data import (
    CoverageStatus,
    SeriesKind,
    clamp_to_retention,
    describe_coverage,
    fetch_market_data,
)
from trading_research.data.ingestion.binance_client import BinanceAPIError, HttpResponse
from trading_research.data.ingestion.market_data_client import (
    SAFE_RETENTION,
    BinanceMarketDataClient,
    RetentionWindowError,
)
from trading_research.data.schema import OPEN_INTEREST_SCHEMA
from trading_research.data.storage.parquet_store import (
    FUNDING,
    LONG_SHORT_RATIO,
    OPEN_INTEREST,
    ParquetStore,
)
from trading_research.features.market_regime import TIME_SLICE_FEATURE_SCHEMA

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
H = timedelta(hours=1)
_KEYS = {"exchange": "binance", "market": "futures", "symbol": "BTCUSDT", "timeframe": "1h"}


# --- кламп окна -------------------------------------------------------------


@pytest.mark.parametrize("kind", [SeriesKind.OPEN_INTEREST, SeriesKind.LONG_SHORT_RATIO])
def test_historical_start_is_clamped_to_available_window(kind: SeriesKind) -> None:
    """Начало четырёхлетней давности урезается до доступного окна."""
    asked = datetime(2022, 1, 1, tzinfo=UTC)
    assert clamp_to_retention(asked, kind=kind, now=NOW) == NOW - SAFE_RETENTION


def test_start_inside_window_is_left_alone() -> None:
    asked = NOW - timedelta(days=10)
    assert clamp_to_retention(asked, kind=SeriesKind.OPEN_INTEREST, now=NOW) == asked


def test_funding_is_never_clamped() -> None:
    """У funding окна retention нет — его история доступна целиком."""
    asked = datetime(2022, 1, 1, tzinfo=UTC)
    assert clamp_to_retention(asked, kind=SeriesKind.FUNDING, now=NOW) == asked


def test_naive_start_is_treated_as_utc() -> None:
    """Typer отдаёт наивное время; сравнение с окном не должно падать."""
    naive = datetime(2026, 10, 25)
    assert clamp_to_retention(naive, kind=SeriesKind.OPEN_INTEREST, now=NOW) == naive.replace(
        tzinfo=UTC
    )


# --- `--series all` на историческом диапазоне не падает ----------------------


class _WindowFetcher:
    """Источник, отдающий точки запрошенного окна и запоминающий границы."""

    def __init__(self) -> None:
        self.windows: dict[str, tuple[datetime, datetime]] = {}

    def _frame(self, start: datetime, end: datetime, name: str) -> pl.DataFrame:
        self.windows[name] = (start, end)
        times, moment = [], start
        while moment < end:
            times.append(moment)
            moment += H
        n = len(times)
        return pl.DataFrame(
            {
                "open_time": times,
                "open_interest": [1000.0] * n,
                "open_interest_value": [5e6] * n,
            },
            schema=OPEN_INTEREST_SCHEMA,
        )

    def fetch_funding_rate(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        self.windows[FUNDING] = (start, end)
        return pl.DataFrame(
            {"open_time": [start], "funding_rate": [0.0001]},
            schema={"open_time": OPEN_INTEREST_SCHEMA["open_time"], "funding_rate": pl.Float64},
        )

    def fetch_open_interest(
        self, symbol: str, start: datetime, end: datetime, *, period: str
    ) -> pl.DataFrame:
        return self._frame(start, end, OPEN_INTEREST)

    def fetch_long_short_ratio(
        self, symbol: str, start: datetime, end: datetime, *, period: str
    ) -> pl.DataFrame:
        frame = self._frame(start, end, LONG_SHORT_RATIO)
        return frame.select("open_time").with_columns(
            long_short_ratio=pl.lit(1.5), long_account=pl.lit(0.6), short_account=pl.lit(0.4)
        )


def test_series_all_on_historical_range_does_not_fail(tmp_path: Path) -> None:
    """Главный сценарий тикета: funding за всю историю, OI/LS за доступное окно."""
    asked_start = datetime(2022, 1, 1, tzinfo=UTC)
    fetcher = _WindowFetcher()

    written = fetch_market_data(
        **_KEYS,
        start=asked_start,
        end=NOW,
        series=SeriesKind.ALL,
        data_root=tmp_path,
        fetcher=fetcher,
        now=NOW,
    )

    assert fetcher.windows[FUNDING][0] == asked_start, "funding грузится за весь диапазон"
    assert fetcher.windows[OPEN_INTEREST][0] == NOW - SAFE_RETENTION
    assert fetcher.windows[LONG_SHORT_RATIO][0] == NOW - SAFE_RETENTION

    assert not written[FUNDING].is_clamped
    assert written[OPEN_INTEREST].is_clamped
    assert written[LONG_SHORT_RATIO].is_clamped


def test_clamped_window_says_so_in_the_report(tmp_path: Path) -> None:
    """«Покрытие полное» на урезанном окне без оговорки вводит в заблуждение."""
    written = fetch_market_data(
        **_KEYS,
        start=datetime(2022, 1, 1, tzinfo=UTC),
        end=NOW,
        series=SeriesKind.OPEN_INTEREST,
        data_root=tmp_path,
        fetcher=_WindowFetcher(),
        now=NOW,
    )
    line = describe_coverage(OPEN_INTEREST, "BTCUSDT", "1h", written[OPEN_INTEREST])
    assert "ОКНО УРЕЗАНО" in line
    assert "2022-01-01" in line, "в строке должно быть то, что просили"
    assert f"{SAFE_RETENTION.days} дней" in line


def test_range_entirely_outside_window_is_not_an_error(tmp_path: Path) -> None:
    """Диапазон целиком в прошлом: не падение и не «неполное покрытие».

    Без отдельного статуса начало уезжало за конец, `expected` уходило в минус, а
    запрос — в `ValueError: end must be after start`, по которому про окно
    retention не догадаться.
    """
    written = fetch_market_data(
        **_KEYS,
        start=datetime(2022, 1, 1, tzinfo=UTC),
        end=datetime(2022, 6, 1, tzinfo=UTC),
        series=SeriesKind.OPEN_INTEREST,
        data_root=tmp_path,
        fetcher=_WindowFetcher(),
        now=NOW,
    )
    coverage = written[OPEN_INTEREST]
    assert coverage.status is CoverageStatus.OUT_OF_WINDOW
    assert coverage.rows == 0
    assert coverage.expected == 0, "ожидание не должно быть отрицательным"

    line = describe_coverage(OPEN_INTEREST, "BTCUSDT", "1h", coverage)
    assert "НЕТ ДАННЫХ" in line
    assert "update-market-data" in line, "сказать, чем эту историю вообще добыть"


def test_out_of_window_does_not_write_anything(tmp_path: Path) -> None:
    """Запрашивать нечего — значит и обращения к источнику быть не должно."""
    fetcher = _WindowFetcher()
    fetch_market_data(
        **_KEYS,
        start=datetime(2022, 1, 1, tzinfo=UTC),
        end=datetime(2022, 6, 1, tzinfo=UTC),
        series=SeriesKind.OPEN_INTEREST,
        data_root=tmp_path,
        fetcher=fetcher,
        now=NOW,
    )
    assert fetcher.windows == {}


# --- сырой -1130 не доходит до пользователя ---------------------------------


class _RetentionErrorTransport:
    """Транспорт, отвечающий как Binance на startTime вне окна."""

    def __init__(self, body: bytes) -> None:
        self.body = body

    def get(self, url: str, params: Mapping[str, str]) -> HttpResponse:
        return HttpResponse(400, {}, self.body)


def test_retention_error_is_translated() -> None:
    """`-1130` превращается в сообщение, называющее окно и способ обойти."""
    body = json.dumps({"msg": "parameter 'startTime' is invalid.", "code": -1130}).encode()
    client = BinanceMarketDataClient(
        transport=_RetentionErrorTransport(body), sleep_fn=lambda _s: None
    )
    with pytest.raises(RetentionWindowError) as excinfo:
        client.fetch_open_interest("BTCUSDT", NOW - timedelta(days=60), NOW, period="1h")

    message = str(excinfo.value)
    assert "30 дней" in message
    assert "update-market-data" in message
    assert "--series funding" in message
    assert "openInterestHist" in message, "сказать, какой именно ряд"


def test_retention_error_message_is_not_truncated() -> None:
    """Родительский конструктор режет тело до 200 символов — объяснение бы срезало."""
    body = json.dumps({"msg": "x" * 500, "code": -1130}).encode()
    client = BinanceMarketDataClient(
        transport=_RetentionErrorTransport(body), sleep_fn=lambda _s: None
    )
    with pytest.raises(RetentionWindowError) as excinfo:
        client.fetch_open_interest("BTCUSDT", NOW - timedelta(days=60), NOW, period="1h")
    assert "--series funding" in str(excinfo.value)


def test_other_api_errors_are_not_disguised_as_retention() -> None:
    """Прочие 400 остаются собой — иначе объяснение будет враньём."""
    body = json.dumps({"msg": "invalid symbol", "code": -1121}).encode()
    client = BinanceMarketDataClient(
        transport=_RetentionErrorTransport(body), sleep_fn=lambda _s: None
    )
    with pytest.raises(BinanceAPIError) as excinfo:
        client.fetch_open_interest("NOPE", NOW - timedelta(days=5), NOW, period="1h")
    assert not isinstance(excinfo.value, RetentionWindowError)


def test_non_json_body_is_not_mistaken_for_retention() -> None:
    client = BinanceMarketDataClient(
        transport=_RetentionErrorTransport(b"<html>gateway</html>"), sleep_fn=lambda _s: None
    )
    with pytest.raises(BinanceAPIError) as excinfo:
        client.fetch_open_interest("BTCUSDT", NOW - timedelta(days=5), NOW, period="1h")
    assert not isinstance(excinfo.value, RetentionWindowError)


# --- покрытие фич в build-features ------------------------------------------


def _features_table(*, oi: list[float | None], rows: int = 3) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * rows,
            "timeframe": ["1h"] * rows,
            "time_slice_id": [f"s{i}" for i in range(rows)],
            "market_return_pct": [1.0] * rows,
            "volatility": [0.1] * rows,
            "trend_strength": [0.2] * rows,
            "adx": [20.0] * rows,
            "funding_avg": [0.0001] * rows,
            "oi_change_pct": oi,
            "ls_ratio_avg": [None] * rows,
            "volume_zscore": [0.5] * rows,
            "drawdown_from_ath": [-5.0] * rows,
            "regime_label": ["bull_trend"] * rows,
        },
        schema=TIME_SLICE_FEATURE_SCHEMA,
    )


def test_empty_feature_is_visible_in_coverage() -> None:
    """Симптом RUN-7: фича пуста целиком, а команда молчала."""
    coverage = feature_coverage(_features_table(oi=[None, None, None]))
    by_name = {c.feature: c for c in coverage}

    assert by_name["oi_change_pct"].is_empty
    assert by_name["ls_ratio_avg"].is_empty
    assert not by_name["volatility"].is_empty
    assert by_name["volatility"].filled == 3


def test_partial_feature_is_distinguished_from_empty() -> None:
    coverage = {c.feature: c for c in feature_coverage(_features_table(oi=[1.0, None, None]))}
    assert coverage["oi_change_pct"].is_partial
    assert not coverage["oi_change_pct"].is_empty
    assert coverage["oi_change_pct"].pct == pytest.approx(100 / 3)


def test_keys_are_not_counted_as_features() -> None:
    names = {c.feature for c in feature_coverage(_features_table(oi=[1.0, 2.0, 3.0]))}
    assert "symbol" not in names
    assert "time_slice_id" not in names
    assert "regime_label" in names, "метка режима тоже может не посчитаться"


def test_coverage_report_names_empty_features() -> None:
    lines = describe_feature_coverage(feature_coverage(_features_table(oi=[None] * 3)))
    header = lines[0]
    assert "пустых — 2" in header
    assert "oi_change_pct" in header
    assert any("← ПУСТО" in line for line in lines)


def test_coverage_of_empty_table_is_one_problem_not_ten() -> None:
    """Пустая таблица — один отказ, а не десять «пустых фич»."""
    empty = pl.DataFrame(schema=TIME_SLICE_FEATURE_SCHEMA)
    lines = describe_feature_coverage(feature_coverage(empty))
    assert lines == ["покрытие фич: таблица пуста — ни одного слайса"]


# --- «нет данных» против «нет связи» в analyze-regimes ----------------------


def _joined(*, oi: list[float | None], n: int = 4) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "param_hash": ["a"] * n,
            "sharpe": [0.1 * i for i in range(n)],
            "market_return_pct": [float(i) for i in range(n)],
            "volatility": [0.1] * n,
            "trend_strength": [0.2] * n,
            "adx": [20.0] * n,
            "funding_avg": [0.0001] * n,
            "oi_change_pct": oi,
            "ls_ratio_avg": [None] * n,
            "volume_zscore": [0.5] * n,
            "drawdown_from_ath": [-5.0] * n,
        }
    )


def test_missing_feature_is_reported_separately_from_no_correlation() -> None:
    """`null` от пустой фичи и `null` от отсутствия связи — разные ответы."""
    availability = {a.feature: a for a in regime_feature_availability(_joined(oi=[None] * 4))}
    assert availability["oi_change_pct"].is_missing
    assert availability["ls_ratio_avg"].is_missing
    assert not availability["volatility"].is_missing

    lines = describe_availability(list(availability.values()))
    assert any("ДАННЫХ НЕТ" in line for line in lines)
    assert any("oi_change_pct" in line for line in lines)


def test_single_point_feature_is_called_out() -> None:
    """Второй источник `null`: одна точка, корреляция не определена."""
    availability = regime_feature_availability(_joined(oi=[1.0, None, None, None]))
    single = next(a for a in availability if a.feature == "oi_change_pct")
    assert single.is_constant
    assert not single.is_missing
    assert any("одна точка" in line for line in describe_availability(availability))


def test_full_data_says_null_means_no_link() -> None:
    """Когда данных хватает, пометка должна говорить обратное — иначе шум."""
    full = _joined(oi=[1.0, 2.0, 3.0, 4.0]).with_columns(ls_ratio_avg=pl.lit(1.0))
    lines = describe_availability(regime_feature_availability(full))
    assert len(lines) == 1
    assert "это отсутствие связи" in lines[0]


# --- сквозной путь: пустой ряд доходит до пометки в корреляциях -------------


def test_empty_series_stays_visible_from_store_to_correlations(tmp_path: Path) -> None:
    """Пустой OI не должен тихо дойти до матрицы — его видно на обоих шагах."""
    store = ParquetStore(tmp_path)
    assert store.read_series(OPEN_INTEREST, **_KEYS).is_empty()

    table = _features_table(oi=[None] * 3)
    assert next(c for c in feature_coverage(table) if c.feature == "oi_change_pct").is_empty

    availability = regime_feature_availability(_joined(oi=[None] * 4))
    assert next(a for a in availability if a.feature == "oi_change_pct").is_missing


def test_naive_boundaries_reach_the_fetcher_as_utc(tmp_path: Path) -> None:
    """Typer отдаёт наивные границы — смешивать их с aware нельзя.

    Регрессия: кламп нормализовал только `start`, и на funding (который не
    клампится) загрузчик сравнивал наивный конец с UTC-aware началом —
    `TypeError: can't compare offset-naive and offset-aware datetimes`. Тесты
    этого не ловили, потому что везде передавали aware-время, как Typer не делает.
    """
    fetcher = _WindowFetcher()
    fetch_market_data(
        **_KEYS,
        start=datetime(2026, 10, 20),
        end=datetime(2026, 10, 25),
        series=SeriesKind.ALL,
        data_root=tmp_path,
        fetcher=fetcher,
        now=NOW,
    )
    for name, (start, end) in fetcher.windows.items():
        assert start.tzinfo is not None, f"{name}: наивное начало дошло до загрузчика"
        assert end.tzinfo is not None, f"{name}: наивный конец дошёл до загрузчика"
