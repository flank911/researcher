"""Tests for OPS-2 — инкрементальная догрузка свечей (без сети).

Проверяется не «догрузилось», а **какое окно ушло в API**: смысл команды в том,
что запрашивается только хвост, и подтвердить это можно лишь по границам запроса,
а не по числу строк в store (запись идемпотентна, поэтому повторная перекачка всей
истории выглядела бы в store точно так же, как аккуратная догрузка).

Отдельно закреплены два следствия схемы «продолжаем от последнего бара + период»:
незакрытый бар не должен попасть в store (вернуться к нему уже не получится), а
повторный запуск без новых данных не должен ходить в сеть.

Время задаётся через ``now=``: тест, привязанный к реальным часам, начал бы падать
сам по себе.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import yaml
from typer.testing import CliRunner

from trading_research.cli.main import app
from trading_research.cli.update_data import (
    CandleUpdate,
    describe_total,
    describe_update,
    has_defects,
    update_data_from_config,
)
from trading_research.data.schema import CANDLE_SCHEMA
from trading_research.data.storage.parquet_store import ParquetStore

runner = CliRunner()

# NOW намеренно не выровнен по часу: на нём проверяется, что правая граница
# округляется вниз, а не берёт текущий (ещё открытый) бар.
NOW = datetime(2026, 10, 31, 12, 30, tzinfo=UTC)
TOP = NOW.replace(minute=0)
CONFIG_START = datetime(2022, 1, 1, tzinfo=UTC)
H = timedelta(hours=1)
_KEYS = {"exchange": "binance", "market": "futures", "symbol": "BTCUSDT", "timeframe": "1h"}


def _config(
    tmp_path: Path,
    *,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
) -> Path:
    """Минимальный валидный конфиг — источник набора пар и начала истории."""
    payload = {
        "schema_version": 2,
        "name": "ops2",
        "data": {
            "exchange": "binance",
            "market": "futures",
            "symbols": symbols or ["BTCUSDT"],
            "timeframes": timeframes or ["1h"],
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


def _bars(times: list[datetime], *, step: timedelta = H) -> pl.DataFrame:
    n = len(times)
    return pl.DataFrame(
        {
            "open_time": times,
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.5] * n,
            "volume": [10.0] * n,
            "close_time": [t + step for t in times],
            "quote_volume": [1000.0] * n,
            "trades": [5] * n,
            "taker_buy_base": [4.0] * n,
            "taker_buy_quote": [400.0] * n,
        },
        schema=CANDLE_SCHEMA,
    )


class _RangeFetcher:
    """Источник, честно отдающий бары запрошенного окна ``[start, end)``.

    Запоминает окна: по ним видно, от какой точки продолжилась догрузка, и ходил
    ли вызов в API вообще.
    """

    def __init__(self, *, step: timedelta = H) -> None:
        self.step = step
        self.windows: list[tuple[str, str, datetime, datetime]] = []

    def fetch_klines(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        self.windows.append((symbol, timeframe, start, end))
        step_ms = int(self.step.total_seconds() * 1000)
        first_ms = -(-int(start.timestamp() * 1000) // step_ms) * step_ms
        times: list[datetime] = []
        moment = datetime.fromtimestamp(first_ms / 1000, UTC)
        while moment < end:
            times.append(moment)
            moment += self.step
        return _bars(times, step=self.step)


class _TruncatedFetcher(_RangeFetcher):
    """Отдаёт окно без последних ``missing`` баров — хвост с дыркой."""

    def __init__(self, *, missing: int) -> None:
        super().__init__()
        self.missing = missing

    def fetch_klines(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        frame = super().fetch_klines(symbol, timeframe, start, end)
        return frame.head(max(0, frame.height - self.missing))


def _seed(root: Path, *, until: datetime, bars: int, **keys: str) -> None:
    """Положить в store ``bars`` часовых баров, последний — ``until``."""
    times = [until - i * H for i in range(bars)][::-1]
    ParquetStore(root).write_candles(_bars(times), **{**_KEYS, **keys})


def _run(tmp_path: Path, **kw: object) -> list[CandleUpdate]:
    config = kw.pop("config", None) or _config(tmp_path)
    params: dict[str, object] = {
        "data_root": tmp_path / "raw",
        "now": NOW,
        "fetcher": _RangeFetcher(),
    }
    params.update(kw)
    return update_data_from_config(config, **params)  # type: ignore[arg-type]


# --- догрузка хвоста --------------------------------------------------------


def test_requests_only_the_missing_tail(tmp_path: Path) -> None:
    """DoD: запрашивается только недостающий хвост, а не весь диапазон конфига."""
    _seed(tmp_path / "raw", until=TOP - 10 * H, bars=48)
    fetcher = _RangeFetcher()

    updates = _run(tmp_path, fetcher=fetcher)

    assert len(updates) == 1
    update = updates[0]
    assert update.stored_until == TOP - 10 * H
    assert update.requested_from == TOP - 9 * H, "продолжаем на бар за последним сохранённым"
    assert update.requested_to == TOP
    assert update.rows == 9
    assert fetcher.windows == [("BTCUSDT", "1h", TOP - 9 * H, TOP)]
    assert not update.is_short


def test_tail_is_appended_to_the_existing_store(tmp_path: Path) -> None:
    """Догрузка дописывает, а не заменяет: старые бары остаются на месте."""
    root = tmp_path / "raw"
    _seed(root, until=TOP - 10 * H, bars=48)

    _run(tmp_path)

    stored = ParquetStore(root).read_candles(**_KEYS)
    times = stored["open_time"].to_list()
    assert stored.height == 48 + 9
    assert times[0] == TOP - 57 * H
    assert times[-1] == TOP - H
    assert times == sorted(times)


def test_open_bar_is_never_written(tmp_path: Path) -> None:
    """Правая граница — начало текущего бара: незакрытый бар в store не попадает.

    Это не косметика. Следующий запуск начнёт с ``последний + период``, то есть к
    незакрытому бару уже не вернётся, и частичные ``high``/``low``/``close``
    остались бы в store навсегда.
    """
    _seed(tmp_path / "raw", until=TOP - 3 * H, bars=5)

    update = _run(tmp_path)[0]

    assert update.requested_to == TOP, "окно кончается на начале текущего бара, не на NOW"
    stored = ParquetStore(tmp_path / "raw").read_candles(**_KEYS)
    assert stored["open_time"].max() == TOP - H


# --- пустой store и новые пары ----------------------------------------------


def test_empty_store_is_a_full_load_from_config(tmp_path: Path) -> None:
    """DoD: пустой store обрабатывается как полная загрузка из конфига."""
    fetcher = _RangeFetcher()

    update = _run(tmp_path, fetcher=fetcher)[0]

    assert update.stored_until is None
    assert update.is_first_load
    assert update.requested_from == CONFIG_START
    assert fetcher.windows == [("BTCUSDT", "1h", CONFIG_START, TOP)]
    assert update.rows == int((TOP - CONFIG_START) / H)


def test_new_symbol_and_timeframe_load_from_scratch(tmp_path: Path) -> None:
    """DoD: новая пара в конфиге загружается с нуля той же командой.

    Старая пара при этом догружает только хвост — расширение покрытия не означает
    перекачку того, что уже есть.
    """
    root = tmp_path / "raw"
    _seed(root, until=TOP - 5 * H, bars=20)
    config = _config(tmp_path, symbols=["BTCUSDT", "ETHUSDT"], timeframes=["1h"])
    fetcher = _RangeFetcher()

    updates = _run(tmp_path, config=config, fetcher=fetcher)

    by_symbol = {u.symbol: u for u in updates}
    assert by_symbol["BTCUSDT"].requested_from == TOP - 4 * H
    assert by_symbol["ETHUSDT"].is_first_load
    assert by_symbol["ETHUSDT"].requested_from == CONFIG_START
    assert [w[0] for w in fetcher.windows] == ["BTCUSDT", "ETHUSDT"]


def test_every_config_pair_gets_a_line(tmp_path: Path) -> None:
    """Пара без данных не должна исчезать из отчёта — иначе провал виден как тишина."""
    config = _config(tmp_path, symbols=["BTCUSDT", "ETHUSDT"], timeframes=["1h", "4h"])

    updates = _run(tmp_path, config=config)

    assert {(u.symbol, u.timeframe) for u in updates} == {
        ("BTCUSDT", "1h"),
        ("BTCUSDT", "4h"),
        ("ETHUSDT", "1h"),
        ("ETHUSDT", "4h"),
    }


def test_timeframe_sets_the_period(tmp_path: Path) -> None:
    """Шаг догрузки — таймфрейм пары, а не час по умолчанию."""
    config = _config(tmp_path, timeframes=["4h"])
    _seed(
        tmp_path / "raw",
        until=TOP - 12 * H,
        bars=4,
        timeframe="4h",
    )

    update = _run(tmp_path, config=config, fetcher=_RangeFetcher(step=4 * H))[0]

    assert update.period == 4 * H
    assert update.requested_from == TOP - 8 * H
    # 12:00 — граница 4-часового бара, поэтому правая граница именно она.
    assert update.requested_to == TOP


# --- повторный запуск -------------------------------------------------------


def test_second_run_without_new_data_touches_nothing(tmp_path: Path) -> None:
    """DoD: повторный запуск без новых данных не зовёт API и не меняет store."""
    root = tmp_path / "raw"
    _seed(root, until=TOP - H, bars=24)
    before = ParquetStore(root).read_candles(**_KEYS)
    fetcher = _RangeFetcher()

    update = _run(tmp_path, fetcher=fetcher)[0]

    assert fetcher.windows == [], "актуальному ряду запрос не нужен"
    assert update.is_up_to_date
    assert update.rows == 0
    assert ParquetStore(root).read_candles(**_KEYS).equals(before)


def test_repeated_run_is_idempotent_after_a_top_up(tmp_path: Path) -> None:
    """Догрузить, затем прогнать ещё раз: второй прогон уже ничего не запрашивает."""
    _seed(tmp_path / "raw", until=TOP - 6 * H, bars=12)

    first = _run(tmp_path)[0]
    fetcher = _RangeFetcher()
    second = _run(tmp_path, fetcher=fetcher)[0]

    assert first.rows == 5
    assert second.is_up_to_date
    assert fetcher.windows == []
    assert second.stored_until == TOP - H


# --- недобор ----------------------------------------------------------------


def test_short_tail_is_reported_as_a_defect(tmp_path: Path) -> None:
    """Недобор на догрузке — дырка в свежей истории, её видно сразу."""
    _seed(tmp_path / "raw", until=TOP - 10 * H, bars=24)

    update = _run(tmp_path, fetcher=_TruncatedFetcher(missing=3))[0]

    assert update.expected == 9
    assert update.rows == 6
    assert update.is_short
    assert has_defects([update])
    assert "НЕДОБОР" in describe_update(update)


def test_short_first_load_is_not_a_defect(tmp_path: Path) -> None:
    """На первой загрузке недобор — норма: символ мог начать торговаться позже."""
    update = _run(tmp_path, fetcher=_TruncatedFetcher(missing=5000))[0]

    assert update.rows < update.expected
    assert not update.is_short, "ругаться на каждый новый тикер нельзя"
    assert not has_defects([update])


# --- отчёт ------------------------------------------------------------------


def test_describe_update_names_the_window_and_the_volume(tmp_path: Path) -> None:
    """Объём запроса в выводе — именно то, ради чего команда существует."""
    _seed(tmp_path / "raw", until=TOP - 10 * H, bars=24)

    line = describe_update(_run(tmp_path)[0])

    assert "BTCUSDT 1h:" in line
    assert "было до 2026-10-31 02:00" in line
    assert "2026-10-31 03:00 … 2026-10-31 12:00" in line
    assert "дописано 9 баров" in line


def test_describe_update_distinguishes_absent_from_up_to_date(tmp_path: Path) -> None:
    first_load = _run(tmp_path)[0]
    _seed(tmp_path / "raw", until=TOP - H, bars=3)
    fresh = _run(tmp_path)[0]

    assert "пары не было в store" in describe_update(first_load)
    assert "догружать нечего" in describe_update(fresh)


def test_describe_total_counts_fresh_and_first_loads(tmp_path: Path) -> None:
    config = _config(tmp_path, symbols=["BTCUSDT", "ETHUSDT"], timeframes=["1h"])
    _seed(tmp_path / "raw", until=TOP - H, bars=3)

    total = describe_total(_run(tmp_path, config=config))

    assert "по 2 парам" in total
    assert "уже актуальны: 1" in total
    assert "загружено с нуля: 1" in total
    assert "недобора нет" in total


def test_describe_total_without_pairs() -> None:
    assert describe_total([]) == "нечего догружать: в конфиге нет пар"


# --- CLI --------------------------------------------------------------------


def test_cli_prints_lines_and_exits_zero(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path / "raw", until=TOP - 4 * H, bars=10)
    config = _config(tmp_path)
    monkeypatch.setattr("trading_research.cli.update_data._now", lambda: NOW)
    monkeypatch.setattr(
        "trading_research.cli.update_data.BinanceFuturesClient", lambda: _RangeFetcher()
    )

    result = runner.invoke(
        app, ["update-data", "--config", str(config), "--data-root", str(tmp_path / "raw")]
    )

    assert result.exit_code == 0, result.output
    assert "BTCUSDT 1h:" in result.output
    assert "дописано 3 бара" in result.output
    assert "итого:" in result.output


def test_cli_exits_nonzero_on_a_short_tail(tmp_path: Path, monkeypatch) -> None:
    """Расписание не должно отчитываться об успехе там, где данных не хватает."""
    _seed(tmp_path / "raw", until=TOP - 10 * H, bars=10)
    config = _config(tmp_path)
    monkeypatch.setattr("trading_research.cli.update_data._now", lambda: NOW)
    monkeypatch.setattr(
        "trading_research.cli.update_data.BinanceFuturesClient",
        lambda: _TruncatedFetcher(missing=4),
    )

    result = runner.invoke(
        app, ["update-data", "--config", str(config), "--data-root", str(tmp_path / "raw")]
    )

    assert result.exit_code == 1
    assert "НЕДОБОР" in result.output
    assert "недобор: 1" in result.output
