"""RUN-2 — бизнес-логика команды ``fetch-market-data``.

Связка «загрузчик рыночных рядов (REG-1) → Parquet-store (DATA-2)». До этого
тикета `fetch_funding_rate` / `fetch_open_interest` / `fetch_long_short_ratio` и
парные им `write_series` / `read_series` существовали, но между ними не было ничего:
загрузить funding/OI/LS можно было только из Python.

Следствие было тише, чем кажется: BT-6 умеет учитывать funding в PnL, но на
реальном прогоне движок получал ``funding=None`` и **молча** считал PnL без
funding-платежей — просто потому, что данные нечем было положить на диск.

Layout повторяет свечи, чтобы RUN-3 читал ряды теми же ключами:

```
{data_root}/{dataset}/exchange=…/market=…/symbol=…/timeframe=…/year=…/month=…
```

``timeframe`` здесь — ключ партиции **и** интервал агрегации Binance для OI/LS
(у funding своей периодичности нет, его расписание задаёт биржа). Поэтому для OI/LS
допустимы только интервалы, которые поддерживает Binance, — см. ``SERIES_PERIODS``.

REG-5 добавил к записи **отчёт о покрытии**: число строк само по себе ничего не
говорит о том, загрузился ли запрошенный диапазон. Там, где период известен
(OI/LS), ожидаемое число точек считается из окна, и неполная загрузка называется
неполной — вместо прежнего «wrote 500 rows», которое выглядело успехом и на
потерянной неделе.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import polars as pl

from trading_research.data.ingestion.market_data_client import (
    PERIOD_MS,
    BinanceMarketDataClient,
)
from trading_research.data.storage.parquet_store import (
    FUNDING,
    LONG_SHORT_RATIO,
    OPEN_INTEREST,
    ParquetStore,
)


class SeriesKind(StrEnum):
    """Что загружаем. ``ALL`` разворачивается в три остальных."""

    FUNDING = FUNDING
    OPEN_INTEREST = OPEN_INTEREST
    LONG_SHORT_RATIO = LONG_SHORT_RATIO
    ALL = "all"


# Интервалы, которые Binance принимает в ``period`` для /futures/data/* (OI и LS).
# У funding интервала нет — расписание выплат задаёт биржа. Набор берётся из
# ``PERIOD_MS``, а не дублируется: загрузчику те же интервалы нужны для нарезки
# диапазона на окна (REG-5), и разъехаться эти два списка не должны.
SERIES_PERIODS: tuple[str, ...] = tuple(PERIOD_MS)

_PERIODIC = (SeriesKind.OPEN_INTEREST, SeriesKind.LONG_SHORT_RATIO)


class CoverageStatus(StrEnum):
    """Насколько загруженный ряд покрывает запрошенное окно."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    # Период задаёт биржа (funding) — ожидаемое число точек посчитать не из чего,
    # поэтому про полноту честнее молчать, чем отвечать «полное».
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class SeriesCoverage:
    """Что фактически загружено по одному ряду за окно ``[start, end)``.

    ``expected`` — ожидаемое число точек; ``None`` для рядов без своего периода.
    """

    rows: int
    first: datetime | None
    last: datetime | None
    expected: int | None
    start: datetime
    end: datetime
    period: timedelta | None

    @property
    def status(self) -> CoverageStatus:
        if self.expected is None or self.period is None:
            return CoverageStatus.UNVERIFIED
        if self.rows < self.expected or self.missing_head or self.missing_tail:
            return CoverageStatus.INCOMPLETE
        return CoverageStatus.COMPLETE

    @property
    def missing_head(self) -> bool:
        """Начало окна не загружено — тот самый симптом REG-5.

        Первая точка должна попасть в первый же период окна: если ``start`` не
        выровнен, она сдвинута меньше чем на период, а потерянное начало даёт
        разрыв в целые периоды.
        """
        if self.period is None:
            return False
        if self.first is None:
            return True
        return self.first - self.start >= self.period

    @property
    def missing_tail(self) -> bool:
        """Конец окна не загружен. Последняя точка — ``end - period``."""
        if self.period is None:
            return False
        if self.last is None:
            return True
        return self.end - self.last > self.period


class MarketDataFetcher(Protocol):
    """Источник рыночных рядов (реальный клиент REG-1 или подставной в тестах)."""

    def fetch_funding_rate(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame: ...

    def fetch_open_interest(
        self, symbol: str, start: datetime, end: datetime, *, period: str
    ) -> pl.DataFrame: ...

    def fetch_long_short_ratio(
        self, symbol: str, start: datetime, end: datetime, *, period: str
    ) -> pl.DataFrame: ...


def expand_series(series: SeriesKind) -> list[SeriesKind]:
    """``ALL`` → три конкретных ряда; остальное — сам ряд."""
    if series is SeriesKind.ALL:
        return [SeriesKind.FUNDING, SeriesKind.OPEN_INTEREST, SeriesKind.LONG_SHORT_RATIO]
    return [series]


def fetch_market_data(
    *,
    exchange: str,
    market: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    series: SeriesKind = SeriesKind.ALL,
    data_root: str | Path,
    fetcher: MarketDataFetcher | None = None,
) -> dict[str, SeriesCoverage]:
    """Загрузить выбранные ряды за ``[start, end)`` и записать в Parquet-store.

    Возвращает покрытие по каждому ряду — не только число строк, потому что число
    строк не отличает полную загрузку от потерянного начала диапазона (REG-5).
    Запись идемпотентна (``write_series`` делает upsert по ``open_time``, как и для
    свечей). ``fetcher=None`` → реальный клиент Binance USDT-M futures.

    Raises:
        ValueError: ``timeframe`` не поддерживается Binance для OI/LS.
    """
    wanted = expand_series(series)
    _validate_period(wanted, timeframe)

    fetcher = fetcher or BinanceMarketDataClient()
    store = ParquetStore(data_root)
    written: dict[str, SeriesCoverage] = {}

    for kind in wanted:
        frame = _fetch_one(fetcher, kind, symbol=symbol, start=start, end=end, period=timeframe)
        store.write_series(
            kind.value,
            frame,
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
        )
        written[kind.value] = _coverage(frame, kind, start=start, end=end, period=timeframe)

    return written


def _coverage(
    frame: pl.DataFrame,
    kind: SeriesKind,
    *,
    start: datetime,
    end: datetime,
    period: str,
) -> SeriesCoverage:
    """Сверить загруженный ряд с запрошенным окном.

    Ожидаемое число точек считается только там, где период известен: у funding его
    задаёт биржа, и выдумывать ожидание для него — значит получить ложную тревогу
    на каждом прогоне.
    """
    # Границы приходят из Typer наивными (``--start 2024-01-01``), а время в
    # загруженном ряде — UTC-aware; без приведения сравнение краёв падает.
    start, end = _as_utc(start), _as_utc(end)

    period_delta = timedelta(milliseconds=PERIOD_MS[period]) if kind in _PERIODIC else None
    expected = int((end - start) / period_delta) if period_delta else None
    times = frame["open_time"].to_list() if frame.height else []
    return SeriesCoverage(
        rows=frame.height,
        first=times[0] if times else None,
        last=times[-1] if times else None,
        expected=expected,
        start=start,
        end=end,
        period=period_delta,
    )


def _as_utc(moment: datetime) -> datetime:
    """Наивное время трактуется как UTC — так же, как в загрузчике (``_to_ms``)."""
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _validate_period(wanted: list[SeriesKind], timeframe: str) -> None:
    """OI и LS агрегируются Binance только по фиксированному набору интервалов.

    Без этой проверки неподдерживаемый интервал возвращает пустой список — то есть
    «успешную» загрузку нуля строк, что выглядит как отсутствие данных, а не как
    ошибка запроса.
    """
    periodic = [k.value for k in wanted if k in _PERIODIC]
    if periodic and timeframe not in SERIES_PERIODS:
        raise ValueError(
            f"timeframe {timeframe!r} is not a valid Binance period for {', '.join(periodic)}; "
            f"supported: {', '.join(SERIES_PERIODS)}"
        )


def _fetch_one(
    fetcher: MarketDataFetcher,
    kind: SeriesKind,
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    period: str,
) -> pl.DataFrame:
    if kind is SeriesKind.FUNDING:
        return fetcher.fetch_funding_rate(symbol, start, end)
    if kind is SeriesKind.OPEN_INTEREST:
        return fetcher.fetch_open_interest(symbol, start, end, period=period)
    if kind is SeriesKind.LONG_SHORT_RATIO:
        return fetcher.fetch_long_short_ratio(symbol, start, end, period=period)
    raise ValueError(f"cannot fetch aggregate series kind {kind!r}")


def describe_coverage(name: str, symbol: str, timeframe: str, coverage: SeriesCoverage) -> str:
    """Строка отчёта по одному ряду — то, что видит глаз вместо «wrote N rows».

    Неполное покрытие называет, сколько точек не хватает и какого края, потому что
    именно край и теряется при хвостовой пагинации.
    """
    head = f"{name} {symbol} {timeframe}:"
    window = f"{_fmt(coverage.first)} … {_fmt(coverage.last)}"

    if coverage.status is CoverageStatus.UNVERIFIED:
        return f"{head} {coverage.rows} rows, {window} (период задаёт биржа, покрытие не сверяется)"
    if coverage.status is CoverageStatus.COMPLETE:
        return f"{head} {coverage.rows} rows, {window}, покрытие полное"

    missing = [
        part
        for part, lost in (("начала", coverage.missing_head), ("конца", coverage.missing_tail))
        if lost
    ]
    edges = f", не хватает {' и '.join(missing)} окна" if missing else ""
    return (
        f"{head} ПОКРЫТИЕ НЕПОЛНОЕ — {coverage.rows} из {coverage.expected} точек, "
        f"получено {window}, запрошено {_fmt(coverage.start)} … {_fmt(coverage.end)}{edges}. "
        f"Записано то, что получено."
    )


def _fmt(moment: datetime | None) -> str:
    return moment.strftime("%Y-%m-%d %H:%M") if moment else "—"


__all__ = [
    "SERIES_PERIODS",
    "CoverageStatus",
    "MarketDataFetcher",
    "SeriesCoverage",
    "SeriesKind",
    "describe_coverage",
    "expand_series",
    "fetch_market_data",
]
