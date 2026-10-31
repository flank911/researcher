"""OPS-2 — инкрементальная догрузка свечей от последнего сохранённого бара.

``fetch-data`` всегда запрашивает весь указанный диапазон, даже если он уже лежит в
store. Запись при этом идемпотентна (``ParquetStore._upsert`` снимает дубли по
``open_time``), так что повторный прогон данные не портит — он платит сетью и
временем: наполнение текущего состояния (156 060 свечей, 6 пар) заняло 87 с, и
столько же занимало бы любое «добить до сегодня».

Отличия от [OPS-1](../../../tickets/OPS-1.md), который делает то же для OI/LS:

- **Retention нет.** Свечи Binance отдаёт за всю историю, поэтому пропуск здесь
  восполним в любой момент и «невосполнимого разрыва» в отчёте не бывает.
- **Период известен всегда** — это таймфрейм пары, а не присланное биржей
  расписание, так что догружать можно от точной границы бара.

Что всё же совпадает — и не является удобством:

**Правая граница — начало текущего бара, а не ``now``.** Binance отдаёт и бар в
процессе формирования, у которого ``high``/``low``/``close``/``volume`` ещё не
окончательны. Записать такой бар особенно опасно именно здесь: следующий запуск
начнёт с ``последний + период``, то есть к незакрытому бару больше не вернётся, и
частичные значения остались бы в store навсегда. Поэтому окно заканчивается на
``floor(now, период)`` — в него попадают только закрытые бары, а текущий подберёт
следующий запуск.

**Правая граница — ``now``, а не ``data.end`` конфига.** Из конфига берётся набор
пар и начало истории для пустого store; конец — это исследовательское окно, и
останавливаться на нём значило бы, что команда не умеет делать ровно то, ради чего
нужна. Диапазон эксперимента отрезает чтение (``DuckDBStore.read_window``), а не
загрузка.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trading_research.cli.fetch_data import KlineFetcher, fetch_data
from trading_research.cli.text import plural
from trading_research.config import ExperimentConfig, load_experiment_config
from trading_research.data.ingestion.binance_client import BinanceFuturesClient
from trading_research.data.schema import timeframe_to_ms
from trading_research.data.storage.parquet_store import ParquetStore


@dataclass(frozen=True)
class CandleUpdate:
    """Итог догрузки свечей по одной паре символ × таймфрейм."""

    symbol: str
    timeframe: str
    period: timedelta
    stored_until: datetime | None
    """Последний бар в store до догрузки; ``None`` — пары в store не было."""
    requested_from: datetime
    requested_to: datetime
    rows: int = 0
    requested: bool = False
    """Был ли запрос к API. ``False`` — пара уже актуальна, окно пустое."""

    @property
    def is_up_to_date(self) -> bool:
        return not self.requested

    @property
    def is_first_load(self) -> bool:
        return self.stored_until is None

    @property
    def expected(self) -> int:
        """Сколько закрытых баров укладывается в запрошенное окно."""
        span = self.requested_to - self.requested_from
        return max(0, int(span / self.period))

    @property
    def is_short(self) -> bool:
        """Дописано меньше, чем укладывается в окно.

        На первой загрузке это норма: символ мог начать торговаться позже начала
        конфига, и требовать полное окно значило бы ругаться на каждый новый тикер.
        А вот на догрузке хвоста недобор означает дырку в свежей истории — её стоит
        увидеть сразу, а не на следующем ``validate-data``.
        """
        return self.requested and not self.is_first_load and self.rows < self.expected


def update_data_from_config(
    config_path: str | Path,
    *,
    data_root: str | Path,
    now: datetime | None = None,
    fetcher: KlineFetcher | None = None,
) -> list[CandleUpdate]:
    """Догрузить свечи по конфигу от последнего сохранённого бара до текущего.

    Набор пар символ × таймфрейм берётся из конфига — одна команда закрывает весь
    набор, и расширение покрытия (новый символ, новый таймфрейм) делается правкой
    конфига, а не подбором окна руками.

    Пустой store не является особым случаем: начало берётся из ``data.start``, то
    есть новая пара загружается с нуля той же командой.
    """
    cfg = load_experiment_config(config_path)
    moment = _as_utc(now or _now())
    fetcher = fetcher or BinanceFuturesClient()
    store = ParquetStore(data_root)

    return [
        _update_one(
            symbol=symbol,
            timeframe=timeframe,
            cfg=cfg,
            now=moment,
            store=store,
            data_root=data_root,
            fetcher=fetcher,
        )
        for symbol in cfg.data.symbols
        for timeframe in cfg.data.timeframes
    ]


def _update_one(
    *,
    symbol: str,
    timeframe: str,
    cfg: ExperimentConfig,
    now: datetime,
    store: ParquetStore,
    data_root: str | Path,
    fetcher: KlineFetcher,
) -> CandleUpdate:
    period = timedelta(milliseconds=timeframe_to_ms(timeframe))
    stored_until = _last_stored(store, cfg=cfg, symbol=symbol, timeframe=timeframe)
    requested_from = _as_utc(cfg.data.start) if stored_until is None else stored_until + period
    requested_to = _right_edge(now, period=period)

    common = {
        "symbol": symbol,
        "timeframe": timeframe,
        "period": period,
        "stored_until": stored_until,
        "requested_from": requested_from,
        "requested_to": requested_to,
    }
    # Пустое окно — не повод звать API: повторный запуск без новых данных не должен
    # ни ходить в сеть, ни трогать store.
    if requested_to <= requested_from:
        return CandleUpdate(**common)  # type: ignore[arg-type]

    rows = fetch_data(
        exchange=cfg.data.exchange,
        market=cfg.data.market,
        symbol=symbol,
        timeframe=timeframe,
        start=requested_from,
        end=requested_to,
        data_root=data_root,
        fetcher=fetcher,
    )
    return CandleUpdate(**common, rows=rows, requested=True)  # type: ignore[arg-type]


def _last_stored(
    store: ParquetStore, *, cfg: ExperimentConfig, symbol: str, timeframe: str
) -> datetime | None:
    """Последний сохранённый бар пары, либо ``None``, если пары в store нет."""
    frame = store.read_candles(
        exchange=cfg.data.exchange,
        market=cfg.data.market,
        symbol=symbol,
        timeframe=timeframe,
    )
    if frame.is_empty():
        return None
    last = frame["open_time"].max()
    return _as_utc(last) if isinstance(last, datetime) else None


def _right_edge(now: datetime, *, period: timedelta) -> datetime:
    """Эксклюзивная правая граница окна — начало текущего (ещё открытого) бара."""
    step = int(period.total_seconds() * 1000)
    floored = (int(now.timestamp() * 1000) // step) * step
    return datetime.fromtimestamp(floored / 1000, UTC)


def _as_utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _now() -> datetime:
    """Шов для тестов — тот же приём, что в OPS-1: флаг ``--now`` нужен только им."""
    return datetime.now(UTC)


# --- отчёт ------------------------------------------------------------------


def describe_update(update: CandleUpdate) -> str:
    """Строка по одной паре: что было в store и что дописано.

    Объём запроса назван явно — иначе «догрузил хвост» и «перекачал годы» выглядят
    в выводе одинаково, а ради первого команда и существует.
    """
    head = f"{update.symbol} {update.timeframe}:"
    if update.is_up_to_date:
        return f"{head} актуален до {_fmt(update.stored_until)}, догружать нечего"

    before = (
        "пары не было в store" if update.is_first_load else f"было до {_fmt(update.stored_until)}"
    )
    line = (
        f"{head} {before}, запрошено {_fmt(update.requested_from)} … "
        f"{_fmt(update.requested_to)}, дописано {_bars(update.rows)}"
    )
    if update.is_short:
        line += f" — НЕДОБОР: в окно укладывается {_bars(update.expected)}"
    return line


def describe_total(updates: list[CandleUpdate]) -> str:
    """Итог прогона — то, на что смотрят после «добить до сегодня»."""
    if not updates:
        return "нечего догружать: в конфиге нет пар"
    rows = sum(u.rows for u in updates)
    fresh = [u for u in updates if u.is_up_to_date]
    first = [u for u in updates if u.requested and u.is_first_load]
    short = [u for u in updates if u.is_short]

    total = f"итого: {_bars(rows)} по {plural(len(updates), 'паре', 'парам', 'парам')}"
    if fresh:
        total += f", уже актуальны: {len(fresh)}"
    if first:
        total += f", загружено с нуля: {len(first)}"
    total += f", недобор: {len(short)}" if short else ", недобора нет"
    return total


def has_defects(updates: list[CandleUpdate]) -> bool:
    """Есть ли то, из-за чего прогон нельзя считать успешным."""
    return any(u.is_short for u in updates)


def _fmt(moment: datetime | None) -> str:
    return moment.strftime("%Y-%m-%d %H:%M") if moment else "—"


def _bars(count: int) -> str:
    return plural(count, "бар", "бара", "баров")


__all__ = [
    "CandleUpdate",
    "describe_total",
    "describe_update",
    "has_defects",
    "update_data_from_config",
]
