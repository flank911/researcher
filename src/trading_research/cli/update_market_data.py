"""OPS-1 — инкрементальная догрузка рыночных рядов от последней сохранённой точки.

Отличие от ``fetch-market-data`` не в удобстве. Binance отдаёт open interest и
long/short ratio **только за последние 30 дней** (замерено: 29 дней назад эндпоинт
отвечает, 30 — уже ``-1130``). Значит история этих двух рядов не загружается, а
**создаётся** регулярным сбором: каждый день без сбора — день, которого в датасете
не будет никогда. Свечи и funding доступны за всю историю и этим свойством не
обладают.

Отсюда три требования, которых нет у разовой загрузки:

1. Начало окна берётся из store (последняя сохранённая точка), а не из аргументов —
   иначе расписание пришлось бы переписывать после каждого запуска.
2. Невосполнимый пропуск называется вслух. Если последняя точка в store старше окна
   retention, разрыв между ней и доступным началом уже не закрыть ничем — и это
   единственная ошибка в проекте, которую нельзя исправить задним числом.
3. Правая граница — **предыдущая** полная точка (``floor(now, period)``), а не
   ``now``: точка текущего периода может быть ещё не опубликована, и её отсутствие
   выглядело бы как неполное покрытие на каждом запуске. Следующий запуск её
   подберёт.

Запись идемпотентна (``write_series`` делает upsert по ``open_time``), поэтому
повторный запуск без новых данных ничего не портит и не дублирует.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trading_research.cli.fetch_market_data import (
    CoverageStatus,
    MarketDataFetcher,
    SeriesCoverage,
    SeriesKind,
    clamp_to_retention,
    expand_series,
    fetch_market_data,
)
from trading_research.cli.text import points
from trading_research.config import ExperimentConfig, load_experiment_config
from trading_research.data.ingestion.market_data_client import (
    PERIOD_MS,
    RETENTION,
    BinanceMarketDataClient,
)
from trading_research.data.storage.parquet_store import ParquetStore


@dataclass(frozen=True)
class SeriesUpdate:
    """Итог догрузки одного ряда по одной паре символ × таймфрейм."""

    series: str
    symbol: str
    timeframe: str
    stored_until: datetime | None
    """Последняя точка в store до догрузки; ``None`` — store пуст."""
    resume_from: datetime
    """Откуда следовало бы продолжить, если бы окно retention не мешало."""
    requested_from: datetime
    requested_to: datetime
    coverage: SeriesCoverage | None
    """``None`` — ряд уже актуален, запроса не было.

    Такой ряд всё равно попадает в отчёт: строка на каждую пару нужна, чтобы
    «актуален» и «выпал из сбора» не выглядели одинаково — тишиной.
    """

    @property
    def is_up_to_date(self) -> bool:
        return self.coverage is None

    @property
    def lost(self) -> timedelta | None:
        """Невосполнимый пропуск: окно retention съело часть истории.

        Это не «ещё не загружено», а «загрузить больше нельзя» — разрыв между тем,
        откуда следовало продолжить, и тем, откуда вообще можно.
        """
        gap = self.requested_from - self.resume_from
        return gap if gap > timedelta(0) else None

    @property
    def is_first_load(self) -> bool:
        return self.stored_until is None

    @property
    def rows(self) -> int:
        return self.coverage.rows if self.coverage else 0

    @property
    def is_incomplete(self) -> bool:
        return self.coverage is not None and self.coverage.status is CoverageStatus.INCOMPLETE


def update_market_data_from_config(
    config_path: str | Path,
    *,
    data_root: str | Path,
    series: SeriesKind = SeriesKind.ALL,
    now: datetime | None = None,
    fetcher: MarketDataFetcher | None = None,
) -> list[SeriesUpdate]:
    """Догрузить рыночные ряды по конфигу от последней сохранённой точки до ``now``.

    Набор пар символ × таймфрейм берётся из конфига, а не из аргументов: состав
    сбора должен меняться правкой конфига, а не правкой расписания.

    Raises:
        ValueError: таймфрейм конфига не является периодом Binance для OI/LS —
            собирать эти ряды по нему нельзя вообще, и молчаливый пропуск оставил
            бы дыру, которую потом не заметить.
    """
    cfg = load_experiment_config(config_path)
    now = _as_utc(now or _now())
    fetcher = fetcher or BinanceMarketDataClient()
    store = ParquetStore(data_root)

    updates: list[SeriesUpdate] = []
    for kind in expand_series(series):
        for symbol in cfg.data.symbols:
            for timeframe in cfg.data.timeframes:
                update = _update_one(
                    kind,
                    symbol=symbol,
                    timeframe=timeframe,
                    cfg=cfg,
                    now=now,
                    store=store,
                    data_root=data_root,
                    fetcher=fetcher,
                )
                updates.append(update)
    return updates


def _update_one(
    kind: SeriesKind,
    *,
    symbol: str,
    timeframe: str,
    cfg: ExperimentConfig,
    now: datetime,
    store: ParquetStore,
    data_root: str | Path,
    fetcher: MarketDataFetcher,
) -> SeriesUpdate:
    """Догрузить один ряд по одной паре."""
    stored_until = _last_stored(store, kind, cfg=cfg, symbol=symbol, timeframe=timeframe)
    period = _period(kind, timeframe)

    resume_from = _resume_from(stored_until, cfg=cfg, period=period)
    requested_from = clamp_to_retention(resume_from, kind=kind, now=now)
    requested_to = _right_edge(now, period=period)

    common = {
        "series": kind.value,
        "symbol": symbol,
        "timeframe": timeframe,
        "stored_until": stored_until,
        "resume_from": resume_from,
        "requested_from": requested_from,
        "requested_to": requested_to,
    }
    if requested_to <= requested_from:
        return SeriesUpdate(**common, coverage=None)  # type: ignore[arg-type]

    written = fetch_market_data(
        exchange=cfg.data.exchange,
        market=cfg.data.market,
        symbol=symbol,
        timeframe=timeframe,
        start=requested_from,
        end=requested_to,
        series=kind,
        data_root=data_root,
        fetcher=fetcher,
    )
    return SeriesUpdate(**common, coverage=written[kind.value])  # type: ignore[arg-type]


def _last_stored(
    store: ParquetStore,
    kind: SeriesKind,
    *,
    cfg: ExperimentConfig,
    symbol: str,
    timeframe: str,
) -> datetime | None:
    """Последняя сохранённая точка ряда, либо ``None`` на пустом store.

    ``read_series`` на пустом store возвращает DataFrame **без колонок** (тот же
    нюанс, что обойдён в ``build_features``), поэтому наличие ``open_time``
    проверяется, а не предполагается.
    """
    frame = store.read_series(
        kind.value,
        exchange=cfg.data.exchange,
        market=cfg.data.market,
        symbol=symbol,
        timeframe=timeframe,
    )
    if frame.is_empty() or "open_time" not in frame.columns:
        return None
    last = frame["open_time"].max()
    return _as_utc(last) if isinstance(last, datetime) else None


def _period(kind: SeriesKind, timeframe: str) -> timedelta | None:
    """Период агрегации ряда. ``None`` у funding — его расписание задаёт биржа."""
    if kind not in (SeriesKind.OPEN_INTEREST, SeriesKind.LONG_SHORT_RATIO):
        return None
    try:
        return timedelta(milliseconds=PERIOD_MS[timeframe])
    except KeyError:
        raise ValueError(
            f"timeframe {timeframe!r} is not a valid Binance period for {kind.value}; "
            f"supported: {', '.join(PERIOD_MS)}"
        ) from None


def _resume_from(
    stored_until: datetime | None, *, cfg: ExperimentConfig, period: timedelta | None
) -> datetime:
    """Откуда продолжать: за последней точкой, а на пустом store — от начала конфига.

    Шаг за последнюю точку нужен, чтобы не перезапрашивать её каждый раз: upsert
    дубль бы снял, но число дописанных строк врало бы на единицу, а именно по нему
    и смотрят, жив ли сбор.
    """
    if stored_until is None:
        return _as_utc(cfg.data.start)
    return stored_until + (period or timedelta(milliseconds=1))


def _right_edge(now: datetime, *, period: timedelta | None) -> datetime:
    """Правая граница окна (эксклюзивная) — начало текущего периода.

    Точка текущего периода может быть не опубликована, поэтому за ней не идём:
    иначе каждый запуск отчитывался бы о неполном покрытии на одну точку.
    """
    if period is None:
        return now
    step = int(period.total_seconds() * 1000)
    floored = (int(now.timestamp() * 1000) // step) * step
    return datetime.fromtimestamp(floored / 1000, UTC)


def _as_utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _now() -> datetime:
    """Текущее время одной точкой — её подменяют тесты про окно retention.

    Подмена самого ``datetime`` в модуле ломала бы ``isinstance``-проверки, а
    отдельный CLI-флаг ``--now`` добавил бы в интерфейс параметр, нужный только
    тестам.
    """
    return datetime.now(UTC)


# --- отчёт ------------------------------------------------------------------


def describe_update(update: SeriesUpdate) -> str:
    """Строка отчёта по одной паре: что было, что дописано, что потеряно.

    Молчаливый успех здесь недопустим: провал расписания обнаружится через месяц,
    когда данные уже потеряны, поэтому каждая строка говорит и про остаточный
    разрыв, и про невосполнимый пропуск.
    """
    head = f"{update.series} {update.symbol} {update.timeframe}:"
    if update.is_up_to_date:
        return f"{head} актуален до {_fmt(update.stored_until)}, догружать нечего"

    assert update.coverage is not None  # noqa: S101 — следствие is_up_to_date выше
    before = "store пуст" if update.is_first_load else f"было до {_fmt(update.stored_until)}"
    added = f"дописано {points(update.rows)}"
    if update.coverage.last is not None:
        added += f" до {_fmt(update.coverage.last)}"

    parts = [head, before + ",", added]
    if (lost := update.lost) is not None:
        parts.append(
            f"— НЕВОСПОЛНИМЫЙ ПРОПУСК {_fmt_delta(lost)} "
            f"({_fmt(update.resume_from)} … {_fmt(update.requested_from)}), "
            f"окно биржи {RETENTION.days} д"
        )
    if update.is_incomplete:
        parts.append(
            f"— покрытие неполное: {update.coverage.rows} из {update.coverage.expected} точек"
        )
    return " ".join(parts)


def describe_total(updates: list[SeriesUpdate]) -> str:
    """Итог одной итерации сбора — то, на что смотрит проверка «расписание живо»."""
    if not updates:
        return "нечего догружать: в конфиге нет пар"
    rows = sum(u.rows for u in updates)
    lost = [u for u in updates if u.lost is not None]
    incomplete = [u for u in updates if u.is_incomplete]
    fresh = [u for u in updates if u.is_up_to_date]

    total = f"итого: {points(rows)} по {len(updates)} рядам"
    if fresh:
        total += f", уже актуальны: {len(fresh)}"
    if lost:
        total += f", невосполнимых пропусков: {len(lost)}"
    if incomplete:
        total += f", неполное покрытие: {len(incomplete)}"
    if not lost and not incomplete:
        total += ", пропусков нет"
    return total


def has_defects(updates: list[SeriesUpdate]) -> bool:
    """Есть ли то, из-за чего итерацию нельзя считать успешной."""
    return any(u.lost is not None or u.is_incomplete for u in updates)


def _fmt(moment: datetime | None) -> str:
    return moment.strftime("%Y-%m-%d %H:%M") if moment else "—"


def _fmt_delta(delta: timedelta) -> str:
    days, seconds = delta.days, delta.seconds
    hours = seconds // 3600
    return f"{days} д {hours} ч" if days else f"{hours} ч"


__all__ = [
    "SeriesUpdate",
    "describe_total",
    "describe_update",
    "has_defects",
    "update_market_data_from_config",
]
