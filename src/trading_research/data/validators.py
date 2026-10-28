"""DATA-3 — валидация качества сырых свечей.

Проверяет гигиену OHLCV до того, как данные попадут в бэктест:

- **Пропуски баров** — gap-report по ожидаемому шагу timeframe (расстояние между
  соседними барами больше одного шага).
- **Дубликаты** времени и **немонотонность** (бары не строго по возрастанию).
- **Нулевые/отрицательные** цены и отрицательный объём.
- **Аномалии цены** — нарушения OHLC-инвариантов (``high``/``low``) и выбросы
  доходности бар-к-бару.

Учитывается дата листинга символа: отсутствие данных до ``listing_at`` — не ошибка
(ведущий пропуск до листинга не попадает в отчёт).

Отчёт — иммутабельная pydantic-модель (`ValidationReport`), сериализуемая в JSON
(`model_dump_json`); помощник `save_report` сохраняет её на диск.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from pydantic import Field

from trading_research.data.schema import timeframe_to_ms
from trading_research.domain import FrozenModel

_REQUIRED = ("open_time", "open", "high", "low", "close", "volume")


class Gap(FrozenModel):
    """Пропуск между соседними барами (или на границе ожидаемого окна)."""

    start: datetime  # последний бар перед пропуском (или начало окна)
    end: datetime  # первый бар после пропуска (или конец окна)
    missing_bars: int = Field(gt=0)


class ValidationReport(FrozenModel):
    """Структурированный отчёт о качестве датасета свечей."""

    timeframe: str
    step_ms: int
    total_bars: int
    gaps: list[Gap] = Field(default_factory=list)
    missing_bars_total: int = 0
    duplicate_times: list[datetime] = Field(default_factory=list)
    is_monotonic: bool = True
    non_positive_prices: int = 0
    negative_volume: int = 0
    ohlc_inconsistencies: list[datetime] = Field(default_factory=list)
    price_outliers: list[datetime] = Field(default_factory=list)
    listing_at: datetime | None = None

    @property
    def is_clean(self) -> bool:
        """Нет ни одного обнаруженного дефекта."""
        return (
            not self.gaps
            and not self.duplicate_times
            and self.is_monotonic
            and self.non_positive_prices == 0
            and self.negative_volume == 0
            and not self.ohlc_inconsistencies
            and not self.price_outliers
        )


def validate_candles(
    df: pl.DataFrame,
    *,
    timeframe: str,
    expected_start: datetime | None = None,
    expected_end: datetime | None = None,
    listing_at: datetime | None = None,
    max_return: float = 0.5,
) -> ValidationReport:
    """Проверить свечи и вернуть отчёт о качестве.

    ``expected_start`` / ``expected_end`` — ожидаемые границы окна (включительно по
    началу): позволяют поймать пропуски на краях, а не только между барами.
    ``listing_at`` сдвигает эффективное начало окна (данных до листинга быть не
    должно, и их отсутствие — не пропуск). ``max_return`` — порог |доходности|
    бар-к-бару для выброса цены.
    """
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"candles missing required columns: {missing}")

    step = timeframe_to_ms(timeframe)
    total = df.height
    listing = _to_utc(listing_at)

    if total == 0:
        return ValidationReport(timeframe=timeframe, step_ms=step, total_bars=0, listing_at=listing)

    ms_orig: list[int] = df["open_time"].dt.epoch("ms").to_list()

    # Монотонность (строго по возрастанию) — в исходном порядке.
    is_monotonic = all(a < b for a, b in zip(ms_orig, ms_orig[1:], strict=False))

    # Дубликаты времени.
    counts = Counter(ms_orig)
    duplicate_times = sorted(_from_ms(m) for m, c in counts.items() if c > 1)

    # Пропуски — на отсортированных уникальных временах + границы окна.
    uniq = sorted(counts)
    gaps = _detect_gaps(
        uniq,
        step=step,
        expected_start=_effective_start(expected_start, listing),
        expected_end=_to_utc(expected_end),
    )

    # Цены/объём.
    non_positive_prices = df.filter(
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
    ).height
    negative_volume = df.filter(pl.col("volume") < 0).height

    # OHLC-инварианты: high — максимум бара, low — минимум.
    hi_lo = pl.max_horizontal("open", "close")
    lo_hi = pl.min_horizontal("open", "close")
    ohlc_bad = df.filter(
        (pl.col("high") < pl.col("low")) | (pl.col("high") < hi_lo) | (pl.col("low") > lo_hi)
    )
    ohlc_inconsistencies = sorted(ohlc_bad["open_time"].to_list())

    # Выбросы доходности close бар-к-бару (во временном порядке).
    ordered = df.sort("open_time")
    ret = ordered.select(
        pl.col("open_time"),
        (pl.col("close") / pl.col("close").shift(1) - 1.0).abs().alias("ret"),
    ).filter(pl.col("ret") > max_return)
    price_outliers = sorted(ret["open_time"].to_list())

    return ValidationReport(
        timeframe=timeframe,
        step_ms=step,
        total_bars=total,
        gaps=gaps,
        missing_bars_total=sum(g.missing_bars for g in gaps),
        duplicate_times=duplicate_times,
        is_monotonic=is_monotonic,
        non_positive_prices=non_positive_prices,
        negative_volume=negative_volume,
        ohlc_inconsistencies=ohlc_inconsistencies,
        price_outliers=price_outliers,
        listing_at=listing,
    )


def save_report(report: ValidationReport, path: str | Path) -> None:
    """Сохранить отчёт в JSON (UTF-8, с отступами)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def _detect_gaps(
    uniq_ms: list[int],
    *,
    step: int,
    expected_start: datetime | None,
    expected_end: datetime | None,
) -> list[Gap]:
    gaps: list[Gap] = []
    if not uniq_ms:
        return gaps

    # Ведущий пропуск: окно начинается раньше первого бара.
    if expected_start is not None:
        start_ms = _epoch_ms(expected_start)
        first = uniq_ms[0]
        if first - start_ms >= step:
            gaps.append(
                Gap(
                    start=expected_start,
                    end=_from_ms(first),
                    missing_bars=(first - start_ms) // step,
                )
            )

    # Внутренние пропуски между соседними барами.
    for a, b in zip(uniq_ms, uniq_ms[1:], strict=False):
        if b - a > step:
            gaps.append(Gap(start=_from_ms(a), end=_from_ms(b), missing_bars=(b - a) // step - 1))

    # Замыкающий пропуск: окно заканчивается после последнего бара.
    if expected_end is not None:
        end_ms = _epoch_ms(expected_end)
        last = uniq_ms[-1]
        if end_ms - last >= step:
            gaps.append(
                Gap(start=_from_ms(last), end=expected_end, missing_bars=(end_ms - last) // step)
            )

    return gaps


def _effective_start(expected_start: datetime | None, listing: datetime | None) -> datetime | None:
    """Эффективное начало окна с учётом листинга (данных до него быть не должно)."""
    start = _to_utc(expected_start)
    if start is None:
        return None
    if listing is not None and listing > start:
        return listing
    return start


def _to_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _epoch_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


__all__ = ["Gap", "ValidationReport", "save_report", "validate_candles"]
