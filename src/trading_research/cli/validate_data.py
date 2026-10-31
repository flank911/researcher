"""DATA-8 — бизнес-логика команды ``validate-data``.

DATA-3 дал `validate_candles`, `validate_series`, `ValidationReport` и
`save_report`, но вызвать их можно было только из Python. На прогоне
[RUN-7](../../../tickets/RUN-7.md) это кончилось одноразовым скриптом: прочитать
store, обойти пары символ × таймфрейм, позвать валидатор, распечатать отчёт. Тот
же разрыв «библиотека есть, CLI-пути нет», что закрывали RUN-1…RUN-5.

Что здесь важнее механики обхода:

**Шаг ряда — не всегда таймфрейм партиции.** Для свечей и для OI/LS таймфрейм и
есть период агрегации: его мы сами просили у биржи. У funding расписание выплат
задаёт биржа, а ``timeframe`` в пути хранения — только ключ партиции. Проверять
funding с шагом ``1h`` значило бы отчитаться о пропуске на каждой второй точке,
поэтому шаг funding выводится из самих данных (`infer_step`).

**Окно сверки для OI/LS урезано retention'ом.** Биржа отдаёт эти ряды только за
последние ~30 дней (REG-4), поэтому ожидаемое начало окна считается тем же
`clamp_to_retention`, что и при загрузке. Иначе каждая проверка показывала бы
многолетний «пропуск» там, где истории физически не существует, и настоящие
дефекты утонули бы в этом шуме.

**Отсутствие ряда — не дефект.** funding/OI/LS опциональны по контракту
`build_features`: без них фичи остаются ``null``. Строка «нет данных» в выводе
это называет, но ``--strict`` на ней не падает — пустую фичу ловит REG-4 на своём
шаге. А вот пустые свечи — дефект: прогонять эксперимент не на чем.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from trading_research.cli.fetch_market_data import SeriesKind, clamp_to_retention
from trading_research.cli.text import plural
from trading_research.config import load_experiment_config
from trading_research.data.schema import TIMEFRAME_MS
from trading_research.data.storage.duckdb_store import DuckDBStore
from trading_research.data.storage.parquet_store import (
    CANDLES,
    FUNDING,
    LONG_SHORT_RATIO,
    OPEN_INTEREST,
    ParquetStore,
)
from trading_research.data.validators import (
    SeriesValidationReport,
    ValidationReport,
    save_report,
    validate_candles,
    validate_series,
)

# Допуск при сопоставлении измеренного шага с таймфреймом: времена биржи плавают
# на единицы миллисекунд, и 8 ч + 1 мс обязан опознаться как «8h».
_STEP_TOLERANCE = 0.01


@dataclass(frozen=True)
class SeriesSpec:
    """Какой ряд проверяем и какие его колонки имеют смысл проверять."""

    dataset: str
    kind: SeriesKind
    value_columns: tuple[str, ...]
    non_negative_columns: tuple[str, ...]
    # Шаг задаёт биржа, а не мы: таймфрейм партиции про него ничего не говорит.
    step_from_data: bool = False


SERIES_SPECS: tuple[SeriesSpec, ...] = (
    SeriesSpec(
        dataset=FUNDING,
        kind=SeriesKind.FUNDING,
        value_columns=("funding_rate",),
        # funding бывает отрицательным — это нормальная ставка, а не дефект.
        non_negative_columns=(),
        step_from_data=True,
    ),
    SeriesSpec(
        dataset=OPEN_INTEREST,
        kind=SeriesKind.OPEN_INTEREST,
        value_columns=("open_interest", "open_interest_value"),
        non_negative_columns=("open_interest", "open_interest_value"),
    ),
    SeriesSpec(
        dataset=LONG_SHORT_RATIO,
        kind=SeriesKind.LONG_SHORT_RATIO,
        value_columns=("long_short_ratio", "long_account", "short_account"),
        non_negative_columns=("long_short_ratio", "long_account", "short_account"),
    ),
)


@dataclass(frozen=True)
class DatasetCheck:
    """Результат проверки одного датасета одной пары символ × таймфрейм."""

    dataset: str
    report: ValidationReport | SeriesValidationReport | None = None
    path: Path | None = None
    step: str | None = None
    # Почему проверка не делалась (ряда нет, шаг не опознан, окно вне retention).
    note: str | None = None

    @property
    def is_candles(self) -> bool:
        return self.dataset == CANDLES

    @property
    def rows(self) -> int:
        if isinstance(self.report, ValidationReport):
            return self.report.total_bars
        return self.report.total_rows if self.report else 0

    @property
    def is_missing(self) -> bool:
        """Данных нет совсем."""
        return self.report is None or self.rows == 0

    @property
    def is_defect(self) -> bool:
        """Считается ли результат провалом для ``--strict``.

        Отсутствующий ряд — не дефект (см. докстринг модуля), отсутствующие
        свечи — дефект.
        """
        if self.is_missing:
            return self.is_candles
        assert self.report is not None  # noqa: S101 — следствие is_missing выше
        return not self.report.is_clean


@dataclass(frozen=True)
class PairValidation:
    """Все проверки одной пары символ × таймфрейм."""

    symbol: str
    timeframe: str
    checks: list[DatasetCheck]

    @property
    def is_defect(self) -> bool:
        return any(c.is_defect for c in self.checks)


def validate_data_from_config(
    config_path: str | Path,
    *,
    data_root: str | Path,
    reports_root: str | Path,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    now: datetime | None = None,
) -> list[PairValidation]:
    """Проверить качество загруженных данных по конфигу эксперимента.

    ``symbols`` / ``timeframes`` сужают набор пар; диапазон, биржа и рынок всегда
    берутся из конфига — проверять надо ровно то окно, на котором пойдёт прогон.
    ``now`` — точка отсчёта окна retention для OI/LS (шов для тестов).
    """
    cfg = load_experiment_config(config_path)
    moment = now or _now()
    pairs_symbols = symbols or cfg.data.symbols
    pairs_timeframes = timeframes or cfg.data.timeframes

    candle_store = DuckDBStore(data_root)
    series_store = ParquetStore(data_root)
    out_dir = Path(reports_root) / cfg.name
    exchange, market = cfg.data.exchange, cfg.data.market

    result: list[PairValidation] = []
    for symbol in pairs_symbols:
        for timeframe in pairs_timeframes:
            candles = candle_store.read_window(
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                start=cfg.data.start,
                end=cfg.data.end,
            )
            checks = [
                _check_candles(
                    candles,
                    timeframe=timeframe,
                    start=cfg.data.start,
                    end=cfg.data.end,
                    path=_report_path(out_dir, symbol, timeframe, CANDLES),
                )
            ]
            for spec in SERIES_SPECS:
                frame = series_store.read_series(
                    spec.dataset,
                    exchange=exchange,
                    market=market,
                    symbol=symbol,
                    timeframe=timeframe,
                )
                checks.append(
                    _check_series(
                        frame,
                        spec=spec,
                        timeframe=timeframe,
                        start=cfg.data.start,
                        end=cfg.data.end,
                        now=moment,
                        path=_report_path(out_dir, symbol, timeframe, spec.dataset),
                    )
                )
            result.append(PairValidation(symbol=symbol, timeframe=timeframe, checks=checks))
    return result


def infer_step(frame: pl.DataFrame) -> str | None:
    """Опознать шаг ряда по самой частой разнице соседних времён.

    Нужен для funding: период выплат задаёт биржа (сейчас 8 ч, исторически
    менялся), и в пути хранения его нет — там таймфрейм партиции. Возвращает
    ``None``, если шаг не похож ни на один поддерживаемый таймфрейм: тогда
    проверка пропусков честно пропускается, а не делается по выдуманному шагу.
    """
    if frame.height < 2 or "open_time" not in frame.columns:
        return None
    ms: list[int] = frame["open_time"].dt.epoch("ms").sort().to_list()
    deltas = Counter(b - a for a, b in zip(ms, ms[1:], strict=False))
    modal, _ = deltas.most_common(1)[0]
    name, length = min(TIMEFRAME_MS.items(), key=lambda item: abs(item[1] - modal))
    return name if abs(length - modal) <= length * _STEP_TOLERANCE else None


def describe_pair(pair: PairValidation) -> list[str]:
    """Строки отчёта по одной паре: заголовок и по строке на датасет."""
    return [
        f"{pair.symbol} {pair.timeframe}:",
        *(f"  {_describe_check(check)}" for check in pair.checks),
    ]


def describe_total(pairs: list[PairValidation], reports_root: Path | None = None) -> str:
    """Итог проверки — то, на что смотрят перед запуском прогона."""
    if not pairs:
        return "проверять нечего: в конфиге нет пар"
    bad = [p for p in pairs if p.is_defect]
    total = f"итого: {plural(len(pairs), 'пара', 'пары', 'пар')}"
    total += f", с дефектами: {len(bad)}" if bad else ", дефектов нет"
    if reports_root is not None:
        total += f", отчёты -> {reports_root}"
    return total


def has_defects(pairs: list[PairValidation]) -> bool:
    """Есть ли то, из-за чего ``--strict`` должен дать ненулевой код."""
    return any(p.is_defect for p in pairs)


def _check_candles(
    frame: pl.DataFrame,
    *,
    timeframe: str,
    start: datetime,
    end: datetime,
    path: Path,
) -> DatasetCheck:
    if frame.is_empty():
        return DatasetCheck(dataset=CANDLES, step=timeframe, note="нет данных")
    report = validate_candles(frame, timeframe=timeframe, expected_start=start, expected_end=end)
    save_report(report, path)
    return DatasetCheck(dataset=CANDLES, report=report, path=path, step=timeframe)


def _check_series(
    frame: pl.DataFrame,
    *,
    spec: SeriesSpec,
    timeframe: str,
    start: datetime,
    end: datetime,
    now: datetime,
    path: Path,
) -> DatasetCheck:
    if frame.is_empty() or "open_time" not in frame.columns:
        return DatasetCheck(dataset=spec.dataset, note="нет данных")

    step = infer_step(frame) if spec.step_from_data else timeframe
    if step is None:
        return DatasetCheck(
            dataset=spec.dataset,
            note=f"{frame.height}: шаг ряда не опознан, пропуски не проверялись",
        )

    # Для OI/LS ожидаемое начало — не начало конфига, а граница retention: более
    # старой истории у биржи не существует, и считать её пропуском нечестно.
    expected_start: datetime | None = clamp_to_retention(start, kind=spec.kind, now=now)
    expected_end: datetime | None = end
    if expected_start is not None and expected_start >= _as_utc(end):
        # Диапазон конфига целиком старше окна биржи: сверять не с чем, но то,
        # что лежит в store, всё равно стоит проверить на дубли и монотонность.
        expected_start = expected_end = None

    report = validate_series(
        frame,
        timeframe=step,
        value_columns=_float_columns(frame, spec.value_columns),
        non_negative_columns=_float_columns(frame, spec.non_negative_columns),
        expected_start=expected_start,
        expected_end=expected_end,
    )
    save_report(report, path)
    note = None if expected_start is not None else "окно конфига старше retention"
    return DatasetCheck(dataset=spec.dataset, report=report, path=path, step=step, note=note)


def _float_columns(frame: pl.DataFrame, columns: tuple[str, ...]) -> tuple[str, ...]:
    """Отобрать существующие вещественные колонки.

    ``validate_series`` проверяет их на NaN/inf, а ``is_nan`` на целочисленной
    колонке падает — целые значения в этой проверке и не нуждаются.
    """
    return tuple(c for c in columns if c in frame.columns and frame.schema[c].is_float())


def _describe_check(check: DatasetCheck) -> str:
    name = "свечи" if check.is_candles else check.dataset
    head = f"{name} ({check.step}):" if check.step else f"{name}:"

    if check.report is None:
        return f"{head} {check.note}"

    unit = ("бар", "бара", "баров") if check.is_candles else ("точка", "точки", "точек")
    parts = [head, plural(check.rows, *unit)]
    defects = _defect_parts(check.report, unit)
    parts.append("— " + (", ".join(defects) if defects else "чисто"))
    if check.note:
        parts.append(f"({check.note})")
    if check.path is not None:
        parts.append(f"-> {check.path}")
    return " ".join(parts)


def _defect_parts(
    report: ValidationReport | SeriesValidationReport, unit: tuple[str, str, str]
) -> list[str]:
    """Перечислить найденные дефекты — то, на что смотрят глазами."""
    parts: list[str] = []
    if report.gaps:
        missing = (
            report.missing_bars_total
            if isinstance(report, ValidationReport)
            else report.missing_total
        )
        parts.append(f"пропусков {len(report.gaps)} ({plural(missing, *unit)})")
    if report.duplicate_times:
        parts.append(f"дублей {len(report.duplicate_times)}")
    if not report.is_monotonic:
        parts.append("немонотонно")

    if isinstance(report, ValidationReport):
        if report.non_positive_prices:
            parts.append(f"неположительных цен {report.non_positive_prices}")
        if report.negative_volume:
            parts.append(f"отрицательный объём {report.negative_volume}")
        if report.ohlc_inconsistencies:
            parts.append(f"нарушений OHLC {len(report.ohlc_inconsistencies)}")
        if report.price_outliers:
            parts.append(f"выбросов цены {len(report.price_outliers)}")
    else:
        if report.non_finite:
            parts.append(f"NaN/inf {report.non_finite}")
        if report.negative:
            parts.append(f"отрицательных значений {report.negative}")
    return parts


def _report_path(out_dir: Path, symbol: str, timeframe: str, dataset: str) -> Path:
    return out_dir / f"{symbol}_{timeframe}_{dataset}.json"


def _as_utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _now() -> datetime:
    """Шов для тестов: окно retention считается от «сейчас»."""
    return datetime.now(UTC)


__all__ = [
    "SERIES_SPECS",
    "DatasetCheck",
    "PairValidation",
    "SeriesSpec",
    "describe_pair",
    "describe_total",
    "has_defects",
    "infer_step",
    "validate_data_from_config",
]
