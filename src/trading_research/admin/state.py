"""ADM-1 — read-model состояния платформы: один источник правды.

Узнать состояние платформы до этого модуля можно было только запустив пайплайн:
свежесть данных показывал ``update-data``, качество — ``validate-data``, состав
результатов — ``analyze-results``. То есть проверка состояния делалась **изменением
состояния**, а между запусками о нём не было известно ничего.

Четыре вещи, которых не было видно (замеры на 2026-11-03):

1. Свечи в store кончались ``2026-10-31 07:00`` — отставание трое суток, потому что
   по расписанию идут только рыночные ряды (OPS-1), а догрузка свечей (OPS-2)
   запускается руками.
2. 28 отчётов валидации говорили «дефектов нет» и не содержали времени проверки.
   Поле ``checked_at`` добавлено в DATA-3 этим же тикетом; отчёты, записанные до
   него, читаются как «время неизвестно», а не как свежие.
3. Из 684 прогонов ``ma_cross_multi`` у 312 стоит ``reliable=0``, и ``rank_params``
   (AN-2) усредняет их в рейтинге без фильтра и без пометки. Доля ненадёжных входит
   в состояние явно; фильтровать ли рейтинг — вопрос к AN-2, не к админке.
4. Запас до невосполнимой потери OI/LS считался в голове.

## Три решения, которые важнее механики обхода

**Читаем только через существующие store.** ``ParquetStore``, ``ResultsStore``,
``load_report``, ``report_path``, ``wf_dir``, ``experiment_store_root`` — своих SQL
и своих формул путей здесь нет. Проект шесть раз ловил расхождение «библиотека
есть, CLI-пути нет» (DATA-8, RUN-1…RUN-5, ``random_search`` в RES-2); админка со
своим чтением дала бы его в обратную сторону — «в панели одно число, в CLI другое»,
и разъехались бы они молча.

**Ничего не пишем.** Обычный ``ResultsStore(root)`` создаёт каталог и таблицы, то
есть взгляд на эксперимент, которого нет, создавал бы пустой эксперимент. Поэтому
хранилище открывается с ``create=False``.

**Часы приходят извне.** ``now`` передаётся сверху и протаскивается во все
вычисления: отставание и запас retention иначе не проверить тестом. Единственный
шов к системным часам — ``_now`` в ``collect_state``.

Состояние расписания выводится **из данных**, а не из лога: свежесть OI/LS
отвечает, жив ли сбор, и именно так это описано в README. Лог отвечает на другой
вопрос — «почему»: когда была последняя итерация и с каким кодом она кончилась.
``launchctl`` не опрашивается: это платформенная зависимость (macOS) и подпроцесс в
read-model ради сведений, которые уже есть в логе.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from trading_research.cli.fetch_market_data import PERIODIC_SERIES, SeriesKind
from trading_research.cli.run_experiment import experiment_store_root
from trading_research.cli.validate_data import SERIES_SPECS, report_path
from trading_research.cli.walk_forward import (
    HOLDOUT_REPORT_FILE,
    OOS_EQUITY_FILE,
    SELECTIONS_FILE,
    wf_dir,
)
from trading_research.config import ExperimentConfig, load_experiment_config
from trading_research.data.ingestion.market_data_client import PERIOD_MS, SAFE_RETENTION
from trading_research.data.schema import timeframe_to_ms
from trading_research.data.storage.parquet_store import CANDLES, ParquetStore
from trading_research.data.validators import (
    SeriesValidationReport,
    ValidationReport,
    load_report,
)
from trading_research.experiments.results_store import ResultsStore

_LOG_TAIL_BYTES = 65_536
_ITERATION_PREFIX = "=== "
_EXIT_PREFIX = "--- exit="
_LOG_TIME_FORMAT = "%Y-%m-%d %H:%M:%S UTC"


# --- данные -----------------------------------------------------------------


@dataclass(frozen=True)
class DatasetState:
    """Покрытие одного ряда по одной паре символ × таймфрейм.

    ``period`` — шаг ряда; у funding его задаёт биржа, поэтому там ``None`` и
    отставание считается только во времени. ``retention`` заполнен лишь у рядов с
    окном доступности (OI/LS): только у них отставание может стать невосполнимым.
    """

    dataset: str
    symbol: str
    timeframe: str
    rows: int
    first_point: datetime | None
    last_point: datetime | None
    period: timedelta | None
    retention: timedelta | None
    now: datetime

    @property
    def is_empty(self) -> bool:
        return self.rows == 0

    @property
    def lag(self) -> timedelta | None:
        """Отставание последней точки от ``now``; ``None`` — данных нет."""
        return None if self.last_point is None else self.now - self.last_point

    @property
    def lag_periods(self) -> int | None:
        """Отставание в периодах ряда; ``None`` — период задаёт биржа или данных нет."""
        lag, period = self.lag, self.period
        if lag is None or period is None:
            return None
        return int(lag / period)

    @property
    def reserve(self) -> timedelta | None:
        """Запас до невосполнимой потери (только ряды с retention).

        Пока отставание меньше окна доступности, пропуск ещё можно догрузить. Когда
        запас кончился, разрыв между последней точкой и доступным началом не закрыть
        уже ничем — это единственная ошибка в проекте, не исправимая задним числом.
        """
        lag, retention = self.lag, self.retention
        if lag is None or retention is None:
            return None
        return retention - lag

    @property
    def is_exhausted(self) -> bool:
        """Запас кончился: часть истории ряда потеряна навсегда."""
        reserve = self.reserve
        return reserve is not None and reserve <= timedelta(0)


def dataset_states(
    cfg: ExperimentConfig, *, data_root: str | Path, now: datetime
) -> list[DatasetState]:
    """Покрытие всех рядов по всем парам конфига.

    Набор пар и рядов берётся из конфига и ``SERIES_SPECS`` — того же места, откуда
    его берут загрузка (OPS-1, OPS-2) и валидация (DATA-8).
    """
    store = ParquetStore(data_root)
    states: list[DatasetState] = []
    for symbol in cfg.data.symbols:
        for timeframe in cfg.data.timeframes:
            frame = store.read_candles(
                exchange=cfg.data.exchange,
                market=cfg.data.market,
                symbol=symbol,
                timeframe=timeframe,
            )
            states.append(
                _dataset_state(
                    frame_bounds(frame),
                    dataset=CANDLES,
                    symbol=symbol,
                    timeframe=timeframe,
                    period=timedelta(milliseconds=timeframe_to_ms(timeframe)),
                    retention=None,
                    now=now,
                )
            )
            for spec in SERIES_SPECS:
                series = store.read_series(
                    spec.dataset,
                    exchange=cfg.data.exchange,
                    market=cfg.data.market,
                    symbol=symbol,
                    timeframe=timeframe,
                )
                states.append(
                    _dataset_state(
                        frame_bounds(series),
                        dataset=spec.dataset,
                        symbol=symbol,
                        timeframe=timeframe,
                        period=_series_period(spec.kind, timeframe),
                        retention=SAFE_RETENTION if spec.kind in PERIODIC_SERIES else None,
                        now=now,
                    )
                )
    return states


def frame_bounds(frame: pl.DataFrame) -> tuple[int, datetime | None, datetime | None]:
    """Число строк и границы по ``open_time``.

    Вынесено функцией, чтобы свечи и ряды мерились одним способом: ``read_candles``
    и ``read_series`` возвращают разные схемы, но время в обеих — ``open_time``.
    Ряда может не быть вовсе — тогда фрейм пуст и колонки в нём нет.
    """
    if frame.height == 0 or "open_time" not in frame.columns:
        return frame.height, None, None
    column = frame["open_time"]
    return frame.height, _as_utc_or_none(column.min()), _as_utc_or_none(column.max())


def _dataset_state(
    bounds: tuple[int, datetime | None, datetime | None],
    *,
    dataset: str,
    symbol: str,
    timeframe: str,
    period: timedelta | None,
    retention: timedelta | None,
    now: datetime,
) -> DatasetState:
    rows, first, last = bounds
    return DatasetState(
        dataset=dataset,
        symbol=symbol,
        timeframe=timeframe,
        rows=rows,
        first_point=first,
        last_point=last,
        period=period,
        retention=retention,
        now=now,
    )


def _series_period(kind: SeriesKind, timeframe: str) -> timedelta | None:
    """Шаг ряда: у OI/LS — период запроса, у funding его задаёт биржа."""
    if kind not in PERIODIC_SERIES:
        return None
    step = PERIOD_MS.get(timeframe)
    return None if step is None else timedelta(milliseconds=step)


# --- валидация --------------------------------------------------------------


@dataclass(frozen=True)
class ValidationState:
    """Последний отчёт валидации по одному датасету одной пары.

    ``checked_at is None`` при существующем отчёте означает «время проверки
    неизвестно» — так читаются отчёты, записанные до ADM-1. Это не то же самое, что
    «проверено сейчас», и показывать это надо по-разному.
    """

    dataset: str
    symbol: str
    timeframe: str
    path: Path
    exists: bool
    checked_at: datetime | None = None
    is_clean: bool | None = None
    rows: int = 0
    error: str | None = None
    """Отчёт есть, но прочитать его не удалось — это состояние, а не исключение."""

    @property
    def is_dated(self) -> bool:
        return self.checked_at is not None

    def age(self, now: datetime) -> timedelta | None:
        """Давность вердикта; ``None`` — отчёта нет или он без времени."""
        return None if self.checked_at is None else now - self.checked_at


def validation_states(cfg: ExperimentConfig, *, reports_root: str | Path) -> list[ValidationState]:
    """Отчёты валидации по всем парам и датасетам конфига.

    Пути берутся у ``report_path`` (DATA-8): где лежат отчёты, знает команда,
    которая их пишет.
    """
    out_dir = Path(reports_root) / cfg.name
    datasets = (CANDLES, *(spec.dataset for spec in SERIES_SPECS))
    states: list[ValidationState] = []
    for symbol in cfg.data.symbols:
        for timeframe in cfg.data.timeframes:
            for dataset in datasets:
                path = report_path(out_dir, symbol, timeframe, dataset)
                states.append(
                    _validation_state(path, dataset=dataset, symbol=symbol, timeframe=timeframe)
                )
    return states


def _validation_state(path: Path, *, dataset: str, symbol: str, timeframe: str) -> ValidationState:
    if not path.exists():
        return ValidationState(
            dataset=dataset, symbol=symbol, timeframe=timeframe, path=path, exists=False
        )
    try:
        report = load_report(path)
    except (ValueError, OSError) as exc:
        return ValidationState(
            dataset=dataset,
            symbol=symbol,
            timeframe=timeframe,
            path=path,
            exists=True,
            error=str(exc),
        )
    return ValidationState(
        dataset=dataset,
        symbol=symbol,
        timeframe=timeframe,
        path=path,
        exists=True,
        checked_at=report.checked_at,
        is_clean=report.is_clean,
        rows=_report_rows(report),
    )


def _report_rows(report: ValidationReport | SeriesValidationReport) -> int:
    if isinstance(report, ValidationReport):
        return report.total_bars
    return report.total_rows


# --- расписание -------------------------------------------------------------


@dataclass(frozen=True)
class ScheduleState:
    """Что известно о регулярном сборе рыночных рядов (OPS-1) из его лога.

    Вердикт «жив или нет» по этому состоянию не выносится: его дают сами данные —
    свежесть OI/LS в ``DatasetState``. Здесь только причина: когда была последняя
    итерация и чем она кончилась.
    """

    log_path: Path | None
    exists: bool
    now: datetime
    last_iteration_at: datetime | None = None
    last_exit_status: int | None = None
    """``None`` при известной итерации — она ещё не кончилась (строки ``exit`` нет)."""

    @property
    def silence(self) -> timedelta | None:
        """Сколько прошло с последней итерации; ``None`` — итераций в логе нет."""
        if self.last_iteration_at is None:
            return None
        return self.now - self.last_iteration_at

    @property
    def last_iteration_failed(self) -> bool:
        return self.last_exit_status is not None and self.last_exit_status != 0


def schedule_state(log_path: str | Path | None, *, now: datetime) -> ScheduleState:
    """Прочитать хвост лога сбора (OPS-1).

    Читается только хвост: лог append-only, а итераций по четыре в сутки, так что
    последняя всегда в последних килобайтах.
    """
    if log_path is None:
        return ScheduleState(log_path=None, exists=False, now=now)
    path = Path(log_path)
    if not path.exists():
        return ScheduleState(log_path=path, exists=False, now=now)

    iteration_at: datetime | None = None
    exit_status: int | None = None
    for line in _tail_lines(path):
        if line.startswith(_ITERATION_PREFIX):
            parsed = _parse_iteration_time(line)
            if parsed is not None:
                # Новая итерация: её код возврата ещё не прочитан.
                iteration_at, exit_status = parsed, None
        elif line.startswith(_EXIT_PREFIX):
            exit_status = _parse_exit_status(line)
    return ScheduleState(
        log_path=path,
        exists=True,
        now=now,
        last_iteration_at=iteration_at,
        last_exit_status=exit_status,
    )


def _tail_lines(path: Path, *, max_bytes: int = _LOG_TAIL_BYTES) -> list[str]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        chunk = handle.read()
    text = chunk.decode("utf-8", errors="replace")
    return text.splitlines()


def _parse_iteration_time(line: str) -> datetime | None:
    """``=== 2026-11-02 18:40:01 UTC — <конфиг>`` → момент итерации."""
    body = line[len(_ITERATION_PREFIX) :]
    stamp = body.split(" — ")[0].strip()
    try:
        return datetime.strptime(stamp, _LOG_TIME_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None


def _parse_exit_status(line: str) -> int | None:
    try:
        return int(line[len(_EXIT_PREFIX) :].strip())
    except ValueError:
        return None


# --- результаты -------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentState:
    """Состав прогонов одного эксперимента.

    ``unreliable`` — прогоны, чьи trade-зависимые метрики статистически ненадёжны
    (``trades_count < min_trades``, BT-7). Доля входит в состояние явно: сегодня
    такие прогоны попадают в рейтинг AN-2 неотличимо от остальных.
    """

    name: str
    runs: int
    by_status: Mapping[str, int]
    strategies: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    slices: int
    param_sets: int
    unreliable: int
    without_metrics: int
    git_commits: tuple[str, ...]
    dataset_hashes: tuple[str, ...]
    first_run_at: datetime | None
    last_run_at: datetime | None

    @property
    def unreliable_share(self) -> float:
        """Доля ненадёжных от **всех** прогонов, а не от удачных."""
        return self.unreliable / self.runs if self.runs else 0.0

    @property
    def is_empty(self) -> bool:
        return self.runs == 0


def experiment_names(results_root: str | Path) -> list[str]:
    """Эксперименты, у которых есть каталог результатов.

    Признак — то, что хранилище открывается на чтение, а не имя папки и не имя файла
    каталога: как называется база, знает ``ResultsStore``, и знать это в двух местах
    незачем. Пустая папка экспериментом не является.
    """
    root = Path(results_root)
    if not root.is_dir():
        return []
    return [child.name for child in sorted(root.iterdir()) if _has_results(root, child.name)]


def _has_results(results_root: Path, name: str) -> bool:
    try:
        ResultsStore(experiment_store_root(results_root, name), create=False)
    except FileNotFoundError:
        return False
    return True


def experiment_state(results_root: str | Path, name: str) -> ExperimentState | None:
    """Состав прогонов эксперимента; ``None`` — каталога результатов нет.

    Хранилище открывается только для чтения (``create=False``): иначе взгляд на
    эксперимент, которого нет, создал бы пустой эксперимент.
    """
    try:
        store = ResultsStore(experiment_store_root(results_root, name), create=False)
    except FileNotFoundError:
        return None

    runs = store.all_runs()
    by_status: dict[str, int] = {}
    for record in runs:
        by_status[record.status.value] = by_status.get(record.status.value, 0) + 1

    times = [t for record in runs for t in (record.created_at,) if t is not None]
    updates = [t for record in runs for t in (record.updated_at,) if t is not None]
    return ExperimentState(
        name=name,
        runs=len(runs),
        by_status=by_status,
        strategies=tuple(sorted({r.strategy for r in runs})),
        pairs=tuple(sorted({(r.symbol, r.timeframe) for r in runs})),
        slices=len({r.slice_id for r in runs}),
        param_sets=len({json.dumps(r.params, sort_keys=True) for r in runs}),
        unreliable=sum(1 for r in runs if r.metrics is not None and not r.metrics.reliable),
        without_metrics=sum(1 for r in runs if r.metrics is None),
        git_commits=tuple(sorted({r.git_commit for r in runs if r.git_commit})),
        dataset_hashes=tuple(sorted({r.dataset_hash for r in runs})),
        first_run_at=min(times, default=None),
        last_run_at=max(updates or times, default=None),
    )


# --- walk-forward -----------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardState:
    """Артефакты walk-forward одной пары.

    ``has_holdout`` — только факт существования отчёта. Значение holdout здесь не
    читается намеренно: резерв WF-3 ценен ровно тем, сколько раз в него смотрели, и
    раскрытие — отдельное действие с журналом (ADM-5).
    """

    symbol: str
    timeframe: str
    directory: Path
    exists: bool
    folds: int = 0
    has_oos_equity: bool = False
    has_holdout: bool = False
    error: str | None = None


def walk_forward_states(
    cfg: ExperimentConfig, *, results_root: str | Path
) -> list[WalkForwardState]:
    """Артефакты walk-forward по всем парам конфига (пути — из ``wf_dir``)."""
    states: list[WalkForwardState] = []
    for symbol in cfg.data.symbols:
        for timeframe in cfg.data.timeframes:
            directory = wf_dir(results_root, cfg.name, symbol, timeframe)
            states.append(_walk_forward_state(directory, symbol=symbol, timeframe=timeframe))
    return states


def _walk_forward_state(directory: Path, *, symbol: str, timeframe: str) -> WalkForwardState:
    if not directory.is_dir():
        return WalkForwardState(
            symbol=symbol, timeframe=timeframe, directory=directory, exists=False
        )
    folds, error = _count_folds(directory / SELECTIONS_FILE)
    return WalkForwardState(
        symbol=symbol,
        timeframe=timeframe,
        directory=directory,
        exists=True,
        folds=folds,
        has_oos_equity=(directory / OOS_EQUITY_FILE).exists(),
        has_holdout=(directory / HOLDOUT_REPORT_FILE).exists(),
        error=error,
    )


def _count_folds(path: Path) -> tuple[int, str | None]:
    if not path.exists():
        return 0, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return 0, str(exc)
    if not isinstance(payload, dict):
        return 0, f"{path}: ожидался объект с полем 'selections'"
    selections = payload.get("selections")
    return (len(selections) if isinstance(selections, list) else 0), None


# --- всё вместе -------------------------------------------------------------


@dataclass(frozen=True)
class PlatformState:
    """Состояние платформы по одному конфигу эксперимента."""

    experiment: str
    config_path: Path
    now: datetime
    datasets: tuple[DatasetState, ...]
    validations: tuple[ValidationState, ...]
    schedule: ScheduleState
    results: ExperimentState | None
    walk_forward: tuple[WalkForwardState, ...]

    @property
    def candles(self) -> tuple[DatasetState, ...]:
        return tuple(state for state in self.datasets if state.dataset == CANDLES)

    @property
    def perishable(self) -> tuple[DatasetState, ...]:
        """Ряды с окном retention: только их отставание бывает невосполнимым."""
        return tuple(state for state in self.datasets if state.retention is not None)

    @property
    def defects(self) -> tuple[ValidationState, ...]:
        return tuple(state for state in self.validations if state.is_clean is False)


def collect_state(
    config_path: str | Path,
    *,
    data_root: str | Path,
    results_root: str | Path,
    reports_root: str | Path,
    log_path: str | Path | None = None,
    now: datetime | None = None,
) -> PlatformState:
    """Собрать состояние платформы по конфигу эксперимента.

    Ничего не запускает, в сеть не ходит и ничего не пишет: все четыре контура
    (данные, расписание, валидация, результаты) читаются с диска через
    существующие store.

    ``now`` — единственный шов к системным часам; дальше он протаскивается во все
    вычисления отставания и запаса, чтобы состояние можно было проверить тестом.
    """
    cfg = load_experiment_config(config_path)
    moment = _as_utc(now or _now())
    return PlatformState(
        experiment=cfg.name,
        config_path=Path(config_path),
        now=moment,
        datasets=tuple(dataset_states(cfg, data_root=data_root, now=moment)),
        validations=tuple(validation_states(cfg, reports_root=reports_root)),
        schedule=schedule_state(log_path, now=moment),
        results=experiment_state(results_root, cfg.name),
        walk_forward=tuple(walk_forward_states(cfg, results_root=results_root)),
    )


def _as_utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _as_utc_or_none(value: object) -> datetime | None:
    return _as_utc(value) if isinstance(value, datetime) else None


def _now() -> datetime:
    """Шов для тестов — тот же приём, что в OPS-1 и OPS-2."""
    return datetime.now(UTC)


__all__ = [
    "DatasetState",
    "ExperimentState",
    "PlatformState",
    "ScheduleState",
    "ValidationState",
    "WalkForwardState",
    "collect_state",
    "dataset_states",
    "experiment_names",
    "experiment_state",
    "frame_bounds",
    "schedule_state",
    "validation_states",
    "walk_forward_states",
]
