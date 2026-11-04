"""ADM-4 — read-model прогонов: откуда взялось число.

[ADM-1](../../../tickets/ADM-1.md) отвечает «что есть в эксперименте» сводкой:
прогонов столько, слайсов столько, ненадёжных столько. Этот модуль отвечает на
следующий вопрос — «из чего посчитано вот это число» — и доводит путь до
конкретного прогона с его параметрами, сделками и происхождением.

## Что показал store, когда путь появился

Рейтинг по ``sharpe`` на 684 прогонах ``ma_cross_multi`` возглавляет
``fast_ma=50, slow_ma=200`` со средним 0,47. Это среднее по **114** прогонам, и
**61 из них** имеет ``reliable=0`` — больше половины. У строки на четвёртом месте
ненадёжных 36. То есть порядок строк в рейтинге частично определяется тем, какая
комбинация набрала больше прогонов со двумя-тремя сделками.

Отдельный прогон этого эксперимента выглядит так: ``sharpe`` 2,35, доходность
18 % — **на двух сделках**. Число красивое, и до этой страницы в нём не было
видно ничего, что сказало бы: смотреть на него нельзя.

Админка это **не чинит**. Фильтровать ли рейтинг по надёжности — вопрос к AN-2 и
решается отдельно: решение «усреднять ли ненадёжные» слишком важное, чтобы
протащить его мимо тикета. Дело страницы — показать долю рядом с каждой строкой,
чтобы решение принималось зная.

## Три решения

**Считают существующие analytics.** Рейтинг — ``rank_params`` (AN-2),
стабильность — ``stability_vector``/``rank_by_stability`` (AN-3), матрица
heatmap — ``build_heatmap`` (AN-4), таблица — ``load_performance_table`` по пути
``experiment_results_path``, то есть ровно то, что читает ``analyze-results``.
Здесь нет ни одной своей формулы метрики: вторая реализация разъехалась бы с CLI
молча, и проект ловил это шесть раз («библиотека есть, CLI-пути нет»).

**Доля ненадёжных считается рядом с рейтингом, а не внутри него.**
``rank_params`` отдаёт ``n`` — сколько строк усреднено, но не отдаёт, сколько из
них ненадёжны, и дописывать это в AN-2 нельзя: функция принадлежит аналитике, и
её вывод уже экспортируется (``export-results``). Поэтому счёт берётся из той же
таблицы, из которой считался рейтинг, и приклеивается к строке по ``param_hash``.

**Идентичность прогона проверяется, а не печатается.** Принцип №5 обещает
воспроизводимость, и в каталоге для неё всё есть: ``run_hash``, ``dataset_hash``,
``git_commit``, слайс. Но проверить обещание можно только пересчётом: ``RunSpec``
из того, что показано на странице, обязан дать тот же ``run_hash``. Если не даёт
— конфиг разошёлся с прогоном, и это главное, что нужно сказать о числе. На
сегодняшнем store сходится 684 из 684; ``git_commit`` прогонов (``e847b0e``) при
этом **не** равен HEAD, и страница говорит об этом отдельно.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from trading_research.analytics.plots import build_heatmap
from trading_research.analytics.queries import (
    QUERYABLE_METRICS,
    load_performance_table,
    param_hash,
    rank_params,
)
from trading_research.analytics.stability import rank_by_stability, stability_vector
from trading_research.backtest.metrics import Metrics
from trading_research.cli.analyze_results import experiment_results_path
from trading_research.cli.run_experiment import experiment_store_root
from trading_research.config import ExperimentConfig
from trading_research.data.catalog import dataset_hash
from trading_research.data.storage.duckdb_store import DuckDBStore
from trading_research.experiments.cache import RunSpec
from trading_research.experiments.results_store import ResultsStore, RunRecord, RunStatus

DEFAULT_METRIC = "total_return_pct"
DEFAULT_TOP = 20
DEFAULT_AGG = "median"

# Сколько точек equity доносится до страницы. Кривая рисуется inline-SVG, и
# 552 точки одного прогона в разметке — это лишние десятки килобайт на глаз,
# который всё равно видит форму, а не отдельный бар.
EQUITY_POINTS = 240

# Сколько сделок показывается таблицей. Прогон с тысячей сделок интересен формой
# кривой и метриками, а не тысячей строк; остаток называется числом.
TRADES_SHOWN = 200

AGGS: tuple[str, ...] = ("median", "mean", "min", "max")

# С какой доли ненадёжных прогонов выборка перестаёт быть надёжной выборкой. Порог
# не из статистики, а из смысла: когда каждый четвёртый прогон в среднем держится
# на двух сделках, среднее уже говорит о сетке больше, чем о стратегии.
UNRELIABLE_ALARM = 0.25


# --- рейтинг (AN-2) ---------------------------------------------------------


@dataclass(frozen=True)
class RankingRow:
    """Строка рейтинга: чем она является и из чего посчитана.

    ``runs`` — сколько прогонов усреднено в ``value``, ``unreliable`` — сколько из
    них ненадёжны (``trades_count < min_trades``, BT-7). Два числа стоят рядом со
    значением намеренно: без них строка рейтинга не отличима от строки, собранной
    из прогонов по две сделки.
    """

    rank: int
    param_hash: str
    params: dict[str, Any]
    metric: str
    value: float | None
    runs: int
    unreliable: int
    axis: str | None = None
    axis_value: str | None = None

    @property
    def unreliable_share(self) -> float:
        return self.unreliable / self.runs if self.runs else 0.0

    @property
    def is_mostly_unreliable(self) -> bool:
        """Больше половины усреднённых прогонов ненадёжны."""
        return self.runs > 0 and self.unreliable * 2 > self.runs


@dataclass(frozen=True)
class Ranking:
    """Рейтинг и то, на какой выборке он посчитан."""

    experiment: str
    metric: str
    rows: tuple[RankingRow, ...]
    selected_runs: int
    selected_unreliable: int
    total_runs: int
    total_unreliable: int
    symbol: str | None = None
    timeframe: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.rows

    @property
    def selected_share(self) -> float:
        return self.selected_unreliable / self.selected_runs if self.selected_runs else 0.0

    @property
    def total_share(self) -> float:
        return self.total_unreliable / self.total_runs if self.total_runs else 0.0

    @property
    def is_filtered(self) -> bool:
        return self.symbol is not None or self.timeframe is not None

    @property
    def is_concerning(self) -> bool:
        """Ненадёжных в выборке столько, что рейтинг нельзя читать как есть."""
        return self.selected_share >= UNRELIABLE_ALARM


# --- стабильность (AN-3) ----------------------------------------------------


@dataclass(frozen=True)
class StabilityRow:
    """Строка вектора устойчивости: робастная альтернатива одной средней метрике.

    ``table_rows`` — сколько строк таблицы вошло в агрегат. Это **не** число
    слайсов, пока в выборке больше одной пары: на сегодняшнем store у каждой
    комбинации 114 строк — 19 слайсов на шести парах.
    """

    rank: int
    param_hash: str
    params: dict[str, Any]
    table_rows: int
    median_return: float | None
    positive_slices_ratio: float | None
    worst_slice: float | None
    iqr: float | None
    return_std: float | None
    stability_score: float | None


@dataclass(frozen=True)
class Stability:
    """Вектор устойчивости и то, на какой выборке он посчитан.

    AN-3 требует фильтровать таблицу до одной пары: ``top_quantile_freq``
    сравнивает комбинации **внутри слайса**, и при шести парах в одном слайсе
    оказываются шесть разных рынков. Требование было записано в докстринге
    ``stability_vector`` и ничем не проверялось — страница говорит о нём вслух.
    """

    metric: str
    rows: tuple[StabilityRow, ...]
    pairs: int

    @property
    def mixes_pairs(self) -> bool:
        return self.pairs > 1

    @property
    def is_empty(self) -> bool:
        return not self.rows


# --- heatmap (AN-4) ---------------------------------------------------------


@dataclass(frozen=True)
class HeatmapCell:
    """Ячейка карты параметров: значение и его место между минимумом и максимумом."""

    x: Any
    y: Any
    value: float | None
    shade: float | None

    @property
    def is_empty(self) -> bool:
        return self.value is None


@dataclass(frozen=True)
class Heatmap:
    """Карта параметров ``x × y`` по агрегату метрики."""

    x_param: str
    y_param: str
    metric: str
    agg: str
    xs: tuple[Any, ...]
    ys: tuple[Any, ...]
    rows: tuple[tuple[HeatmapCell, ...], ...]
    low: float | None
    high: float | None

    @property
    def is_empty(self) -> bool:
        return self.low is None

    @property
    def labelled_rows(self) -> tuple[tuple[Any, tuple[HeatmapCell, ...]], ...]:
        """Строки вместе с подписью оси Y — чтобы шаблон не склеивал два списка."""
        return tuple(zip(self.ys, self.rows, strict=True))


# --- прогоны ----------------------------------------------------------------


@dataclass(frozen=True)
class RunRow:
    """Один прогон в списке: идентичность, статус и метрики, если они есть."""

    run_hash: str
    symbol: str
    timeframe: str
    slice_id: str
    status: str
    params: dict[str, Any]
    param_hash: str
    metrics: Metrics | None
    error: str | None
    has_trades: bool
    has_equity: bool

    @property
    def failed(self) -> bool:
        return self.status == RunStatus.FAILED.value

    @property
    def reliable(self) -> bool | None:
        """``None`` — метрик нет, то есть надёжность не определена, а не «ложь»."""
        return None if self.metrics is None else self.metrics.reliable

    @property
    def trades_count(self) -> int | None:
        return None if self.metrics is None else self.metrics.trades_count

    def value(self, metric: str) -> float | None:
        """Значение метрики прогона; ``None`` — метрик нет."""
        if self.metrics is None:
            return None
        raw = getattr(self.metrics, metric, None)
        return None if raw is None else float(raw)


@dataclass(frozen=True)
class Provenance:
    """Происхождение числа: чем прогон задан и воспроизводится ли он.

    ``recomputed`` — ``run_hash``, пересчитанный из показанной идентичности. Если
    он равен сохранённому, страница показывает полный набор входов; если нет —
    конфиг разошёлся с прогоном, и об этом нужно сказать раньше любых метрик.
    """

    run_hash: str
    dataset_hash: str
    git_commit: str | None
    head_commit: str | None
    recomputed: str | None
    command: tuple[str, ...]

    @property
    def identity_matches(self) -> bool | None:
        """``None`` — пересчитать не из чего (конфиг эксперимента не открыт)."""
        return None if self.recomputed is None else self.recomputed == self.run_hash

    @property
    def code_changed(self) -> bool:
        """Код изменился с момента прогона: воспроизведение требует ``git checkout``."""
        return (
            self.git_commit is not None
            and self.head_commit is not None
            and self.git_commit != self.head_commit
        )


@dataclass(frozen=True)
class DataCheck:
    """Те ли сегодня данные, по которым прогон считался.

    ``dataset_hash`` — контентный хэш свечей окна (DATA-4), и он входит в
    ``run_hash``. Значит ответ «да» делает прогон воспроизводимым, а ответ «нет»
    означает, что данные под числом изменились: сырьё обещано неизменным, и
    расхождение здесь — повод разбираться, а не мелочь.

    Проверка стоит около 0,4 с на паре и поэтому делается по требованию, а не на
    каждый показ страницы.
    """

    stored: str
    today: str | None
    rows: int
    error: str | None = None

    @property
    def matches(self) -> bool | None:
        """``None`` — проверить не удалось (данных нет или окно пусто)."""
        return None if self.today is None else self.today == self.stored


@dataclass(frozen=True)
class EquityCurve:
    """Кривая капитала прогона, прореженная до размера, уместного в разметке."""

    points: tuple[tuple[datetime, float], ...]
    total_points: int
    low: float
    high: float

    @property
    def is_empty(self) -> bool:
        return not self.points

    def shape(self) -> tuple[tuple[float, float], ...]:
        """Точки в долях ``(0..1, 0..1)``: готовый материал для inline-SVG."""
        if len(self.points) < 2:
            return ()
        span = self.high - self.low
        last = len(self.points) - 1
        return tuple(
            (index / last, 0.5 if span == 0 else (value - self.low) / span)
            for index, (_, value) in enumerate(self.points)
        )


@dataclass(frozen=True)
class RunDetail:
    """Прогон целиком: что это, из чего и как повторить."""

    row: RunRow
    provenance: Provenance
    equity: EquityCurve
    trades: pl.DataFrame | None
    trades_total: int

    @property
    def trades_hidden(self) -> int:
        shown = 0 if self.trades is None else self.trades.height
        return max(0, self.trades_total - shown)


# --- сборка -----------------------------------------------------------------


def performance_table(results_root: str | Path, experiment: str) -> pl.DataFrame | None:
    """Каноническая таблица AN-1 эксперимента; ``None`` — её ещё не писали.

    Путь и чтение — ``experiment_results_path`` и ``load_performance_table``, то
    есть ровно то, чем таблицу читает ``analyze-results``.
    """
    path = experiment_results_path(results_root, experiment)
    if not path.exists():
        return None
    return load_performance_table(path)


def open_store(results_root: str | Path, experiment: str) -> ResultsStore | None:
    """Каталог прогонов только для чтения; ``None`` — эксперимента нет."""
    try:
        return ResultsStore(experiment_store_root(results_root, experiment), create=False)
    except FileNotFoundError:
        return None


def param_names(table: pl.DataFrame) -> tuple[str, ...]:
    """Имена параметров сетки — из самой таблицы, а не из конфига.

    Конфиг мог измениться после прогона; оси карты должны соответствовать тому,
    что действительно посчитано.
    """
    names: set[str] = set()
    for raw in table.get_column("params_json").unique().to_list():
        if raw:
            names.update(json.loads(raw))
    return tuple(sorted(names))


def metric_names() -> tuple[str, ...]:
    """Метрики, по которым можно ранжировать и агрегировать (whitelist AN-1)."""
    return tuple(sorted(QUERYABLE_METRICS))


def ranking(
    table: pl.DataFrame,
    *,
    experiment: str,
    metric: str = DEFAULT_METRIC,
    top: int = DEFAULT_TOP,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> Ranking:
    """Рейтинг AN-2 плюс доля ненадёжных прогонов в каждой строке."""
    selected = _filtered(table, symbol=symbol, timeframe=timeframe)
    counts = _reliability_by_param(selected)
    rows: tuple[RankingRow, ...] = ()
    if selected.height:
        ranked = rank_params(
            selected, metric=metric, top=top, symbol=None, timeframe=None, descending=True
        )
        rows = tuple(
            RankingRow(
                rank=int(row["rank"]),
                param_hash=str(row["param_hash"]),
                params=_params(row["params_json"]),
                metric=metric,
                value=_optional_float(row[metric]),
                runs=int(row["n"]),
                unreliable=counts.get(str(row["param_hash"]), 0),
            )
            for row in ranked.iter_rows(named=True)
        )
    return Ranking(
        experiment=experiment,
        metric=metric,
        rows=rows,
        selected_runs=selected.height,
        selected_unreliable=sum(counts.values()),
        total_runs=table.height,
        total_unreliable=sum(_reliability_by_param(table).values()),
        symbol=symbol,
        timeframe=timeframe,
    )


def stability(
    table: pl.DataFrame,
    *,
    metric: str = DEFAULT_METRIC,
    top: int = DEFAULT_TOP,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> Stability:
    """Вектор устойчивости AN-3, отранжированный по композиту."""
    selected = _filtered(table, symbol=symbol, timeframe=timeframe)
    if not selected.height:
        return Stability(metric=metric, rows=(), pairs=0)
    ranked = rank_by_stability(stability_vector(selected, return_metric=metric), top=top)
    rows = tuple(
        StabilityRow(
            rank=int(row["rank"]),
            param_hash=str(row["param_hash"]),
            params=_params(row["params_json"]),
            table_rows=int(row["n_slices"]),
            median_return=_optional_float(row["median_return"]),
            positive_slices_ratio=_optional_float(row["positive_slices_ratio"]),
            worst_slice=_optional_float(row["worst_slice"]),
            iqr=_optional_float(row["iqr"]),
            return_std=_optional_float(row["return_std"]),
            stability_score=_optional_float(row["stability_score"]),
        )
        for row in ranked.iter_rows(named=True)
    )
    return Stability(metric=metric, rows=rows, pairs=_pairs(selected))


def heatmap(
    table: pl.DataFrame,
    *,
    x: str,
    y: str,
    metric: str = DEFAULT_METRIC,
    agg: str = DEFAULT_AGG,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> Heatmap:
    """Карта параметров: матрица из ``build_heatmap`` плюс место каждой ячейки."""
    data = build_heatmap(
        table, x=x, y=y, metric=metric, agg=agg, symbol=symbol, timeframe=timeframe
    )
    values = [value for row in data.values for value in row if value is not None]
    low = min(values) if values else None
    high = max(values) if values else None
    rows = tuple(
        tuple(
            HeatmapCell(x=xv, y=yv, value=value, shade=_shade(value, low, high))
            for xv, value in zip(data.xs, row, strict=True)
        )
        for yv, row in zip(data.ys, data.values, strict=True)
    )
    return Heatmap(
        x_param=data.x_param,
        y_param=data.y_param,
        metric=data.metric,
        agg=data.agg,
        xs=tuple(data.xs),
        ys=tuple(data.ys),
        rows=rows,
        low=low,
        high=high,
    )


def run_rows(
    store: ResultsStore,
    *,
    param_hash_value: str | None = None,
    params: dict[str, Any] | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    slice_id: str | None = None,
    reliable: bool | None = None,
    status: str | None = None,
) -> tuple[RunRow, ...]:
    """Прогоны каталога под фильтрами — то самое «из чего посчитано число».

    ``params`` отбирает по подмножеству значений (ячейка карты задаёт две оси из
    сетки), ``param_hash_value`` — по точному набору (строка рейтинга).
    """
    rows = [_run_row(record) for record in store.all_runs()]
    if param_hash_value is not None:
        rows = [row for row in rows if row.param_hash == param_hash_value]
    if params:
        rows = [row for row in rows if _matches(row.params, params)]
    if symbol is not None:
        rows = [row for row in rows if row.symbol == symbol]
    if timeframe is not None:
        rows = [row for row in rows if row.timeframe == timeframe]
    if slice_id is not None:
        rows = [row for row in rows if row.slice_id == slice_id]
    if reliable is not None:
        rows = [row for row in rows if row.reliable is reliable]
    if status is not None:
        rows = [row for row in rows if row.status == status]
    return tuple(sorted(rows, key=lambda row: (row.symbol, row.timeframe, row.slice_id)))


def run_detail(
    store: ResultsStore,
    run_hash: str,
    *,
    config_path: str | Path,
    data_root: str | Path,
    results_root: str | Path,
    config: ExperimentConfig | None = None,
    head_commit: str | None = None,
) -> RunDetail | None:
    """Прогон целиком; ``None`` — такого ``run_hash`` в каталоге нет.

    ``config`` нужен только для пересчёта идентичности: модель исполнения в
    каталог не пишется, она живёт в конфиге. Без него страница показывает всё
    остальное и честно говорит, что проверить происхождение не из чего.
    """
    record = store.get_run(run_hash)
    if record is None:
        return None

    row = _run_row(record)
    equity = _equity(store, record)
    trades, total = _trades(store, record)
    return RunDetail(
        row=row,
        provenance=_provenance(
            record,
            config=config,
            config_path=config_path,
            data_root=data_root,
            results_root=results_root,
            head_commit=head_commit,
        ),
        equity=equity,
        trades=trades,
        trades_total=total,
    )


def check_data(
    config: ExperimentConfig,
    record_dataset_hash: str,
    *,
    symbol: str,
    timeframe: str,
    data_root: str | Path,
) -> DataCheck:
    """Пересчитать ``dataset_hash`` окна по сегодняшнему store и сравнить.

    Окно берётся из конфига теми же границами, какими его берёт
    ``run-experiment``: иначе сравнивались бы разные множества свечей.
    """
    try:
        candles = DuckDBStore(data_root).read_window(
            exchange=config.data.exchange,
            market=config.data.market,
            symbol=symbol,
            timeframe=timeframe,
            start=config.data.start,
            end=config.data.end,
        )
    except Exception as exc:  # noqa: BLE001 — любая причина здесь равно «проверить не вышло»
        return DataCheck(stored=record_dataset_hash, today=None, rows=0, error=str(exc))
    if candles.is_empty():
        return DataCheck(
            stored=record_dataset_hash, today=None, rows=0, error="в окне конфига нет свечей"
        )
    return DataCheck(stored=record_dataset_hash, today=dataset_hash(candles), rows=candles.height)


def reproduce_command(
    *,
    config_path: str | Path,
    data_root: str | Path,
    results_root: str | Path,
    checkout: str | None = None,
) -> tuple[str, ...]:
    """Команда, которой прогон повторяется — как строки, готовые к копированию.

    ``checkout`` добавляет возврат к коду прогона: ``run_hash`` включает
    ``git_commit``, поэтому на другом коммите получится другой ран, а не этот же.
    """
    lines: list[str] = []
    if checkout is not None:
        lines.append(f"git checkout {checkout}")
    lines.append(
        f"trading-research run-experiment --config {config_path} "
        f"--data-root {data_root} --results-root {results_root}"
    )
    return tuple(lines)


# --- внутреннее -------------------------------------------------------------


def _filtered(table: pl.DataFrame, *, symbol: str | None, timeframe: str | None) -> pl.DataFrame:
    if symbol is not None:
        table = table.filter(pl.col("symbol") == symbol)
    if timeframe is not None:
        table = table.filter(pl.col("timeframe") == timeframe)
    return table


def _pairs(table: pl.DataFrame) -> int:
    """Сколько пар символ × таймфрейм в выборке."""
    return table.select("symbol", "timeframe").unique().height


def _reliability_by_param(table: pl.DataFrame) -> dict[str, int]:
    """Сколько ненадёжных прогонов у каждого набора параметров."""
    if not table.height:
        return {}
    counted = table.group_by("param_hash").agg(
        (~pl.col("reliable").fill_null(False)).sum().alias("unreliable")
    )
    return {str(row["param_hash"]): int(row["unreliable"]) for row in counted.iter_rows(named=True)}


def _params(raw: object) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw:
        return {}
    parsed = json.loads(raw)
    return dict(parsed) if isinstance(parsed, dict) else {}


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _shade(value: float | None, low: float | None, high: float | None) -> float | None:
    """Место значения между минимумом и максимумом карты, 0..1."""
    if value is None or low is None or high is None:
        return None
    if high == low:
        return 0.5
    return (value - low) / (high - low)


def _matches(params: dict[str, Any], wanted: dict[str, Any]) -> bool:
    """Совпадает ли набор параметров с подмножеством значений (сравнение по тексту).

    Значения приходят из query-строки, то есть строками; сравнивать их с числами
    параметров приходится в одном виде, и текст — тот вид, в котором они и
    показаны на странице.
    """
    for key, value in wanted.items():
        if key not in params or str(params[key]) != str(value):
            return False
    return True


def _run_row(record: RunRecord) -> RunRow:
    params = dict(record.params)
    return RunRow(
        run_hash=record.run_hash,
        symbol=record.symbol,
        timeframe=record.timeframe,
        slice_id=record.slice_id,
        status=record.status.value,
        params=params,
        param_hash=param_hash(params),
        metrics=record.metrics,
        error=record.error,
        has_trades=record.trades_path is not None,
        has_equity=record.equity_path is not None,
    )


def _provenance(
    record: RunRecord,
    *,
    config: ExperimentConfig | None,
    config_path: str | Path,
    data_root: str | Path,
    results_root: str | Path,
    head_commit: str | None,
) -> Provenance:
    recomputed: str | None = None
    if config is not None:
        recomputed = RunSpec(
            strategy=record.strategy,
            params=dict(record.params),
            symbol=record.symbol,
            timeframe=record.timeframe,
            slice_id=record.slice_id,
            dataset_hash=record.dataset_hash,
            execution=config.execution,
            git_commit=record.git_commit,
        ).run_hash()
    checkout = (
        record.git_commit
        if record.git_commit is not None
        and head_commit is not None
        and record.git_commit != head_commit
        else None
    )
    return Provenance(
        run_hash=record.run_hash,
        dataset_hash=record.dataset_hash,
        git_commit=record.git_commit,
        head_commit=head_commit,
        recomputed=recomputed,
        command=reproduce_command(
            config_path=config_path,
            data_root=data_root,
            results_root=results_root,
            checkout=checkout,
        ),
    )


def _equity(store: ResultsStore, record: RunRecord) -> EquityCurve:
    if record.equity_path is None:
        return EquityCurve(points=(), total_points=0, low=0.0, high=0.0)
    frame = store.load_equity(record.run_hash)
    if not frame.height:
        return EquityCurve(points=(), total_points=0, low=0.0, high=0.0)
    time_col, value_col = frame.columns[0], frame.columns[1]
    # Шаг округляется вверх: при делении вниз 600 точек на лимите 240 дают 300,
    # то есть лимит перестаёт быть лимитом.
    step = max(1, -(-frame.height // EQUITY_POINTS))
    thinned = frame.gather_every(step) if step > 1 else frame
    if thinned.height and thinned.row(-1) != frame.row(-1):
        thinned = pl.concat([thinned, frame.tail(1)])
    values = [float(v) for v in thinned.get_column(value_col).to_list()]
    return EquityCurve(
        points=tuple(zip(thinned.get_column(time_col).to_list(), values, strict=True)),
        total_points=frame.height,
        low=min(values),
        high=max(values),
    )


def _trades(store: ResultsStore, record: RunRecord) -> tuple[pl.DataFrame | None, int]:
    if record.trades_path is None:
        return None, 0
    frame = store.load_trades(record.run_hash)
    return frame.head(TRADES_SHOWN), frame.height


__all__ = [
    "AGGS",
    "DEFAULT_AGG",
    "DEFAULT_METRIC",
    "DEFAULT_TOP",
    "EQUITY_POINTS",
    "TRADES_SHOWN",
    "UNRELIABLE_ALARM",
    "DataCheck",
    "EquityCurve",
    "Heatmap",
    "HeatmapCell",
    "Provenance",
    "Ranking",
    "RankingRow",
    "RunDetail",
    "RunRow",
    "Stability",
    "StabilityRow",
    "check_data",
    "heatmap",
    "metric_names",
    "open_store",
    "param_names",
    "performance_table",
    "ranking",
    "reproduce_command",
    "run_detail",
    "run_rows",
    "stability",
]
