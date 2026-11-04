"""ADM-3 + ADM-4 — веб-админка, только для чтения.

Состояние даёт read-model ([ADM-1](../../../tickets/ADM-1.md)), пороги и
формулировки — команда ``status`` ([ADM-2](../../../tickets/ADM-2.md)), прогоны и
их происхождение — ``admin/runs.py`` ([ADM-4](../../../tickets/ADM-4.md)). Здесь
только страницы и ни одного обработчика, который что-нибудь меняет: запуск шагов
пайплайна — отдельный тикет (ADM-6), и до него админка физически не может ничего
испортить.

## Почему веб, а при чём тут CLI

Команда ``status`` отвечает «всё ли живо», но углубиться в ней нельзя: 684
прогона, 19 слайсов и шесть пар в строках терминала не разглядеть, а карту
параметров в нём не нарисовать вовсе. Ноутбук эту задачу тоже не решает: он
требует ядра и исполнения, и показывает состояние на момент запуска, а не на
сейчас.

Выбран FastAPI с серверным HTML (Jinja) — без фронтенд-фреймворка. Причина не в
моде: страницы отдаются готовыми, и их можно проверить ``TestClient``'ом как
обычные функции, под ``mypy strict``. Streamlit отвергнут именно по этому: вся
логика живёт в скрипте, который перезапускается на каждый клик, и 644 теста
такой код не удержат.

## Четыре решения

**Одна правда на две поверхности.** Страницы не считают ничего своего: блоки
«Расписание», «Валидация» и «Эксперименты» — это те же строки, что печатает
``status``, рейтинг и карта — те же ``analytics.*``, что у ``analyze-results`` и
``plot-heatmap``, а ``/api/status`` отдаёт тот же payload, что ``status --json``.
Расхождение числа в CLI и на странице было бы хуже отсутствия страницы.

**Состояние читается на каждый запрос.** Кэша нет: сборка состояния на реальном
store — 0,19 с, рейтинг на 684 прогонах — 48 мс, и страница, показывающая
вчерашнюю свежесть данных, отвечает не на тот вопрос, ради которого открыта.

**У каждого числа есть ссылка на прогоны, из которых оно посчитано.** Поэтому
карта параметров рисуется разметкой, а не PNG: в PNG ячейка — тупик, а здесь она
ведёт к своим прогонам. PNG остаётся за ``plot-heatmap`` для отчётов.

**Ни одного внешнего адреса.** Ни CDN, ни шрифтов, ни иконок со стороны: админка
обязана открываться на машине без интернета, и то же свойство потребуется снимку
отчёта (ADM-7).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.templating import _TemplateResponse

from trading_research.admin import runs as runs_model
from trading_research.admin.state import (
    DatasetState,
    ExperimentState,
    ValidationState,
    experiment_names,
    experiment_state,
)
from trading_research.cli.status import (
    StatusReport,
    describe_dataset,
    describe_results,
    describe_schedule,
    describe_total,
    describe_validation,
    fmt_moment,
    fmt_periods,
    fmt_span,
    status_from_config,
    status_payload,
)
from trading_research.config import ExperimentConfig, load_experiment_config
from trading_research.hashing import git_commit_hash

TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class Roots:
    """Где лежит то, что админка читает. Ничего из этого она не пишет."""

    config: Path
    data: Path
    results: Path
    reports: Path
    log: Path | None = None


@dataclass(frozen=True)
class Shell:
    """То немногое, что нужно шапке любой страницы.

    Отдельно от ``PlatformState`` намеренно: страницам прогонов состояние данных
    не нужно, а собирать его ради заголовка значило бы платить 0,19 с за строку.
    """

    experiment: str
    config_path: Path
    now: datetime


@dataclass(frozen=True)
class DataRow:
    """Строка страницы «Данные»: покрытие ряда и вердикт о его качестве."""

    dataset: DatasetState
    validation: ValidationState | None

    @property
    def verdict(self) -> str:
        """Что известно о качестве: вердикт, его отсутствие или дефекты."""
        validation = self.validation
        if validation is None or not validation.exists:
            return "отчёта нет"
        if validation.error is not None:
            return f"отчёт не прочитан: {validation.error}"
        if validation.defects:
            return ", ".join(validation.defects)
        return "дефектов нет"

    @property
    def checked_at(self) -> str:
        """Когда проверяли. «Время неизвестно» — не то же самое, что «только что»."""
        validation = self.validation
        if validation is None or not validation.exists:
            return "—"
        if not validation.is_dated:
            return "время неизвестно"
        return fmt_moment(validation.checked_at)

    @property
    def is_defect(self) -> bool:
        return self.validation is not None and self.validation.is_clean is False


def create_app(roots: Roots) -> FastAPI:
    """Собрать приложение админки. Все маршруты — ``GET``, записи нет нигде."""
    app = FastAPI(title="trading-research admin", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["moment"] = fmt_moment
    templates.env.filters["span"] = fmt_span
    templates.env.filters["periods"] = fmt_periods
    templates.env.filters["metric"] = fmt_metric
    templates.env.filters["share"] = fmt_share
    templates.env.filters["params"] = fmt_params

    def report() -> StatusReport:
        return status_from_config(
            roots.config,
            data_root=roots.data,
            results_root=roots.results,
            reports_root=roots.reports,
            log_path=roots.log,
        )

    def config() -> ExperimentConfig:
        return load_experiment_config(roots.config)

    def shell() -> Shell:
        cfg = config()
        return Shell(experiment=cfg.name, config_path=roots.config, now=datetime.now(UTC))

    @app.get("/")
    def overview(request: Request) -> _TemplateResponse:
        current = report()
        return templates.TemplateResponse(
            request=request,
            name="overview.html",
            context={
                "shell": Shell(
                    experiment=current.state.experiment,
                    config_path=current.state.config_path,
                    now=current.state.now,
                ),
                "report": current,
                "state": current.state,
                # Строки блоков — те же, что печатает `status`: одна правда.
                "schedule_line": describe_schedule(current.state),
                "validation_line": describe_validation(current),
                "results_line": describe_results(current.state),
                "total_line": describe_total(current),
                "page": "overview",
            },
        )

    @app.get("/data")
    def data(request: Request) -> _TemplateResponse:
        current = report()
        return templates.TemplateResponse(
            request=request,
            name="data.html",
            context={
                "shell": Shell(
                    experiment=current.state.experiment,
                    config_path=current.state.config_path,
                    now=current.state.now,
                ),
                "report": current,
                "state": current.state,
                "pairs": data_rows(current),
                "page": "data",
                "describe_dataset": describe_dataset,
            },
        )

    @app.get("/experiments")
    def experiments(request: Request) -> _TemplateResponse:
        return templates.TemplateResponse(
            request=request,
            name="experiments.html",
            context={
                "shell": shell(),
                "experiments": experiment_overviews(roots.results),
                "page": "experiments",
            },
        )

    @app.get("/experiments/{name}")
    def experiment(
        request: Request,
        name: str,
        metric: str = runs_model.DEFAULT_METRIC,
        top: int = runs_model.DEFAULT_TOP,
        symbol: str | None = None,
        timeframe: str | None = None,
        x: str | None = None,
        y: str | None = None,
        agg: str = runs_model.DEFAULT_AGG,
    ) -> _TemplateResponse:
        state = _experiment_or_404(roots.results, name)
        table = runs_model.performance_table(roots.results, name)
        metric = _checked(metric, runs_model.metric_names(), runs_model.DEFAULT_METRIC)
        agg = _checked(agg, runs_model.AGGS, runs_model.DEFAULT_AGG)

        context: dict[str, object] = {
            "shell": shell(),
            "page": "experiments",
            "experiment": state,
            "table": table,
            "metric": metric,
            "metrics": runs_model.metric_names(),
            "top": top,
            "symbol": symbol,
            "timeframe": timeframe,
            "agg": agg,
            "aggs": runs_model.AGGS,
            "symbols": sorted({pair[0] for pair in state.pairs}),
            "timeframes": sorted({pair[1] for pair in state.pairs}),
        }
        if table is not None and table.height:
            axes = runs_model.param_names(table)
            x_axis = x if x in axes else (axes[0] if axes else None)
            y_axis = y if y in axes else (axes[1] if len(axes) > 1 else x_axis)
            context |= {
                "ranking": runs_model.ranking(
                    table,
                    experiment=name,
                    metric=metric,
                    top=top,
                    symbol=symbol,
                    timeframe=timeframe,
                ),
                "stability": runs_model.stability(
                    table, metric=metric, top=top, symbol=symbol, timeframe=timeframe
                ),
                "heatmap": (
                    runs_model.heatmap(
                        table,
                        x=x_axis,
                        y=y_axis,
                        metric=metric,
                        agg=agg,
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                    if x_axis is not None and y_axis is not None
                    else None
                ),
                "axes": axes,
                "x": x_axis,
                "y": y_axis,
            }
        return templates.TemplateResponse(request=request, name="experiment.html", context=context)

    @app.get("/experiments/{name}/runs")
    def experiment_runs(
        request: Request,
        name: str,
        param_hash: str | None = None,
        param: Annotated[list[str] | None, Query()] = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        slice_id: Annotated[str | None, Query(alias="slice")] = None,
        reliable: bool | None = None,
        status: str | None = None,
        metric: str = runs_model.DEFAULT_METRIC,
    ) -> _TemplateResponse:
        state = _experiment_or_404(roots.results, name)
        store = runs_model.open_store(roots.results, name)
        metric = _checked(metric, runs_model.metric_names(), runs_model.DEFAULT_METRIC)
        params = parse_params(param or [])
        rows = (
            ()
            if store is None
            else runs_model.run_rows(
                store,
                param_hash_value=param_hash,
                params=params,
                symbol=symbol,
                timeframe=timeframe,
                slice_id=slice_id,
                reliable=reliable,
                status=status,
            )
        )
        return templates.TemplateResponse(
            request=request,
            name="runs.html",
            context={
                "shell": shell(),
                "page": "experiments",
                "experiment": state,
                "rows": rows,
                "metric": metric,
                "metrics": runs_model.metric_names(),
                "filters": describe_filters(
                    param_hash=param_hash,
                    params=params,
                    symbol=symbol,
                    timeframe=timeframe,
                    slice_id=slice_id,
                    reliable=reliable,
                    status=status,
                ),
                "unreliable": sum(1 for row in rows if row.reliable is False),
                "without_metrics": sum(1 for row in rows if row.metrics is None),
            },
        )

    @app.get("/experiments/{name}/runs/{run_hash}")
    def experiment_run(
        request: Request, name: str, run_hash: str, verify: str | None = None
    ) -> _TemplateResponse:
        _experiment_or_404(roots.results, name)
        store = runs_model.open_store(roots.results, name)
        cfg = config()
        detail = (
            None
            if store is None
            else runs_model.run_detail(
                store,
                run_hash,
                config_path=roots.config,
                data_root=roots.data,
                results_root=roots.results,
                # Модель исполнения живёт в конфиге, а не в каталоге: пересчитать
                # идентичность можно только для эксперимента этого конфига.
                config=cfg if cfg.name == name else None,
                head_commit=git_commit_hash(),
            )
        )
        if detail is None:
            raise HTTPException(status_code=404, detail=f"прогон {run_hash} не найден")
        # Пересчёт dataset_hash читает всё окно свечей (0,4 с на паре), поэтому
        # делается по требованию: страница про прогон, а не про данные.
        data_check = (
            runs_model.check_data(
                cfg,
                detail.provenance.dataset_hash,
                symbol=detail.row.symbol,
                timeframe=detail.row.timeframe,
                data_root=roots.data,
            )
            if verify == "data" and cfg.name == name
            else None
        )
        return templates.TemplateResponse(
            request=request,
            name="run.html",
            context={
                "shell": shell(),
                "page": "experiments",
                "name": name,
                "detail": detail,
                "data_check": data_check,
                "metrics": runs_model.metric_names(),
            },
        )

    @app.get("/api/status")
    def api_status() -> JSONResponse:
        return JSONResponse(status_payload(report()))

    return app


def data_rows(report: StatusReport) -> list[tuple[tuple[str, str], list[DataRow]]]:
    """Строки страницы «Данные», сгруппированные по паре символ × таймфрейм.

    Вердикты приклеиваются к ряду, а не печатаются отдельным списком: вопрос «что с
    этим рядом» и вопрос «проверяли ли его» на странице должны иметь один ответ в
    одной строке.
    """
    verdicts = {(v.dataset, v.symbol, v.timeframe): v for v in report.state.validations}
    grouped: dict[tuple[str, str], list[DataRow]] = {}
    for dataset in report.state.datasets:
        key = (dataset.symbol, dataset.timeframe)
        row = DataRow(
            dataset=dataset,
            validation=verdicts.get((dataset.dataset, dataset.symbol, dataset.timeframe)),
        )
        grouped.setdefault(key, []).append(row)
    return list(grouped.items())


def experiment_overviews(results_root: str | Path) -> list[ExperimentState]:
    """Сводки всех экспериментов с каталогом результатов (ADM-1, без своих обходов)."""
    states = [experiment_state(results_root, name) for name in experiment_names(results_root)]
    return [state for state in states if state is not None]


def parse_params(raw: list[str]) -> dict[str, str]:
    """Фильтры по параметрам из query-строки: ``param=fast_ma:50``.

    Такой вид нужен ячейке карты параметров: она задаёт две оси сетки, а не
    полный набор, поэтому фильтр — подмножество значений, а не ``param_hash``.
    """
    parsed: dict[str, str] = {}
    for item in raw:
        key, _, value = item.partition(":")
        if key and value:
            parsed[key] = value
    return parsed


def describe_filters(
    *,
    param_hash: str | None,
    params: dict[str, str],
    symbol: str | None,
    timeframe: str | None,
    slice_id: str | None,
    reliable: bool | None,
    status: str | None,
) -> list[str]:
    """Чем сужен список прогонов — словами, а не повторением query-строки."""
    parts: list[str] = []
    if param_hash is not None:
        parts.append(f"набор параметров {param_hash}")
    for key, value in params.items():
        parts.append(f"{key} = {value}")
    if symbol is not None:
        parts.append(symbol)
    if timeframe is not None:
        parts.append(timeframe)
    if slice_id is not None:
        parts.append(f"слайс {slice_id}")
    if reliable is not None:
        parts.append("только надёжные" if reliable else "только ненадёжные")
    if status is not None:
        parts.append(f"статус {status}")
    return parts


def fmt_metric(value: float | None, digits: int = 3) -> str:
    """Метрика для страницы. «—» вместо ``None``: пустое значение не ноль."""
    if value is None:
        return "—"
    return f"{value:,.{digits}f}".replace(",", " ")


def fmt_share(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.0f} %"


def fmt_params(params: dict[str, object]) -> str:
    """Набор параметров как ``fast_ma=50, slow_ma=200`` — читается, а не парсится."""
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))


def _experiment_or_404(results_root: str | Path, name: str) -> ExperimentState:
    state = experiment_state(results_root, name)
    if state is None:
        raise HTTPException(status_code=404, detail=f"эксперимента {name} нет в результатах")
    return state


def _checked(value: str, allowed: tuple[str, ...], default: str) -> str:
    """Значение из query-строки — только из whitelist.

    Иначе опечатка в адресе роняет страницу пятисоткой из глубины аналитики,
    где метрика проверяется ``ValueError``'ом.
    """
    return value if value in allowed else default


__all__ = [
    "DataRow",
    "Roots",
    "Shell",
    "create_app",
    "data_rows",
    "describe_filters",
    "experiment_overviews",
    "fmt_metric",
    "fmt_params",
    "fmt_share",
    "parse_params",
]
