"""ADM-3 — веб-админка, только для чтения.

Состояние даёт read-model ([ADM-1](../../../tickets/ADM-1.md)), пороги и
формулировки — команда ``status`` ([ADM-2](../../../tickets/ADM-2.md)). Здесь только
две страницы и ни одного обработчика, который что-нибудь меняет: запуск шагов
пайплайна — отдельный тикет (ADM-6), и до него админка физически не может ничего
испортить.

## Почему веб, а при чём тут CLI

Команда ``status`` отвечает «всё ли живо», но углубиться в ней нельзя: 684 прогона,
19 слайсов и шесть пар в строках терминала не разглядеть, а карту параметров в нём
не нарисовать вовсе. Ноутбук эту задачу тоже не решает: он требует ядра и
исполнения, и показывает состояние на момент запуска, а не на сейчас.

Выбран FastAPI с серверным HTML (Jinja) — без фронтенд-фреймворка. Причина не в
моде: страницы отдаются готовыми, и их можно проверить ``TestClient``'ом как обычные
функции, под ``mypy strict``. Streamlit отвергнут именно по этому: вся логика живёт в
скрипте, который перезапускается на каждый клик, и 644 теста такой код не удержат.

## Три решения

**Одна правда на две поверхности.** Страницы не считают ничего своего: блоки
«Расписание», «Валидация» и «Эксперименты» — это те же строки, что печатает
``status``, а ``/api/status`` отдаёт тот же payload, что ``status --json``.
Расхождение числа в CLI и на странице было бы хуже отсутствия страницы.

**Состояние читается на каждый запрос.** Кэша нет: сборка состояния на реальном
store — 0,19 с, а страница, показывающая вчерашнюю свежесть данных, отвечает не на
тот вопрос, ради которого открыта.

**Ни одного внешнего адреса.** Ни CDN, ни шрифтов, ни иконок со стороны: админка
обязана открываться на машине без интернета, и то же свойство потребуется снимку
отчёта (ADM-7).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.templating import _TemplateResponse

from trading_research.admin.state import DatasetState, ValidationState
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

    def report() -> StatusReport:
        return status_from_config(
            roots.config,
            data_root=roots.data,
            results_root=roots.results,
            reports_root=roots.reports,
            log_path=roots.log,
        )

    @app.get("/")
    def overview(request: Request) -> _TemplateResponse:
        current = report()
        return templates.TemplateResponse(
            request=request,
            name="overview.html",
            context={
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
                "report": current,
                "state": current.state,
                "pairs": data_rows(current),
                "page": "data",
                "describe_dataset": describe_dataset,
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


__all__ = ["DataRow", "Roots", "create_app", "data_rows"]
