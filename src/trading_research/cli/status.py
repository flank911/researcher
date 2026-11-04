"""ADM-2 — бизнес-логика команды ``status``: живо ли всё, одним вызовом.

README описывал проверку «живо ли расписание» как процедуру из трёх шагов, и
главный из них — **запустить сбор руками** и посмотреть, догружает он что-нибудь
или говорит «актуален, догружать нечего». У такой проверки два изъяна: она ходит в
сеть, и отвечает только про один контур из четырёх. Про остальные три — качество
данных, состав результатов, незакрытые прогоны — процедуры не было вовсе.

Состояние даёт read-model [ADM-1](../../../tickets/ADM-1.md), поэтому здесь только
два слоя: пороги, по которым состояние превращается в поводы вмешаться, и
форматирование.

## Пороги — не круглые числа, а следствия

**Отставание ряда: 3 периода.** Один период — норма: текущий бар ещё не закрыт, и
загрузка его намеренно не берёт (OPS-2). Два — запуск мог не совпасть с границей
бара. Три подряд означают, что ряд не наполняется.

**Ряд без известного периода: сутки.** У funding расписание выплат задаёт биржа, и
отставание в периодах не считается вовсе. Выплаты идут раз в 8 часов, так что сутки
тишины — это уже пропуск, а не сдвиг расписания.

**Запас OI/LS: 7 суток.** Окно биржи — 29 суток, сбор идёт каждые 6 часов, то есть
неделя запаса — это 28 пропущенных итераций подряд. Раньше — ложная тревога,
позже — поздно: потерянную историю этих рядов не восстановить ничем.

**Вердикт валидации старше данных.** Это не порог, а точное сравнение: если
проверка была раньше последней точки ряда, она просто не видела того, что лежит в
store сейчас. Отдельный повод — вердикт **без даты**: покрытие тогда не определить
в принципе. На сегодняшнем store таких 24 из 24, и это ровно та находка, из-за
которой в DATA-3 появилось время проверки; первый же прогон ``validate-data`` его
ставит.

## Чего команда не считает поводом

**Отсутствия лога сбора.** Расписание может быть не поднято на этой машине — это
решение владельца, а не дефект. Вердикт о сборе всё равно дают данные: свежесть
OI/LS.

**Отсутствия funding/OI/LS.** Они опциональны по контракту ``build_features``: без
них фичи режима остаются ``null``. Так же на них смотрит и ``validate-data``
(DATA-8) — расхождение между двумя командами на одном и том же вопросе было бы
хуже любого из двух ответов. А вот пустые свечи — повод: прогонять не на чем.

**Отсутствия эксперимента.** «Прогонов ещё не было» — это состояние платформы, а не
её неисправность.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from trading_research.admin.state import (
    DatasetState,
    PlatformState,
    ValidationState,
    collect_state,
)
from trading_research.cli.text import plural, points
from trading_research.data.storage.parquet_store import CANDLES

STALE_PERIODS = 3
"""Отставание ряда в периодах, после которого это уже не «бар не закрылся»."""

STALE_WITHOUT_PERIOD = timedelta(days=1)
"""Порог для рядов, чей период задаёт биржа (funding): выплаты раз в 8 часов."""

RESERVE_FLOOR = timedelta(days=7)
"""Запас по OI/LS, ниже которого сбор пора проверять руками: 28 итераций подряд."""

SCHEDULE_SILENCE = timedelta(days=1)
"""Тишина в логе сбора, после которой расписание стоит считать сломанным."""


@dataclass(frozen=True)
class Alert:
    """Повод вмешаться: блок, к которому он относится, и что именно не так."""

    block: str
    message: str


@dataclass(frozen=True)
class ValidationSummary:
    """Сводка по отчётам валидации одного эксперимента.

    Четыре счётчика вместо списка из 24 строк: вердикты интересны не по одному, а
    тем, сколько их не хватает, сколько нашли дефекты, сколько нельзя датировать и
    сколько не покрывают текущие данные.
    """

    total: int
    missing: int
    defects: int
    undated: int
    stale: int

    @property
    def present(self) -> int:
        return self.total - self.missing


@dataclass(frozen=True)
class StatusReport:
    """Состояние платформы плюс поводы вмешаться."""

    state: PlatformState
    validation: ValidationSummary
    alerts: tuple[Alert, ...]

    @property
    def needs_attention(self) -> bool:
        return bool(self.alerts)


def status_from_config(
    config_path: str | Path,
    *,
    data_root: str | Path,
    results_root: str | Path,
    reports_root: str | Path,
    log_path: str | Path | None = None,
    now: datetime | None = None,
) -> StatusReport:
    """Собрать состояние платформы и перечислить поводы вмешаться.

    Ничего не запускает и в сеть не ходит — всё читается с диска (ADM-1).
    """
    state = collect_state(
        config_path,
        data_root=data_root,
        results_root=results_root,
        reports_root=reports_root,
        log_path=log_path,
        now=now,
    )
    return StatusReport(
        state=state,
        validation=summarize_validation(state),
        alerts=tuple(alerts(state)),
    )


def summarize_validation(state: PlatformState) -> ValidationSummary:
    """Свести вердикты валидации к четырём числам."""
    last_points = {(d.dataset, d.symbol, d.timeframe): d.last_point for d in state.datasets}
    return ValidationSummary(
        total=len(state.validations),
        missing=sum(1 for v in state.validations if not v.exists),
        defects=sum(1 for v in state.validations if v.is_clean is False),
        undated=sum(1 for v in state.validations if v.exists and not v.is_dated),
        stale=sum(1 for v in state.validations if _is_stale(v, last_points)),
    )


def _is_stale(
    validation: ValidationState, last_points: dict[tuple[str, str, str], datetime | None]
) -> bool:
    """Вердикт получен раньше последней точки ряда, то есть её не видел."""
    checked_at = validation.checked_at
    if checked_at is None:
        return False
    last = last_points.get((validation.dataset, validation.symbol, validation.timeframe))
    return last is not None and checked_at < last


def alerts(state: PlatformState) -> list[Alert]:
    """Всё, что требует вмешательства; пустой список — вмешиваться не во что."""
    found: list[Alert] = []
    found.extend(_data_alerts(state))
    found.extend(_schedule_alerts(state))
    found.extend(_validation_alerts(state))
    found.extend(_results_alerts(state))
    return found


def _data_alerts(state: PlatformState) -> list[Alert]:
    found: list[Alert] = []
    for dataset in state.datasets:
        name = _pair(dataset)
        if dataset.is_empty:
            # Пустые свечи — повод: прогонять не на чем. Пустые funding/OI/LS —
            # нет: они опциональны, так же как в validate-data (DATA-8).
            if dataset.dataset == CANDLES:
                found.append(Alert("данные", f"{name} — свечей нет, прогонять не на чем"))
            continue
        if dataset.is_exhausted:
            found.append(
                Alert(
                    "данные",
                    f"{name} — НЕВОСПОЛНИМЫЙ ПРОПУСК: последняя точка старше окна биржи",
                )
            )
        elif dataset.reserve is not None and dataset.reserve < RESERVE_FLOOR:
            found.append(
                Alert("данные", f"{name} — запас до потери истории {_span(dataset.reserve)}")
            )
        found.extend(_lag_alert(dataset, name))
    return found


def _lag_alert(dataset: DatasetState, name: str) -> list[Alert]:
    """Отставание как повод — только у рядов, которые можно догрузить в любой момент.

    У OI/LS отставание на часы — норма: их создаёт расписание раз в шесть часов, то
    есть на 1h они **по построению** отстают на несколько баров. Замер на живом
    store: 7 периодов при живом сборе и `exit=0`. Повод у этих рядов один — запас до
    потери истории, а смерть расписания видна тишиной в логе; считать поводом
    каждый бар значило бы ругаться на нормальную работу.
    """
    lag, periods = dataset.lag, dataset.lag_periods
    if lag is None or dataset.retention is not None:
        return []
    if periods is None:
        if lag > STALE_WITHOUT_PERIOD:
            return [Alert("данные", f"{name} — нет новых точек {_span(lag)}")]
        return []
    if periods > STALE_PERIODS:
        return [Alert("данные", f"{name} — отстаёт на {_periods(periods)} ({_span(lag)})")]
    return []


def _schedule_alerts(state: PlatformState) -> list[Alert]:
    schedule = state.schedule
    # Лога может не быть: сбор по расписанию — решение владельца машины, а не
    # обязанность платформы. Вердикт о сборе дают данные, а не этот файл.
    if not schedule.exists:
        return []
    found: list[Alert] = []
    if schedule.last_iteration_failed:
        found.append(
            Alert(
                "расписание",
                f"последняя итерация сбора кончилась кодом {schedule.last_exit_status}",
            )
        )
    silence = schedule.silence
    if silence is None:
        found.append(Alert("расписание", "в логе сбора нет ни одной итерации"))
    elif silence > SCHEDULE_SILENCE:
        found.append(Alert("расписание", f"сбор молчит {_span(silence)}"))
    return found


def _validation_alerts(state: PlatformState) -> list[Alert]:
    summary = summarize_validation(state)
    found: list[Alert] = []
    if summary.missing:
        found.append(
            Alert("валидация", f"нет вердикта о качестве: отчётов не хватает {summary.missing}")
        )
    if summary.defects:
        found.append(Alert("валидация", f"дефекты в данных: отчётов с дефектами {summary.defects}"))
    if summary.undated:
        found.append(
            Alert(
                "валидация",
                f"вердикт без даты у {summary.undated} отчётов: "
                "покрытие текущих данных не определить",
            )
        )
    if summary.stale:
        found.append(Alert("валидация", f"вердикт старше данных у {summary.stale} отчётов"))
    return found


def _results_alerts(state: PlatformState) -> list[Alert]:
    results = state.results
    # Эксперимента может не быть вовсе — это состояние, а не неисправность.
    if results is None:
        return []
    unfinished = {
        status: count for status, count in results.by_status.items() if status != "completed"
    }
    return [
        Alert("эксперименты", f"прогонов в статусе {status}: {count}")
        for status, count in sorted(unfinished.items())
    ]


# --- отчёт ------------------------------------------------------------------


def describe_report(report: StatusReport) -> list[str]:
    """Весь вывод команды: четыре блока и, если есть, поводы вмешаться."""
    lines = ["Данные"]
    lines.extend(f"  {line}" for line in describe_data(report.state))
    lines.append("Расписание")
    lines.append(f"  {describe_schedule(report.state)}")
    lines.append("Валидация")
    lines.append(f"  {describe_validation(report)}")
    lines.append("Эксперименты")
    lines.append(f"  {describe_results(report.state)}")
    if report.alerts:
        lines.append("Внимание")
        lines.extend(f"  {alert.block}: {alert.message}" for alert in report.alerts)
    lines.append(describe_total(report))
    return lines


def describe_data(state: PlatformState) -> list[str]:
    return [describe_dataset(dataset) for dataset in state.datasets]


def describe_dataset(dataset: DatasetState) -> str:
    """Строка по одному ряду: покрытие, объём и то, насколько он отстал."""
    head = f"{_pair(dataset)}:"
    if dataset.is_empty:
        return f"{head} данных нет"

    rows = _rows(dataset)
    span = f"{_fmt(dataset.first_point)} … {_fmt(dataset.last_point)}"
    parts = [f"{head} {span}, {rows}"]
    lag, periods = dataset.lag, dataset.lag_periods
    if lag is not None and lag <= timedelta(0):
        parts.append("актуален")
    elif periods is not None:
        parts.append(f"отстаёт на {_periods(periods)}")
    elif lag is not None:
        parts.append(f"последняя точка {_span(lag)} назад")
    if dataset.is_exhausted:
        parts.append("ИСТОРИЯ ПОТЕРЯНА")
    elif dataset.reserve is not None:
        parts.append(f"запас {_span(dataset.reserve)}")
    return ", ".join(parts)


def describe_schedule(state: PlatformState) -> str:
    """Строка про сбор рыночных рядов: когда была итерация и чем кончилась."""
    schedule = state.schedule
    if not schedule.exists:
        return "лога сбора нет — расписание на этой машине не поднято"
    if schedule.last_iteration_at is None:
        return f"{schedule.log_path}: ни одной итерации в логе"

    parts = [f"последняя итерация {_fmt(schedule.last_iteration_at)}"]
    silence = schedule.silence
    if silence is not None and silence > timedelta(0):
        parts.append(f"{_span(silence)} назад")
    if schedule.last_exit_status is None:
        parts.append("ещё не кончилась")
    else:
        parts.append(f"exit={schedule.last_exit_status}")
    return ", ".join(parts)


def describe_validation(report: StatusReport) -> str:
    """Строка про вердикты: сколько есть, сколько с дефектами, сколько без даты."""
    summary = report.validation
    if summary.total == 0:
        return "проверять нечего: в конфиге нет пар"
    if summary.present == 0:
        return f"отчётов нет ни одного (ожидалось {summary.total})"

    parts = [f"отчётов {summary.present} из {summary.total}"]
    parts.append(f"с дефектами {summary.defects}" if summary.defects else "дефектов нет")
    if summary.undated:
        parts.append(f"без даты {summary.undated}")
    if summary.stale:
        parts.append(f"старше данных {summary.stale}")
    return ", ".join(parts)


def describe_results(state: PlatformState) -> str:
    """Строка про прогоны: сколько, в каких статусах и сколько из них ненадёжны."""
    results = state.results
    if results is None:
        return "прогонов ещё не было"

    runs = plural(results.runs, "прогон", "прогона", "прогонов")
    statuses = ", ".join(
        f"{status}: {count}" for status, count in sorted(results.by_status.items())
    )
    parts = [f"{runs} ({statuses})"]
    if results.runs:
        parts.append(
            f"ненадёжных по числу сделок {results.unreliable} ({results.unreliable_share:.0%})"
        )
    if results.last_run_at is not None:
        parts.append(f"последний {_fmt(results.last_run_at)}")
    return ", ".join(parts)


def describe_total(report: StatusReport) -> str:
    """Итог — то, на что смотрят, когда не читают остальное."""
    if not report.alerts:
        return "итого: вмешательство не требуется"
    blocks = sorted({alert.block for alert in report.alerts})
    reasons = plural(len(report.alerts), "повод", "повода", "поводов")
    return f"итого: требуется вмешательство — {reasons} ({', '.join(blocks)})"


def status_payload(report: StatusReport) -> dict[str, object]:
    """То же состояние для машинного чтения (``--json``).

    Существует не ради красоты: без него единственным потребителем состояния
    оказался бы экран, и веб-админке (ADM-3) пришлось бы считать пороги заново.
    """
    state = report.state
    summary = report.validation
    return {
        "experiment": state.experiment,
        "config": str(state.config_path),
        "now": state.now.isoformat(),
        "needs_attention": report.needs_attention,
        "datasets": [
            {
                "dataset": d.dataset,
                "symbol": d.symbol,
                "timeframe": d.timeframe,
                "rows": d.rows,
                "first_point": _iso(d.first_point),
                "last_point": _iso(d.last_point),
                "lag_seconds": _seconds(d.lag),
                "lag_periods": d.lag_periods,
                "reserve_seconds": _seconds(d.reserve),
                "exhausted": d.is_exhausted,
            }
            for d in state.datasets
        ],
        "schedule": {
            "log": str(state.schedule.log_path) if state.schedule.log_path else None,
            "exists": state.schedule.exists,
            "last_iteration_at": _iso(state.schedule.last_iteration_at),
            "last_exit_status": state.schedule.last_exit_status,
            "silence_seconds": _seconds(state.schedule.silence),
        },
        "validation": {
            "total": summary.total,
            "present": summary.present,
            "missing": summary.missing,
            "defects": summary.defects,
            "undated": summary.undated,
            "stale": summary.stale,
        },
        "results": None
        if state.results is None
        else {
            "runs": state.results.runs,
            "by_status": dict(state.results.by_status),
            "unreliable": state.results.unreliable,
            "unreliable_share": state.results.unreliable_share,
            "without_metrics": state.results.without_metrics,
            "slices": state.results.slices,
            "param_sets": state.results.param_sets,
            "strategies": list(state.results.strategies),
            "git_commits": list(state.results.git_commits),
            "last_run_at": _iso(state.results.last_run_at),
        },
        "walk_forward": [
            {
                "symbol": w.symbol,
                "timeframe": w.timeframe,
                "exists": w.exists,
                "folds": w.folds,
                "has_holdout": w.has_holdout,
            }
            for w in state.walk_forward
        ],
        "alerts": [{"block": a.block, "message": a.message} for a in report.alerts],
    }


def _pair(dataset: DatasetState) -> str:
    return f"{dataset.symbol} {dataset.timeframe} {dataset.dataset}"


def _rows(dataset: DatasetState) -> str:
    """Объём ряда: барами для свечей, точками для остальных — как в OPS-1 и OPS-2.

    Разряды разделены, потому что числа здесь пятизначные: «42 344 бара» читается,
    «42344 бара» — нет. Группировка живёт здесь, а не в ``cli/text``: остальные
    команды печатают приросты в десятках баров, и менять их вывод незачем.
    """
    if dataset.dataset == CANDLES:
        form = plural(dataset.rows, "бар", "бара", "баров")
    else:
        form = points(dataset.rows)
    count, noun = form.split(" ", 1)
    return f"{int(count):,} {noun}".replace(",", " ")


def _fmt(moment: datetime | None) -> str:
    return moment.strftime("%Y-%m-%d %H:%M") if moment else "—"


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment else None


def _seconds(span: timedelta | None) -> float | None:
    return span.total_seconds() if span is not None else None


def _periods(count: int) -> str:
    return plural(count, "период", "периода", "периодов")


def _span(span: timedelta) -> str:
    """Длительность по-человечески: «3 суток 11 ч», «5 ч», «40 мин».

    Формат живёт здесь, а не в ``cli/text``: остальные команды печатают времена и
    бары, а длительности появились впервые — выносить общее стоит со второго
    вызывающего, не с первого.
    """
    total = int(abs(span).total_seconds())
    days, rest = divmod(total, 86_400)
    hours, rest = divmod(rest, 3_600)
    minutes = rest // 60
    if days:
        return (
            f"{plural(days, 'сутки', 'суток', 'суток')} {hours} ч"
            if hours
            else plural(days, "сутки", "суток", "суток")
        )
    if hours:
        return f"{hours} ч {minutes} мин" if minutes else f"{hours} ч"
    return f"{minutes} мин"


__all__ = [
    "RESERVE_FLOOR",
    "SCHEDULE_SILENCE",
    "STALE_PERIODS",
    "STALE_WITHOUT_PERIOD",
    "Alert",
    "StatusReport",
    "ValidationSummary",
    "alerts",
    "describe_data",
    "describe_dataset",
    "describe_report",
    "describe_results",
    "describe_schedule",
    "describe_total",
    "describe_validation",
    "status_from_config",
    "status_payload",
    "summarize_validation",
]
