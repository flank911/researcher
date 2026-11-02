"""WF-3 — untouched holdout и контроль переоптимизации (улучшение №8).

Защита от data dredging / multiple testing: часть данных резервируется как
**holdout** и недоступна на research-фазе. Найденные на research закономерности
(выбранные параметры) проверяются на holdout **ровно один раз, в самом конце**.

Политика (важно соблюдать):

- Holdout задаётся :class:`HoldoutSplit` — хвостом периода (``holdout_start``) и/или
  набором символов (``holdout_symbols``), целиком уходящих в резерв.
- Research-код обращается к данным только через ``research_data(...)`` — holdout в
  research-запросы не попадает. ``assert_research_only(...)`` — страховочная
  проверка на уровне кода, что во фрейме нет holdout-баров.
- ``evaluate_holdout`` + ``build_holdout_report`` формируют **отдельный** отчёт по
  holdout и степень деградации метрики research → holdout. Запускать однократно.

**WF-4 — резерв прогревается барами до ``holdout_start``, и это не трата резерва.**
Прогон по резерву — отдельный бэктест, то есть он начинался с холодного индикатора:
на 4h при ``slow_ma=200`` первые 200 бар резерва (из ~1100) не давали сигналов
вовсе. Решение: индикаторы греются последними барами research-части, прогрев не
торгуется (``runner.run_backtest``), в метрики резерва его участок не входит.

Почему это не нарушает политику резерва: прогрев смотрит только **назад** от
оцениваемого бара, то есть не подглядывает в будущее; и на резерве по-прежнему
ничего не отбирается — параметры пришли с research-фазы, резерв смотрится один раз.
«Untouched» в WF-3 значит «на резерве не принимают решений», а не «резерв не знает,
какой была цена вчера». Обратное решение — не греть вовсе — означало бы, что отчёт
по резерву систематически недотарговывает тем сильнее, чем длиннее индикатор.

(Опц. на будущее: deflated Sharpe / White's Reality Check — здесь не реализованы.)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from trading_research.backtest.engine import BacktestEngine
from trading_research.backtest.metrics import DEFAULT_MIN_TRADES, Metrics, compute_metrics
from trading_research.domain import ExecutionModel
from trading_research.experiments.runner import run_backtest
from trading_research.strategies.base import Strategy

_TIME_COL = "open_time"


@dataclass(frozen=True)
class HoldoutSplit:
    """Политика резерва holdout: хвост периода и/или зарезервированные символы."""

    holdout_start: datetime | None = None
    holdout_symbols: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.holdout_start is None and not self.holdout_symbols:
            raise ValueError("HoldoutSplit requires holdout_start and/or holdout_symbols")
        if self.holdout_start is not None:
            object.__setattr__(self, "holdout_start", _to_utc(self.holdout_start))

    def is_holdout(self, *, symbol: str, time: datetime) -> bool:
        """Относится ли бар ``(symbol, time)`` к holdout."""
        if symbol in self.holdout_symbols:
            return True
        return self.holdout_start is not None and _to_utc(time) >= self.holdout_start

    def research_data(self, data: pl.DataFrame, *, symbol: str) -> pl.DataFrame:
        """Research-подвыборка: holdout исключён (пустая, если символ зарезервирован)."""
        if symbol in self.holdout_symbols:
            return data.clear()
        if self.holdout_start is None:
            return data
        return data.filter(pl.col(_TIME_COL) < self.holdout_start)

    def holdout_data(self, data: pl.DataFrame, *, symbol: str) -> pl.DataFrame:
        """Holdout-подвыборка (дополнение к research)."""
        if symbol in self.holdout_symbols:
            return data
        if self.holdout_start is None:
            return data.clear()
        return data.filter(pl.col(_TIME_COL) >= self.holdout_start)


def assert_research_only(data: pl.DataFrame, split: HoldoutSplit, *, symbol: str) -> None:
    """Проверка уровня кода: во фрейме нет holdout-баров (иначе ``ValueError``)."""
    holdout = split.holdout_data(data, symbol=symbol)
    if not holdout.is_empty():
        raise ValueError(
            f"research frame for {symbol!r} contains {holdout.height} holdout bar(s); "
            "use HoldoutSplit.research_data() to exclude the reserve"
        )


@dataclass(frozen=True)
class HoldoutReport:
    """Отдельный отчёт по holdout + деградация метрики research → holdout."""

    symbol: str
    params: dict[str, Any]
    metric: str
    research_value: float
    holdout_value: float
    degradation: float  # research_value - holdout_value (>0 → ухудшение на holdout)
    holdout_metrics: Metrics


def evaluate_holdout(
    holdout_data: pl.DataFrame,
    strategy: Strategy,
    params: dict[str, Any],
    execution: ExecutionModel,
    timeframe: str,
    *,
    min_trades: int = DEFAULT_MIN_TRADES,
    engine: BacktestEngine | None = None,
    warmup_data: pl.DataFrame | None = None,
) -> Metrics:
    """Прогнать выбранные ``params`` на holdout и вернуть метрики (BT-7).

    ``warmup_data`` — бары **непосредственно перед** резервом (хвост research-части):
    они греют индикаторы, не торгуются и в метрики не входят (WF-4, см. докстринг
    модуля). Без них прогон резерва стартует с холодного индикатора и теряет начало
    резерва тем сильнее, чем длиннее индикатор.
    """
    if holdout_data.is_empty():
        raise ValueError("holdout_data is empty; nothing to evaluate")
    frame, warmup_bars = _with_warmup(holdout_data, warmup_data)
    result = run_backtest(
        strategy, frame, params, execution, engine=engine, warmup_bars=warmup_bars
    )
    # Прогрев из метрик вырезан: сигналы на нём погашены, поэтому сделок там нет, а
    # кривая обрезается по тому же числу баров — равенство высот даёт движок.
    curve = result.equity_curve.slice(warmup_bars)
    return compute_metrics(curve, result.trades, timeframe, min_trades=min_trades)


def _with_warmup(
    holdout_data: pl.DataFrame, warmup_data: pl.DataFrame | None
) -> tuple[pl.DataFrame, int]:
    """Приклеить прогрев перед резервом; вернуть фрейм и число баров прогрева.

    Непрерывность проверяется, а не предполагается: прогрев, взятый не из баров
    прямо перед резервом, дал бы индикатору разрыв в истории и тихо неверные
    значения на первых барах резерва.
    """
    if warmup_data is None or warmup_data.is_empty():
        return holdout_data, 0
    last_warm = warmup_data[_TIME_COL][-1]
    first_holdout = holdout_data[_TIME_COL][0]
    if last_warm >= first_holdout:
        raise ValueError(
            f"warmup_data must end before the reserve starts: "
            f"last warmup bar {last_warm} >= first holdout bar {first_holdout}"
        )
    return pl.concat([warmup_data, holdout_data], how="vertical"), warmup_data.height


def build_holdout_report(
    *,
    symbol: str,
    params: dict[str, Any],
    research_value: float,
    holdout_metrics: Metrics,
    metric: str = "total_return_pct",
) -> HoldoutReport:
    """Собрать отчёт: значение метрики на research vs holdout и деградацию."""
    raw = getattr(holdout_metrics, metric)
    holdout_value = float("nan") if raw is None else float(raw)
    return HoldoutReport(
        symbol=symbol,
        params=dict(params),
        metric=metric,
        research_value=research_value,
        holdout_value=holdout_value,
        degradation=research_value - holdout_value,
        holdout_metrics=holdout_metrics,
    )


def save_holdout_report(report: HoldoutReport, path: str | Path) -> Path:
    """Сохранить отчёт по holdout в JSON (отдельный артефакт)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _to_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


__all__ = [
    "HoldoutReport",
    "HoldoutSplit",
    "assert_research_only",
    "build_holdout_report",
    "evaluate_holdout",
    "save_holdout_report",
]
