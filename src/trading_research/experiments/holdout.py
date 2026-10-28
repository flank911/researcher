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
) -> Metrics:
    """Прогнать выбранные ``params`` на holdout и вернуть метрики (BT-7)."""
    if holdout_data.is_empty():
        raise ValueError("holdout_data is empty; nothing to evaluate")
    result = run_backtest(strategy, holdout_data, params, execution, engine=engine)
    return compute_metrics(result.equity_curve, result.trades, timeframe, min_trades=min_trades)


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
