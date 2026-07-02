"""BT-7 — расчёт метрик результата прогона.

Метрики делятся на два семейства по источнику ряда:

- **Equity-based** (``total_return_pct``, ``max_drawdown_pct``, ``sharpe``,
  ``sortino``, ``recovery_factor``) — считаются из кривой капитала. Ряд доходностей
  для Sharpe/Sortino — это **бар-доходности** непрерывной equity-кривой
  (``equity[t] / equity[t-1] - 1``), а не доходности отдельных сделок. Выбор
  осознанный: в research-режиме equity непрерывна, баланс переносится между
  окнами (см. EXP-2), поэтому бар-ряд корректно отражает риск даже при редких
  сделках и одинаково масштабируется между таймфреймами.
- **Trade-based** (``profit_factor``, ``win_rate``, ``avg_trade_pct``,
  ``expectancy``) — из списка реализованных сделок.

Annualization factor **явный** и выводится из таймфрейма (число баров в году),
что делает Sharpe/Sortino сравнимыми между разными таймфреймами.

Надёжность на малой выборке: при ``trades_count < min_trades`` результат помечается
``reliable=False`` — trade-зависимые метрики статистически ненадёжны и их следует
трактовать с осторожностью (значения при этом сохраняются, не зануляются).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from trading_research.backtest.portfolio import Trade
from trading_research.data.schema import timeframe_to_ms

MS_PER_YEAR: int = 365 * 24 * 60 * 60 * 1000
DEFAULT_MIN_TRADES: int = 10


def periods_per_year(timeframe: str) -> float:
    """Число баров таймфрейма в календарном году (annualization factor)."""
    return MS_PER_YEAR / timeframe_to_ms(timeframe)


@dataclass(frozen=True)
class Metrics:
    """Результат расчёта метрик одного прогона.

    Поля с типом ``float | None`` равны ``None``, когда метрика не определена
    (например, ``profit_factor`` без убыточных сделок или ``sharpe`` на плоской
    equity-кривой).
    """

    total_return_pct: float
    max_drawdown_pct: float
    sharpe: float | None
    sortino: float | None
    profit_factor: float | None
    win_rate: float | None
    avg_trade_pct: float | None
    expectancy: float | None
    recovery_factor: float | None
    trades_count: int
    annualization_factor: float
    min_trades: int
    reliable: bool


def compute_metrics(
    equity_curve: pl.DataFrame,
    trades: list[Trade],
    timeframe: str,
    *,
    min_trades: int = DEFAULT_MIN_TRADES,
    risk_free_rate: float = 0.0,
) -> Metrics:
    """Рассчитать метрики по кривой капитала и списку сделок.

    Args:
        equity_curve: DataFrame с колонкой ``equity`` (в порядке баров).
        trades: реализованные сделки прогона.
        timeframe: таймфрейм баров (например ``"1h"``) для annualization.
        min_trades: порог, ниже которого результат помечается ненадёжным.
        risk_free_rate: годовая безрисковая ставка (доля), вычитается из
            доходностей по-барно при расчёте Sharpe/Sortino. По умолчанию 0.
    """
    equity = [float(x) for x in equity_curve["equity"].to_list()]
    ppy = periods_per_year(timeframe)
    rf_per_bar = risk_free_rate / ppy if ppy else 0.0

    total_return_pct = _total_return_pct(equity)
    max_dd_pct, max_dd_abs = _max_drawdown(equity)
    returns = _bar_returns(equity)
    excess = [r - rf_per_bar for r in returns]
    sharpe = _sharpe(excess, ppy)
    sortino = _sortino(excess, ppy)
    recovery_factor = _recovery_factor(equity, max_dd_abs)

    pnls = [t.pnl for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    trades_count = len(trades)
    profit_factor = _profit_factor(pnls)
    win_rate = _win_rate(pnls)
    avg_trade_pct = _mean(pnl_pcts) if pnl_pcts else None
    expectancy = _mean(pnls) if pnls else None

    return Metrics(
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_dd_pct,
        sharpe=sharpe,
        sortino=sortino,
        profit_factor=profit_factor,
        win_rate=win_rate,
        avg_trade_pct=avg_trade_pct,
        expectancy=expectancy,
        recovery_factor=recovery_factor,
        trades_count=trades_count,
        annualization_factor=ppy,
        min_trades=min_trades,
        reliable=trades_count >= min_trades,
    )


def _total_return_pct(equity: list[float]) -> float:
    if len(equity) < 2 or equity[0] <= 0:
        return 0.0
    return (equity[-1] / equity[0] - 1.0) * 100.0


def _max_drawdown(equity: list[float]) -> tuple[float, float]:
    """Максимальная просадка: (в процентах от пика, в абсолютных единицах)."""
    peak = equity[0] if equity else 0.0
    max_pct = 0.0
    max_abs = 0.0
    for e in equity:
        if e > peak:
            peak = e
        drop = peak - e
        if drop > max_abs:
            max_abs = drop
        if peak > 0 and drop / peak > max_pct:
            max_pct = drop / peak
    return max_pct * 100.0, max_abs


def _bar_returns(equity: list[float]) -> list[float]:
    """Простые по-барные доходности; бары с неположительным пиком пропускаются."""
    out: list[float] = []
    for prev, cur in zip(equity, equity[1:], strict=False):
        if prev > 0:
            out.append(cur / prev - 1.0)
    return out


def _sharpe(excess_returns: list[float], ppy: float) -> float | None:
    if len(excess_returns) < 2:
        return None
    std = _stdev(excess_returns)
    if std == 0.0:
        return None
    return _mean(excess_returns) / std * math.sqrt(ppy)


def _sortino(excess_returns: list[float], ppy: float) -> float | None:
    if len(excess_returns) < 2:
        return None
    downside = _downside_deviation(excess_returns)
    if downside == 0.0:
        return None
    return _mean(excess_returns) / downside * math.sqrt(ppy)


def _recovery_factor(equity: list[float], max_dd_abs: float) -> float | None:
    if len(equity) < 2 or max_dd_abs <= 0:
        return None
    net_profit = equity[-1] - equity[0]
    return net_profit / max_dd_abs


def _profit_factor(pnls: list[float]) -> float | None:
    """Отношение суммы прибылей к модулю суммы убытков.

    ``None`` — если нет сделок или нет убыточных сделок (метрика не определена).
    """
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p < 0)
    if not pnls or gross_loss == 0.0:
        return None
    return gross_profit / gross_loss


def _win_rate(pnls: list[float]) -> float | None:
    if not pnls:
        return None
    wins = sum(1 for p in pnls if p > 0)
    return wins / len(pnls) * 100.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _stdev(values: list[float]) -> float:
    """Выборочное стандартное отклонение (ddof=1)."""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def _downside_deviation(values: list[float]) -> float:
    """Отклонение вниз относительно нуля (ddof=1, нормировка на n-1)."""
    n = len(values)
    downside_sq = sum(v**2 for v in values if v < 0)
    return math.sqrt(downside_sq / (n - 1))
