"""REG-2 — чистые индикаторы для фич рыночного режима.

Все функции работают на списках значений **одного слайса** (без будущих баров) и
возвращают скаляр; на пустых/вырожденных входах дают ``0.0`` (без NaN/inf).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def bar_returns(closes: Sequence[float]) -> list[float]:
    """Побарные доходности ``close_i/close_{i-1} - 1`` (пропускает нулевые базы)."""
    out: list[float] = []
    for prev, cur in zip(closes, closes[1:], strict=False):
        out.append(cur / prev - 1.0 if prev else 0.0)
    return out


def market_return_pct(closes: Sequence[float]) -> float:
    """Доходность слайса в процентах: ``(last/first - 1) * 100``."""
    if len(closes) < 2 or closes[0] == 0:
        return 0.0
    return (closes[-1] / closes[0] - 1.0) * 100.0


def volatility_pct(closes: Sequence[float]) -> float:
    """Волатильность = стандартное отклонение побарных доходностей (в процентах)."""
    rets = bar_returns(closes)
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * 100.0


def trend_strength(closes: Sequence[float]) -> float:
    """Сила тренда = корреляция Пирсона ``close`` с индексом времени, в ``[-1, 1]``."""
    n = len(closes)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(closes) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, closes, strict=True))
    var_x = sum((x - mx) ** 2 for x in xs)
    var_y = sum((y - my) ** 2 for y in closes)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def volume_zscore(volumes: Sequence[float]) -> float:
    """Z-оценка объёма последнего бара относительно распределения слайса."""
    n = len(volumes)
    if n < 2:
        return 0.0
    mean = sum(volumes) / n
    var = sum((v - mean) ** 2 for v in volumes) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (volumes[-1] - mean) / std


def drawdown_from_ath(closes: Sequence[float]) -> float:
    """Максимальная просадка от бегущего пика внутри слайса (в процентах, ``<= 0``)."""
    if not closes:
        return 0.0
    peak = closes[0]
    worst = 0.0
    for c in closes:
        peak = max(peak, c)
        if peak > 0:
            worst = min(worst, (c / peak - 1.0) * 100.0)
    return worst


def adx(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], *, period: int = 14
) -> float:
    """Средний индекс направленного движения (Wilder ADX), последнее значение.

    Недостаточно данных (< ``2*period``) → ``0.0``.
    """
    n = len(closes)
    if n < 2 * period or not (len(highs) == len(lows) == n):
        return 0.0

    tr: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        tr.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )

    def _wilder(values: list[float]) -> list[float]:
        # Сглаживание Уайлдера: первый элемент — сумма первых ``period``.
        smoothed = [sum(values[:period])]
        for v in values[period:]:
            smoothed.append(smoothed[-1] - smoothed[-1] / period + v)
        return smoothed

    tr_s = _wilder(tr)
    plus_s = _wilder(plus_dm)
    minus_s = _wilder(minus_dm)

    dx: list[float] = []
    for t, p, m in zip(tr_s, plus_s, minus_s, strict=True):
        if t == 0:
            dx.append(0.0)
            continue
        plus_di = 100.0 * p / t
        minus_di = 100.0 * m / t
        denom = plus_di + minus_di
        dx.append(0.0 if denom == 0 else 100.0 * abs(plus_di - minus_di) / denom)

    if len(dx) < period:
        return 0.0
    return sum(dx[-period:]) / period


__all__ = [
    "adx",
    "bar_returns",
    "drawdown_from_ath",
    "market_return_pct",
    "trend_strength",
    "volatility_pct",
    "volume_zscore",
]
