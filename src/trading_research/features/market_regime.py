"""REG-2 — фичи рыночного режима на слайс + ``regime_label``.

Считает на каждый ``TimeSlice`` вектор фич (из OHLCV и рядов funding/OI/LS за
**этот же слайс**) и присваивает метку режима по правилам. Результат — таблица
``time_slice_features``; её колонки режима (``market_return_pct``, ``volatility``,
``trend_strength``, ``funding_avg``, ``oi_change_pct``, ``ls_ratio_avg``,
``regime_label``) совпадают по именам со слотами таблицы AN-1 (вход для REG-3).

**Без утечки будущего (улучшение №9):** фичи слайса вычисляются только из баров
``[slice.start, slice.end)`` — данных за пределами окна не используется. Для
walk-forward это и есть «режим из прошлого»: на момент решения (конец окна) все
использованные бары уже известны. Research-режим (разметка всей истории постфактум)
и торговый режим (только до точки решения) отличаются лишь тем, какие слайсы
доступны вызывающему коду; сама функция будущее не читает.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

import polars as pl

from trading_research.domain import FrozenModel, TimeSlice
from trading_research.features.funding_features import funding_avg
from trading_research.features.indicators import (
    adx,
    drawdown_from_ath,
    market_return_pct,
    trend_strength,
    volatility_pct,
    volume_zscore,
)
from trading_research.features.ls_ratio_features import long_short_ratio_avg
from trading_research.features.oi_features import oi_change_pct

_TIME_COL = "open_time"


class RegimeLabel(StrEnum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS_LOW_VOL = "sideways_low_vol"
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"
    PANIC_DUMP = "panic_dump"
    RECOVERY = "recovery"
    OVERHEATED_FUNDING = "overheated_funding"


class RegimeThresholds(FrozenModel):
    """Пороги правил классификации режима (в процентах, кроме долей)."""

    trend_strong: float = 0.5  # |корреляция| тренда
    bull_return: float = 5.0
    bear_return: float = -5.0
    vol_high: float = 1.0  # порог волатильности (per-bar std, %)
    dump_return: float = -15.0
    recovery_return: float = 5.0
    recovery_drawdown: float = -10.0
    funding_overheated: float = 0.0005  # средний funding за слайс


class SliceFeatures(FrozenModel):
    """Вектор фич режима одного слайса."""

    symbol: str
    timeframe: str
    time_slice_id: str
    market_return_pct: float
    volatility: float
    trend_strength: float
    adx: float
    funding_avg: float | None
    oi_change_pct: float | None
    ls_ratio_avg: float | None
    volume_zscore: float
    drawdown_from_ath: float
    regime_label: str


# Порядок колонок таблицы time_slice_features.
TIME_SLICE_FEATURE_COLUMNS: tuple[str, ...] = tuple(SliceFeatures.model_fields)

# Схема таблицы time_slice_features. Публичная: её нужно знать читателям, которым
# требуется пустая таблица с правильными типами (RUN-3 — нет данных ни по одному
# символу; RUN-5 — пустой join).
TIME_SLICE_FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String(),
    "timeframe": pl.String(),
    "time_slice_id": pl.String(),
    "market_return_pct": pl.Float64(),
    "volatility": pl.Float64(),
    "trend_strength": pl.Float64(),
    "adx": pl.Float64(),
    "funding_avg": pl.Float64(),
    "oi_change_pct": pl.Float64(),
    "ls_ratio_avg": pl.Float64(),
    "volume_zscore": pl.Float64(),
    "drawdown_from_ath": pl.Float64(),
    "regime_label": pl.String(),
}


def classify_regime(
    *,
    market_return: float,
    volatility: float,
    trend: float,
    drawdown: float,
    funding: float | None,
    thresholds: RegimeThresholds,
) -> RegimeLabel:
    """Присвоить метку режима по правилам (приоритет сверху вниз)."""
    thr = thresholds
    if funding is not None and funding > thr.funding_overheated:
        return RegimeLabel.OVERHEATED_FUNDING
    if market_return <= thr.dump_return and volatility >= thr.vol_high:
        return RegimeLabel.PANIC_DUMP
    if market_return >= thr.recovery_return and drawdown <= thr.recovery_drawdown:
        return RegimeLabel.RECOVERY
    if trend >= thr.trend_strong and market_return >= thr.bull_return:
        return RegimeLabel.BULL_TREND
    if trend <= -thr.trend_strong and market_return <= thr.bear_return:
        return RegimeLabel.BEAR_TREND
    if volatility >= thr.vol_high:
        return RegimeLabel.SIDEWAYS_HIGH_VOL
    return RegimeLabel.SIDEWAYS_LOW_VOL


def compute_slice_features(
    candles: pl.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    time_slice_id: str,
    funding: pl.DataFrame | None = None,
    oi: pl.DataFrame | None = None,
    ls: pl.DataFrame | None = None,
    adx_period: int = 14,
    thresholds: RegimeThresholds | None = None,
) -> SliceFeatures:
    """Посчитать вектор фич режима для одного (уже нарезанного по слайсу) окна."""
    thr = thresholds or RegimeThresholds()
    closes = [float(x) for x in candles["close"].to_list()]
    highs = [float(x) for x in candles["high"].to_list()]
    lows = [float(x) for x in candles["low"].to_list()]
    volumes = [float(x) for x in candles["volume"].to_list()]

    ret = market_return_pct(closes)
    vol = volatility_pct(closes)
    trend = trend_strength(closes)
    dd = drawdown_from_ath(closes)
    fund = funding_avg(funding)

    label = classify_regime(
        market_return=ret, volatility=vol, trend=trend, drawdown=dd, funding=fund, thresholds=thr
    )
    return SliceFeatures(
        symbol=symbol,
        timeframe=timeframe,
        time_slice_id=time_slice_id,
        market_return_pct=ret,
        volatility=vol,
        trend_strength=trend,
        adx=adx(highs, lows, closes, period=adx_period),
        funding_avg=fund,
        oi_change_pct=oi_change_pct(oi),
        ls_ratio_avg=long_short_ratio_avg(ls),
        volume_zscore=volume_zscore(volumes),
        drawdown_from_ath=dd,
        regime_label=label.value,
    )


def build_time_slice_features(
    candles: pl.DataFrame,
    slices: list[TimeSlice],
    *,
    symbol: str,
    timeframe: str,
    funding: pl.DataFrame | None = None,
    oi: pl.DataFrame | None = None,
    ls: pl.DataFrame | None = None,
    adx_period: int = 14,
    thresholds: RegimeThresholds | None = None,
) -> pl.DataFrame:
    """Собрать таблицу ``time_slice_features`` (строка на слайс).

    Каждый слайс использует только свои бары ``[start, end)`` (без будущего).
    """
    rows: list[dict[str, object]] = []
    for sl in slices:
        window = candles.filter((pl.col(_TIME_COL) >= sl.start) & (pl.col(_TIME_COL) < sl.end))
        if window.is_empty():
            continue
        features = compute_slice_features(
            window,
            symbol=symbol,
            timeframe=timeframe,
            time_slice_id=sl.id,
            funding=_window(funding, sl.start, sl.end),
            oi=_window(oi, sl.start, sl.end),
            ls=_window(ls, sl.start, sl.end),
            adx_period=adx_period,
            thresholds=thresholds,
        )
        rows.append(features.model_dump())
    return pl.DataFrame(rows, schema=_table_schema()).select(TIME_SLICE_FEATURE_COLUMNS)


def _window(df: pl.DataFrame | None, start: datetime, end: datetime) -> pl.DataFrame | None:
    if df is None:
        return None
    return df.filter((pl.col(_TIME_COL) >= start) & (pl.col(_TIME_COL) < end))


def _table_schema() -> dict[str, pl.DataType]:
    return dict(TIME_SLICE_FEATURE_SCHEMA)


__all__ = [
    "TIME_SLICE_FEATURE_COLUMNS",
    "TIME_SLICE_FEATURE_SCHEMA",
    "RegimeLabel",
    "RegimeThresholds",
    "SliceFeatures",
    "build_time_slice_features",
    "classify_regime",
    "compute_slice_features",
]
