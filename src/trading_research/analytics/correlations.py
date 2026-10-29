"""REG-3 — связи параметров стратегии с рыночным режимом.

Объединяет таблицу AN-1 ``parameter_performance_by_slice`` с ``time_slice_features``
(REG-2) по ``(symbol, timeframe, time_slice_id)`` и анализирует, как доходность
параметров связана с фичами режима (volatility / trend / funding / OI / LS) и с
дискретной меткой ``regime_label``. Отвечает на вопросы вида «какие параметры лучше
при высокой волатильности / в бычьем тренде / при перегретом funding».
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from trading_research.analytics.queries import REGIME_SLOTS

# Числовые фичи режима (всё из слотов AN-1, кроме дискретного regime_label).
NUMERIC_REGIME_FEATURES: tuple[str, ...] = tuple(c for c in REGIME_SLOTS if c != "regime_label")
_JOIN_KEYS = ("symbol", "timeframe", "time_slice_id")


def attach_regime_features(
    performance: pl.DataFrame,
    features: pl.DataFrame,
    *,
    keys: Sequence[str] = _JOIN_KEYS,
) -> pl.DataFrame:
    """Приклеить фичи режима (REG-2) к таблице производительности (AN-1).

    Слоты режима в ``performance`` (пустые плейсхолдеры) заменяются значениями из
    ``features``; left-join сохраняет все строки производительности.
    """
    key_list = list(keys)
    missing = [k for k in key_list if k not in performance.columns or k not in features.columns]
    if missing:
        raise ValueError(f"join keys missing in performance/features: {missing}")

    drop = [c for c in REGIME_SLOTS if c in performance.columns]
    base = performance.drop(drop)
    feat_cols = key_list + [c for c in features.columns if c not in key_list]
    return base.join(features.select(feat_cols), on=key_list, how="left")


def regime_correlations(
    joined: pl.DataFrame,
    *,
    metric: str = "total_return_pct",
    features: Sequence[str] | None = None,
    by: str | None = "param_hash",
) -> pl.DataFrame:
    """Корреляции Пирсона ``metric`` ↔ фичи режима.

    ``by="param_hash"`` — матрица «параметр × фича» (строка на комбинацию); ``by=None``
    — одна строка глобальной корреляции. Фичи с нулевой дисперсией дают ``null``.
    """
    if metric not in joined.columns:
        raise ValueError(f"metric {metric!r} not in table")
    feats = (
        list(features)
        if features is not None
        else [c for c in NUMERIC_REGIME_FEATURES if c in joined.columns]
    )
    if not feats:
        raise ValueError("no regime feature columns present to correlate")

    exprs = [pl.corr(pl.col(metric), pl.col(f)).alias(f) for f in feats]
    if by is None:
        return joined.select(exprs)
    return joined.group_by(by).agg(exprs).sort(by)


def regime_metric_summary(
    joined: pl.DataFrame,
    *,
    metric: str = "total_return_pct",
    by: Sequence[str] = ("param_hash", "regime_label"),
) -> pl.DataFrame:
    """Средняя ``metric`` по группам (по умолчанию ``param_hash`` × ``regime_label``)."""
    by_list = list(by)
    missing = [c for c in (*by_list, metric) if c not in joined.columns]
    if missing:
        raise ValueError(f"columns missing: {missing}")
    return (
        joined.group_by(by_list)
        .agg(pl.col(metric).mean().alias(metric), pl.len().alias("n"))
        .sort(by_list)
    )


def best_params_in_regime(
    joined: pl.DataFrame,
    *,
    regime_label: str,
    metric: str = "total_return_pct",
    top: int = 20,
    descending: bool = True,
) -> pl.DataFrame:
    """Top-N комбинаций параметров внутри конкретного режима по средней ``metric``."""
    if top <= 0:
        raise ValueError(f"top must be > 0, got {top}")
    sub = joined.filter(pl.col("regime_label") == regime_label)
    agg = sub.group_by("param_hash").agg(
        pl.first("params_json").alias("params_json"),
        pl.col(metric).mean().alias(metric),
        pl.len().alias("n"),
    )
    return agg.sort(metric, descending=descending, nulls_last=True).head(top)


__all__ = [
    "NUMERIC_REGIME_FEATURES",
    "attach_regime_features",
    "best_params_in_regime",
    "regime_correlations",
    "regime_metric_summary",
]
