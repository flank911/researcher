"""REG-3 — связи параметров стратегии с рыночным режимом.

Объединяет таблицу AN-1 ``parameter_performance_by_slice`` с ``time_slice_features``
(REG-2) по ``(symbol, timeframe, time_slice_id)`` и анализирует, как доходность
параметров связана с фичами режима (volatility / trend / funding / OI / LS) и с
дискретной меткой ``regime_label``. Отвечает на вопросы вида «какие параметры лучше
при высокой волатильности / в бычьем тренде / при перегретом funding».

**REG-4 — ``null`` в матрице двусмыслен.** ``pl.corr`` отдаёт ``null`` и когда связи
нет, и когда коррелировать нечего: столбец пуст или в нём одна точка. Для читателя
это разные ответы — «проверили, связи нет» против «не проверяли». Поэтому рядом с
матрицей считается ``regime_feature_availability``, и команда печатает, по каким
фичам данных не было вовсе.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class FeatureAvailability:
    """Сколько строк склеенной таблицы несут значение фичи режима.

    Нужна, чтобы отличать «данных нет» от «связи нет» (REG-4): `pl.corr` по
    пустому столбцу даёт `null` — ровно то же, что корреляция, которой не
    существует. На прогоне RUN-7 `oi_change_pct` и `ls_ratio_avg` были пусты
    целиком, а в матрице выглядели как честно посчитанный `NaN`.
    """

    feature: str
    filled: int
    total: int

    @property
    def is_missing(self) -> bool:
        """Данных нет вовсе — `null` в матрице означает не «связи нет»."""
        return self.filled == 0

    @property
    def is_constant(self) -> bool:
        """Данные есть, но корреляция всё равно будет `null`.

        Второй источник `null`, который легко принять за отсутствие связи:
        на одной точке корреляция не определена.
        """
        return self.filled == 1


def regime_feature_availability(
    joined: pl.DataFrame, *, features: Sequence[str] | None = None
) -> list[FeatureAvailability]:
    """Заполненность каждой фичи режима в склеенной таблице."""
    feats = (
        list(features)
        if features is not None
        else [c for c in NUMERIC_REGIME_FEATURES if c in joined.columns]
    )
    total = joined.height
    return [
        FeatureAvailability(feature=f, filled=total - joined[f].null_count(), total=total)
        for f in feats
    ]


def describe_availability(availability: list[FeatureAvailability]) -> list[str]:
    """Строки под матрицей корреляций: где `null` значит «данных не было»."""
    missing = [a.feature for a in availability if a.is_missing]
    constant = [a.feature for a in availability if a.is_constant]
    if not missing and not constant:
        return ["данные есть по всем фичам режима: null в матрице — это отсутствие связи"]

    lines = []
    if missing:
        lines.append(
            f"ДАННЫХ НЕТ по фичам: {', '.join(missing)} — "
            f"null у них означает «нечего коррелировать», а не «связи нет»"
        )
    if constant:
        lines.append(f"одна точка по фичам: {', '.join(constant)} — корреляция не определена")
    filled = [a for a in availability if not a.is_missing]
    if filled:
        lines.append("заполнено: " + ", ".join(f"{a.feature} {a.filled}/{a.total}" for a in filled))
    return lines


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
    "FeatureAvailability",
    "describe_availability",
    "regime_feature_availability",
    "attach_regime_features",
    "best_params_in_regime",
    "regime_correlations",
    "regime_metric_summary",
]
