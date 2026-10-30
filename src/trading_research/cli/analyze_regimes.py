"""RUN-5 — бизнес-логика команды ``analyze-regimes``.

REG-3 умеет считать корреляции «параметры ↔ режим», но вызвать его можно было только
из Python. Здесь — связка сохранённых таблиц: AN-1 (`performance.parquet`, RUN-1) ×
`time_slice_features` (RUN-3) → матрица корреляций и top-N параметров внутри режима.

**Слоты режима в AN-1 остаются null — и это не баг.** `performance.parquet` пишется
прогоном, который о фичах ничего не знает; заполняет слоты join на этапе анализа
(`attach_regime_features`). Так у `run-experiment` и `build-features` нет порядкового
зависимости друг от друга: их можно считать в любом порядке и даже параллельно.

**Пустой join — ошибка, а не таблица из null.** Это главный тихий отказ в этом
месте: `attach_regime_features` делает left-join, поэтому при несовпадении
`time_slice_id` (фичи построены по другому конфигу) он вернёт все режимные колонки
пустыми, а `pl.corr` по пустому столбцу отдаст `null` — аналитик получит матрицу
«корреляций нет» вместо сообщения «данные не склеились».
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from trading_research.analytics.correlations import (
    NUMERIC_REGIME_FEATURES,
    attach_regime_features,
    best_params_in_regime,
    describe_availability,
    regime_correlations,
    regime_feature_availability,
    regime_metric_summary,
)
from trading_research.analytics.queries import QUERYABLE_METRICS, load_performance_table
from trading_research.cli.analyze_results import experiment_results_path
from trading_research.cli.build_features import features_path

_REGIME_COL = "regime_label"


def load_joined(
    results_root: str | Path,
    features_root: str | Path,
    experiment: str,
) -> pl.DataFrame:
    """Склеить сохранённые таблицы AN-1 и ``time_slice_features``.

    Raises:
        FileNotFoundError: одна из таблиц не построена.
        ValueError: join не дал ни одной строки с режимом (ключи разошлись).
    """
    perf_path = experiment_results_path(results_root, experiment)
    feat_path = features_path(features_root, experiment)
    if not Path(perf_path).exists():
        raise FileNotFoundError(
            f"performance table not found: {perf_path}; run 'run-experiment' first"
        )
    if not Path(feat_path).exists():
        raise FileNotFoundError(
            f"features table not found: {feat_path}; run 'build-features' first"
        )

    performance = load_performance_table(perf_path)
    features = pl.read_parquet(feat_path)
    joined = attach_regime_features(performance, features)

    if joined.is_empty():
        raise ValueError("joined table is empty: performance table has no rows")
    if _REGIME_COL not in joined.columns or joined[_REGIME_COL].null_count() == joined.height:
        raise ValueError(
            "regime join produced no matches: every row has a null regime. "
            "Most likely the features table was built from a different config "
            "(mismatching symbol / timeframe / time_slice_id). "
            "Rebuild it with 'build-features --config <same config>'."
        )
    return joined


def analyze_regimes(
    results_root: str | Path,
    features_root: str | Path,
    experiment: str,
    *,
    metric: str = "total_return_pct",
    regime: str | None = None,
    top: int = 20,
    by_param: bool = True,
) -> pl.DataFrame:
    """Матрица корреляций ``metric`` ↔ фичи режима, либо top-N внутри режима.

    ``regime`` задан → top-N комбинаций параметров внутри этого режима
    (``best_params_in_regime``). Иначе — матрица корреляций; ``by_param=False``
    сводит её к одной глобальной строке.
    """
    if metric not in QUERYABLE_METRICS:
        raise ValueError(
            f"metric {metric!r} not queryable; expected one of {sorted(QUERYABLE_METRICS)}"
        )
    joined = load_joined(results_root, features_root, experiment)

    if regime is not None:
        known = set(joined[_REGIME_COL].drop_nulls().unique().to_list())
        if regime not in known:
            raise ValueError(f"regime {regime!r} not present; found: {sorted(known)}")
        return best_params_in_regime(joined, regime_label=regime, metric=metric, top=top)

    return regime_correlations(joined, metric=metric, by="param_hash" if by_param else None)


def regime_availability_lines(
    results_root: str | Path,
    features_root: str | Path,
    experiment: str,
) -> list[str]:
    """Строки про заполненность фич — печатаются под матрицей корреляций.

    Отдельная функция, а не поле результата: матрица уходит и в экспорт, где
    пояснительные строки были бы мусором в CSV.
    """
    joined = load_joined(results_root, features_root, experiment)
    return describe_availability(regime_feature_availability(joined))


def regime_summary(
    results_root: str | Path,
    features_root: str | Path,
    experiment: str,
    *,
    metric: str = "total_return_pct",
) -> pl.DataFrame:
    """Средняя ``metric`` по ``param_hash`` × ``regime_label`` (REG-3)."""
    joined = load_joined(results_root, features_root, experiment)
    return regime_metric_summary(joined, metric=metric)


def export_regimes(
    results_root: str | Path,
    features_root: str | Path,
    experiment: str,
    out_path: str | Path,
    *,
    metric: str = "total_return_pct",
    regime: str | None = None,
    top: int = 20,
    fmt: str = "csv",
) -> Path:
    """Выгрузить матрицу корреляций (или top-N по режиму) в CSV/Parquet."""
    if fmt not in ("csv", "parquet"):
        raise ValueError(f"unknown format {fmt!r}; expected 'csv' or 'parquet'")
    table = analyze_regimes(
        results_root, features_root, experiment, metric=metric, regime=regime, top=top
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        table.write_csv(out)
    else:
        table.write_parquet(out)
    return out


def available_features(joined: pl.DataFrame) -> list[str]:
    """Фичи режима, реально присутствующие в склеенной таблице."""
    return [c for c in NUMERIC_REGIME_FEATURES if c in joined.columns]


__all__ = [
    "analyze_regimes",
    "available_features",
    "export_regimes",
    "load_joined",
    "regime_summary",
]
