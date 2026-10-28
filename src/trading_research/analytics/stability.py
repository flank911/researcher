"""AN-3 — robust-вектор стабильности параметров (улучшение №7).

Вместо хрупкой одиночной формулы — **вектор** устойчивости по каждой комбинации
параметров (``param_hash``), агрегированный по её временным окнам в таблице AN-1:

- ``median_return`` — медиана доходности по слайсам (робастнее среднего);
- ``positive_slices_ratio`` — доля прибыльных слайсов;
- ``worst_slice`` — худший слайс (минимальная доходность);
- ``top_quantile_freq`` — как часто параметр попадает в верхний квантиль **внутри
  слайса** (множество сравнения — все param-комбинации данного ``time_slice_id``);
- ``iqr`` / ``return_std`` — разброс доходности по слайсам.

Опциональный композит ``stability_score`` = ``median_return *
positive_slices_ratio − dispersion_weight * iqr`` (в тех же единицах доходности;
не делит на просадку/дисперсию, поэтому не даёт NaN/inf на граничных случаях —
нулевой разброс, один слайс). Основной вывод — вектор + многокритериальное
ранжирование; композит иллюстративен.

Множество сравнения для ``top_quantile_freq`` — строки с одним ``time_slice_id``;
для мультисимвольного/мультитаймфреймового анализа отфильтруйте таблицу заранее.
"""

from __future__ import annotations

import polars as pl

from trading_research.analytics.queries import QUERYABLE_METRICS

_SLICE_KEY = "time_slice_id"


def stability_vector(
    table: pl.DataFrame,
    *,
    return_metric: str = "total_return_pct",
    top_quantile: float = 0.25,
    dispersion_weight: float = 0.5,
) -> pl.DataFrame:
    """Вектор стабильности по ``param_hash`` поверх таблицы AN-1.

    ``top_quantile`` — доля верхних комбинаций внутри слайса, считающихся
    «топовыми» (0 < q <= 1). ``dispersion_weight`` — вес штрафа за ``iqr`` в
    композите.
    """
    if return_metric not in QUERYABLE_METRICS:
        raise ValueError(
            f"return_metric {return_metric!r} not queryable; "
            f"expected one of {sorted(QUERYABLE_METRICS)}"
        )
    if not 0.0 < top_quantile <= 1.0:
        raise ValueError(f"top_quantile must be in (0, 1], got {top_quantile}")

    # Попадание в верхний квантиль внутри каждого слайса (множество — весь слайс).
    rank_in_slice = pl.col(return_metric).rank(method="ordinal", descending=True).over(_SLICE_KEY)
    cutoff = (pl.len().over(_SLICE_KEY) * top_quantile).ceil()
    enriched = table.with_columns((rank_in_slice <= cutoff).alias("_in_top_q"))

    agg = (
        enriched.group_by("param_hash")
        .agg(
            pl.first("params_json").alias("params_json"),
            pl.len().alias("n_slices"),
            pl.col(return_metric).median().alias("median_return"),
            (pl.col(return_metric) > 0).mean().alias("positive_slices_ratio"),
            pl.col(return_metric).min().alias("worst_slice"),
            pl.col(return_metric).quantile(0.25).alias("_q25"),
            pl.col(return_metric).quantile(0.75).alias("_q75"),
            pl.col(return_metric).std().fill_null(0.0).alias("return_std"),
            pl.col("_in_top_q").mean().alias("top_quantile_freq"),
        )
        .with_columns((pl.col("_q75") - pl.col("_q25")).fill_null(0.0).alias("iqr"))
        .with_columns(
            (
                pl.col("median_return") * pl.col("positive_slices_ratio")
                - dispersion_weight * pl.col("iqr")
            ).alias("stability_score")
        )
        .drop("_q25", "_q75")
    )
    return agg.sort("param_hash")


def rank_by_stability(
    vector: pl.DataFrame, *, top: int = 20, descending: bool = True
) -> pl.DataFrame:
    """Отранжировать вектор по ``stability_score`` (ранг 1 — самый стабильный)."""
    if top <= 0:
        raise ValueError(f"top must be > 0, got {top}")
    ranked = vector.sort("stability_score", descending=descending, nulls_last=True).head(top)
    return ranked.with_row_index("rank", offset=1)


__all__ = ["rank_by_stability", "stability_vector"]
