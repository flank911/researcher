"""AN-4 — heatmap карты параметров поверх таблицы AN-1.

Строит сетку ``x-параметр × y-параметр`` с цветом = агрегат (медиана по умолчанию)
выбранной метрики. Значения параметров извлекаются из ``params_json`` канонической
таблицы. Срезы — фильтры по ``symbol`` / ``timeframe`` / ``time_slice_id`` (год или
месяц задаются гранулярностью слайсов эксперимента).

``build_heatmap`` считает данные (без зависимости от бэкенда рисования),
``plot_heatmap`` рендерит PNG через matplotlib (Agg, headless).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from trading_research.analytics.queries import QUERYABLE_METRICS

_AGGS = ("median", "mean", "min", "max")


@dataclass(frozen=True)
class HeatmapData:
    """Матрица для heatmap: оси параметров и 2D-сетка агрегата метрики."""

    x_param: str
    y_param: str
    metric: str
    agg: str
    xs: list[Any]  # отсортированные значения x-параметра (столбцы)
    ys: list[Any]  # отсортированные значения y-параметра (строки)
    values: list[list[float | None]]  # values[y][x]; None — пары без данных

    def to_frame(self) -> pl.DataFrame:
        """Длинная форма ``(y_param, x_param, metric)`` для инспекции/тестов."""
        rows = [
            {self.y_param: y, self.x_param: x, self.metric: self.values[iy][ix]}
            for iy, y in enumerate(self.ys)
            for ix, x in enumerate(self.xs)
        ]
        return pl.DataFrame(rows)


def build_heatmap(
    table: pl.DataFrame,
    *,
    x: str,
    y: str,
    metric: str = "total_return_pct",
    agg: str = "median",
    symbol: str | None = None,
    timeframe: str | None = None,
    time_slice: str | None = None,
) -> HeatmapData:
    """Посчитать матрицу heatmap ``x × y`` по агрегату ``metric``."""
    if metric not in QUERYABLE_METRICS:
        raise ValueError(
            f"metric {metric!r} not queryable; expected one of {sorted(QUERYABLE_METRICS)}"
        )
    if agg not in _AGGS:
        raise ValueError(f"agg {agg!r} unknown; expected one of {list(_AGGS)}")

    df = table
    if symbol is not None:
        df = df.filter(pl.col("symbol") == symbol)
    if timeframe is not None:
        df = df.filter(pl.col("timeframe") == timeframe)
    if time_slice is not None:
        df = df.filter(pl.col("time_slice_id") == time_slice)
    if df.is_empty():
        raise ValueError("no rows after filtering; nothing to plot")

    parsed = [json.loads(s) for s in df["params_json"].to_list()]
    sample = parsed[0]
    for key in (x, y):
        if key not in sample:
            raise ValueError(f"param {key!r} not found in params_json (keys: {sorted(sample)})")

    decoded = df.with_columns(
        pl.Series("_x", [p[x] for p in parsed]),
        pl.Series("_y", [p[y] for p in parsed]),
    )
    agg_expr = getattr(pl.col(metric), agg)().alias("_value")
    long = decoded.group_by("_x", "_y").agg(agg_expr)

    xs = sorted(decoded["_x"].unique().to_list())
    ys = sorted(decoded["_y"].unique().to_list())
    lookup = {(row["_x"], row["_y"]): row["_value"] for row in long.to_dicts()}
    values = [[lookup.get((xv, yv)) for xv in xs] for yv in ys]

    return HeatmapData(x_param=x, y_param=y, metric=metric, agg=agg, xs=xs, ys=ys, values=values)


def plot_heatmap(data: HeatmapData, out_path: str | Path, *, title: str | None = None) -> Path:
    """Отрендерить heatmap в PNG (matplotlib Agg); вернуть путь к файлу."""
    import matplotlib

    matplotlib.use("Agg")  # headless, без дисплея
    import matplotlib.pyplot as plt

    grid = [[float("nan") if v is None else v for v in row] for row in data.values]

    fig, ax = plt.subplots(figsize=(max(4, len(data.xs)), max(3, len(data.ys))))
    image = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(data.xs)), [str(v) for v in data.xs])
    ax.set_yticks(range(len(data.ys)), [str(v) for v in data.ys])
    ax.set_xlabel(data.x_param)
    ax.set_ylabel(data.y_param)
    ax.set_title(title or f"{data.agg}({data.metric})")
    fig.colorbar(image, ax=ax, label=f"{data.agg}({data.metric})")

    # Аннотации значений в ячейках.
    for iy, row in enumerate(data.values):
        for ix, value in enumerate(row):
            if value is not None:
                ax.text(ix, iy, f"{value:.1f}", ha="center", va="center", color="white", fontsize=8)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def heatmap_to_png(
    table: pl.DataFrame,
    out_path: str | Path,
    *,
    x: str,
    y: str,
    metric: str = "total_return_pct",
    agg: str = "median",
    symbol: str | None = None,
    timeframe: str | None = None,
    time_slice: str | None = None,
    title: str | None = None,
) -> Path:
    """Посчитать и отрендерить heatmap в один вызов."""
    data = build_heatmap(
        table,
        x=x,
        y=y,
        metric=metric,
        agg=agg,
        symbol=symbol,
        timeframe=timeframe,
        time_slice=time_slice,
    )
    return plot_heatmap(data, out_path, title=title)


__all__ = ["HeatmapData", "build_heatmap", "heatmap_to_png", "plot_heatmap"]
