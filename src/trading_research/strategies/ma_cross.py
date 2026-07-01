"""BT-1 — стратегия пересечения скользящих средних (MA Cross).

Stop-and-reverse система: пересечение быстрой MA выше медленной — сигнал в long,
ниже — сигнал в short. Это первая стратегия MVP.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from trading_research.strategies.base import (
    LONG_SIGNAL,
    SHORT_SIGNAL,
    register_strategy,
)


def _parse_params(params: Mapping[str, Any]) -> tuple[int, int]:
    try:
        fast = int(params["fast_ma"])
        slow = int(params["slow_ma"])
    except KeyError as exc:
        raise KeyError(f"ma_cross requires param {exc}") from None
    if fast < 1 or slow < 1:
        raise ValueError(f"ma periods must be >= 1, got fast={fast}, slow={slow}")
    if fast >= slow:
        raise ValueError(f"fast_ma must be < slow_ma, got fast={fast}, slow={slow}")
    return fast, slow


class MovingAverageCrossStrategy:
    name = "ma_cross"

    def warmup_bars(self, params: Mapping[str, Any]) -> int:
        _, slow = _parse_params(params)
        return slow

    def generate_signals(self, data: pl.DataFrame, params: Mapping[str, Any]) -> pl.DataFrame:
        fast, slow = _parse_params(params)
        out = data.with_columns(
            pl.col("close").rolling_mean(fast).alias("fast_ma"),
            pl.col("close").rolling_mean(slow).alias("slow_ma"),
        )
        return out.with_columns(
            (
                (pl.col("fast_ma") > pl.col("slow_ma"))
                & (pl.col("fast_ma").shift(1) <= pl.col("slow_ma").shift(1))
            )
            .fill_null(False)  # noqa: FBT003
            .alias(LONG_SIGNAL),
            (
                (pl.col("fast_ma") < pl.col("slow_ma"))
                & (pl.col("fast_ma").shift(1) >= pl.col("slow_ma").shift(1))
            )
            .fill_null(False)  # noqa: FBT003
            .alias(SHORT_SIGNAL),
        )


register_strategy(MovingAverageCrossStrategy())
