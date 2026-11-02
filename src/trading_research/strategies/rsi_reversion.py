"""RES-1 — возврат к среднему на RSI (вторая стратегия платформы).

Это не «ещё одна стратегия для ассортимента», а **разделяющий эксперимент**. У
отрицательного вывода [RUN-7] два равно правдоподобных прочтения: «MA Cross не
работает на этих инструментах» и «платформа отвергает всё». На одной стратегии они
не различимы, а требуют противоположных действий. Поэтому вторая стратегия выбрана
структурно иной: MA Cross — трендследящая, эта — возврат к среднему, и предпочтения
по режиму у них противоположны. Если отрицателен и этот кандидат, вывод «дело не в
направлении логики» становится обоснованным, а не предположительным.

Логика: вход в зону перепроданности — сигнал в long (ждём возврата вверх), вход в
зону перекупленности — сигнал в short. Сигналы — события пересечения порога, а не
состояние «RSI ниже 30»: интерпретацию (вход/выход/переворот) делает слой исполнения,
как и для MA Cross (контракт BT-1).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from trading_research.features.indicators import rsi
from trading_research.strategies.base import (
    LONG_SIGNAL,
    SHORT_SIGNAL,
    register_strategy,
)

RSI_COLUMN = "rsi"

# Запас прогрева в периодах. Выбран замером, не вкусом: сглаживание Уайлдера
# экспоненциальное, вес затравки падает как ``exp(-k / period)``, то есть точного
# прогрева у RSI не существует. Максимальное расхождение значений RSI, посчитанных
# от разных точек старта, на всей истории BTCUSDT 1h (42 344 бара):
#
#   запас | period=7 | period=14 | period=21 | period=30
#      8x | 8,6e-03  | 1,3e-02   | 6,2e-04   | 1,3e-03
#     10x | 1,5e-03  | 5,1e-04   | 1,5e-03   | 5,9e-04
#     12x | 2,6e-04  | 1,2e-05   | 7,4e-06   | 8,1e-05
#
# При 12x расхождение ниже 1e-3 пункта RSI, то есть порог 30/70 оно сдвинуть не
# может. При запасе в один период расхождение доходит до 26 пунктов — такой прогрев
# менял бы сами сигналы.
WARMUP_PERIODS = 12


def _parse_params(params: Mapping[str, Any]) -> tuple[int, float, float]:
    try:
        period = int(params["rsi_period"])
        oversold = float(params["oversold"])
        overbought = float(params["overbought"])
    except KeyError as exc:
        raise KeyError(f"rsi_reversion requires param {exc}") from None
    if period < 2:
        raise ValueError(f"rsi_period must be >= 2, got {period}")
    if not 0.0 < oversold < overbought < 100.0:
        raise ValueError(
            "levels must satisfy 0 < oversold < overbought < 100, "
            f"got oversold={oversold}, overbought={overbought}"
        )
    return period, oversold, overbought


class RsiReversionStrategy:
    name = "rsi_reversion"

    def warmup_bars(self, params: Mapping[str, Any]) -> int:
        """Период на затравку плюс запас: сглаживание Уайлдера помнит всё.

        Возвращает ``WARMUP_PERIODS * period`` — обоснование и замеры см. выше.
        Заниженный прогрев здесь не падает: он тихо сдвигает значения RSI у начала
        окна, то есть меняет сами сигналы (WF-4).
        """
        period, _, _ = _parse_params(params)
        return WARMUP_PERIODS * period

    def generate_signals(self, data: pl.DataFrame, params: Mapping[str, Any]) -> pl.DataFrame:
        period, oversold, overbought = _parse_params(params)
        values = rsi([float(c) for c in data["close"].to_list()], period=period)
        out = data.with_columns(pl.Series(RSI_COLUMN, values, dtype=pl.Float64))
        current, previous = pl.col(RSI_COLUMN), pl.col(RSI_COLUMN).shift(1)
        return out.with_columns(
            # Вход в перепроданность: на прошлом баре порог ещё не пробит.
            ((current < oversold) & (previous >= oversold))
            .fill_null(False)  # noqa: FBT003
            .alias(LONG_SIGNAL),
            ((current > overbought) & (previous <= overbought))
            .fill_null(False)  # noqa: FBT003
            .alias(SHORT_SIGNAL),
        )


register_strategy(RsiReversionStrategy())
