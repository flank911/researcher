"""BT-4 — модели размера позиции.

Поддержаны ``fixed`` (фиксированный нотионал в котируемой валюте) и
``percent_balance`` (процент от баланса). ``risk_based`` требует дистанции до
стопа и реализуется вместе с риск-слоем (BT-5).
"""

from __future__ import annotations

from trading_research.domain import ExecutionModel, PositionSizeType


def target_qty(execution: ExecutionModel, balance: float, price: float) -> float:
    """Целевой объём позиции в единицах базового актива.

    ``price`` — эффективная цена входа (уже с учётом проскальзывания).
    """
    if balance <= 0 or price <= 0:
        return 0.0

    size_type = execution.position_size_type
    value = execution.position_size_value
    leverage = execution.leverage

    if size_type is PositionSizeType.PERCENT_BALANCE:
        notional = balance * (value / 100.0) * leverage
    elif size_type is PositionSizeType.FIXED:
        notional = value * leverage
    elif size_type is PositionSizeType.RISK_BASED:
        raise NotImplementedError(
            "risk_based sizing requires stop distance and is implemented in BT-5"
        )
    else:  # pragma: no cover - защита от новых значений enum
        raise NotImplementedError(f"unsupported position_size_type: {size_type!r}")

    return notional / price
