"""BT-4/BT-5 — модели размера позиции.

Поддержаны ``fixed`` (фиксированный нотионал в котируемой валюте),
``percent_balance`` (процент от баланса) и ``risk_based`` (BT-5): объём подбирается
так, чтобы срабатывание stop-loss стоило ``position_size_value`` процентов баланса.
``risk_based`` требует заданного ``stop_loss_pct`` (дистанция до стопа); экспозиция
дополнительно ограничивается плечом.
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
        if execution.stop_loss_pct is None:
            raise ValueError("risk_based sizing requires stop_loss_pct to be set")
        # Риск = value% баланса; убыток на единицу при стопе = price * stop_loss_pct.
        stop_distance = price * execution.stop_loss_pct
        if stop_distance <= 0:
            return 0.0
        qty = balance * (value / 100.0) / stop_distance
        # Экспозиция ограничена плечом.
        max_qty = balance * leverage / price
        return min(qty, max_qty)
    else:  # pragma: no cover - защита от новых значений enum
        raise NotImplementedError(f"unsupported position_size_type: {size_type!r}")

    return notional / price
