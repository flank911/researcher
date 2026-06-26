"""Strategy implementations (pure signal generation).

Импорт модулей стратегий регистрирует их в реестре ``base``.
"""

from trading_research.strategies import ma_cross  # noqa: F401  (side-effect: register)
from trading_research.strategies.base import (
    available_strategies,
    get_strategy,
    register_strategy,
)

__all__ = ["available_strategies", "get_strategy", "register_strategy"]
