"""BT-1 — интерфейс стратегии и реестр.

Стратегия — чистый генератор сигналов: она не знает о данных на диске, комиссиях,
размере позиции, сохранении результатов или id эксперимента. Только сигналы.

Контракт сигналов: ``generate_signals`` возвращает входной DataFrame с добавленными
индикаторными колонками и булевыми колонками-событиями ``long_signal`` /
``short_signal`` (без null — заполнены False на участке прогрева). Интерпретацию
(вход/выход/переворот) выполняет слой исполнения (BT-2+).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import polars as pl

LONG_SIGNAL = "long_signal"
SHORT_SIGNAL = "short_signal"
SIGNAL_COLUMNS = (LONG_SIGNAL, SHORT_SIGNAL)


@runtime_checkable
class Strategy(Protocol):
    """Протокол стратегии."""

    name: str

    def warmup_bars(self, params: Mapping[str, Any]) -> int:
        """Сколько баров прогрева нужно индикаторам при данных параметрах."""
        ...

    def generate_signals(self, data: pl.DataFrame, params: Mapping[str, Any]) -> pl.DataFrame:
        """Добавить индикаторы и булевы сигнальные колонки к ``data``."""
        ...


_REGISTRY: dict[str, Strategy] = {}


def register_strategy(strategy: Strategy) -> Strategy:
    """Зарегистрировать инстанс стратегии под его ``name``."""
    if strategy.name in _REGISTRY:
        raise ValueError(f"Strategy already registered: {strategy.name!r}")
    _REGISTRY[strategy.name] = strategy
    return strategy


def get_strategy(name: str) -> Strategy:
    """Получить зарегистрированную стратегию по имени."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown strategy: {name!r}. Available: {', '.join(sorted(_REGISTRY)) or '(none)'}"
        ) from None


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)
