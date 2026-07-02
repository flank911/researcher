"""EXP-1 — случайная выборка комбинаций параметров с фиксированным seed.

Выборка воспроизводима: при одинаковых ``params_grid``, ``n`` и ``seed`` результат
(включая порядок) идентичен. Реализация разворачивает полную сетку и берёт из неё
выборку без повторов — при MVP-масштабе сеток (десятки комбинаций) это дёшево и
даёт корректный дедуп «из коробки».
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from trading_research.experiments.grid_search import (
    ParamCombo,
    Predicate,
    expand_grid,
)


def random_search(
    params_grid: Mapping[str, Sequence[Any]],
    n: int,
    seed: int,
    predicate: Predicate | None = None,
) -> list[ParamCombo]:
    """Случайно выбрать до ``n`` уникальных комбинаций из сетки.

    Args:
        params_grid: отображение ``имя_параметра -> список значений``.
        n: желаемое число комбинаций. Если больше размера сетки — возвращается
            вся сетка (в перемешанном порядке).
        seed: seed генератора для воспроизводимости.
        predicate: фильтр допустимых комбинаций (см. ``expand_grid``).

    Returns:
        Список уникальных комбинаций в детерминированном (при данном seed) порядке.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    population = expand_grid(params_grid, predicate)
    rng = random.Random(seed)
    return rng.sample(population, min(n, len(population)))
