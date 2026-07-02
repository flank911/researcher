"""EXP-1 — разворачивание сетки параметров в комбинации.

Порядок комбинаций **детерминирован**: ключи параметров сортируются, затем берётся
декартово произведение значений в порядке ключей. Порядок не зависит от порядка
ключей в YAML-конфиге — это важно для стабильного ``run_hash`` (INF-3) и
воспроизводимости.

Опциональный ``predicate`` отсеивает недопустимые комбинации до прогона (например
ограничение ``fast_ma < slow_ma`` в MA Cross).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import product
from typing import Any

ParamCombo = dict[str, Any]
Predicate = Callable[[ParamCombo], bool]


def expand_grid(
    params_grid: Mapping[str, Sequence[Any]],
    predicate: Predicate | None = None,
) -> list[ParamCombo]:
    """Развернуть ``params_grid`` в детерминированный список комбинаций.

    Args:
        params_grid: отображение ``имя_параметра -> список значений``.
        predicate: если задан, оставляются только комбинации, для которых он
            вернул ``True``.

    Returns:
        Список словарей параметров в детерминированном порядке. Пустой
        ``params_grid`` даёт одну пустую комбинацию ``[{}]`` (единственный прогон
        с дефолтными параметрами).
    """
    for name, values in params_grid.items():
        if len(values) == 0:
            raise ValueError(f"params_grid[{name!r}] is empty; every param needs >=1 value")

    keys = sorted(params_grid)
    combos = [
        dict(zip(keys, values, strict=True)) for values in product(*(params_grid[k] for k in keys))
    ]
    if predicate is not None:
        combos = [c for c in combos if predicate(c)]
    return combos
