"""Мелочи форматирования вывода CLI, общие для нескольких команд.

Здесь живёт только то, что иначе копировалось бы по модулям команд. Согласование
числительных — ровно такой случай: его завели в `update_market_data` (точки), затем
повторили в `validate_data` (бары), и к третьему разу (пары в `walk_forward`) три
копии одной таблицы окончаний разъехались бы молча.
"""

from __future__ import annotations


def plural(count: int, one: str, few: str, many: str) -> str:
    """«1 бар» / «3 бара» / «18 баров» — строку читают люди.

    Формы передаются вызывающим: для 1, для 2–4 и для остальных.
    """
    tail, hundred = count % 10, count % 100
    if tail == 1 and hundred != 11:
        form = one
    elif tail in (2, 3, 4) and hundred not in (12, 13, 14):
        form = few
    else:
        form = many
    return f"{count} {form}"


def points(count: int) -> str:
    """Частый случай: «1 точка» / «3 точки» / «18 точек»."""
    return plural(count, "точка", "точки", "точек")


__all__ = ["plural", "points"]
