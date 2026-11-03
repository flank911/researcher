"""ADM-1 — состояние платформы как данные (E10, админка).

Пакет отвечает на вопрос «что сейчас есть и что с этим не так», ничего не запуская.
Поверх него строятся команда ``status`` (ADM-2) и веб-админка (ADM-3…ADM-7).
"""

from trading_research.admin.state import (
    DatasetState,
    ExperimentState,
    PlatformState,
    ScheduleState,
    ValidationState,
    WalkForwardState,
    collect_state,
    dataset_states,
    experiment_names,
    experiment_state,
    schedule_state,
    validation_states,
    walk_forward_states,
)

__all__ = [
    "DatasetState",
    "ExperimentState",
    "PlatformState",
    "ScheduleState",
    "ValidationState",
    "WalkForwardState",
    "collect_state",
    "dataset_states",
    "experiment_names",
    "experiment_state",
    "schedule_state",
    "validation_states",
    "walk_forward_states",
]
