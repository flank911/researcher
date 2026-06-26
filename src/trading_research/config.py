"""INF-2 — конфиг эксперимента (pydantic v2) + загрузчик YAML.

Валидация даёт читаемые ошибки на уровне поля. Поле ``schema_version``
позволяет эволюционировать формат конфига.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from trading_research.domain import SCHEMA_VERSION, ExecutionModel, TimeSliceKind


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    exchange: str
    market: str
    symbols: list[str] = Field(min_length=1)
    timeframe: str
    start: datetime
    end: datetime


class TimeSlicesConfig(StrictModel):
    type: TimeSliceKind
    start: datetime | None = None
    end: datetime | None = None
    min_bars: int = Field(default=0, ge=0)


class StrategyConfig(StrictModel):
    name: str


class MetricsConfig(StrictModel):
    primary: str = "stability_score"
    secondary: list[str] = Field(default_factory=list)


class OutputConfig(StrictModel):
    save_trades: bool = True
    save_equity_curve: bool = True
    save_plots: bool = False


class ExperimentConfig(StrictModel):
    """Полный конфиг эксперимента (см. пример configs/experiments/)."""

    schema_version: int = SCHEMA_VERSION
    name: str
    description: str = ""
    data: DataConfig
    time_slices: TimeSlicesConfig
    strategy: StrategyConfig
    params_grid: dict[str, list[Any]] = Field(default_factory=dict)
    execution: ExecutionModel
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Загрузить и провалидировать YAML-конфиг эксперимента."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment config must be a mapping, got {type(raw).__name__}")
    return ExperimentConfig.model_validate(raw)
