"""INF-2 — конфиг эксперимента (pydantic v2) + загрузчик YAML.

Валидация даёт читаемые ошибки на уровне поля. Поле ``schema_version``
позволяет эволюционировать формат конфига.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trading_research.backtest.metrics import Metrics
from trading_research.domain import SCHEMA_VERSION, ExecutionModel, SliceMode, TimeSliceKind

# Числовые метрики, пригодные для отбора на train (WF-2 отвергает булев ``reliable``).
NUMERIC_METRICS: frozenset[str] = frozenset(f.name for f in fields(Metrics)) - {"reliable"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    """Источник данных эксперимента.

    RUN-6: ``timeframes`` — список, как и ``symbols``; прогон идёт по всем парам
    символ × таймфрейм. Скалярное ``timeframe`` из v1 намеренно **не**
    поддерживается: ``extra="forbid"`` даёт на него явную ошибку, а тихое
    превращение одного значения в список спрятало бы смену формата.
    """

    exchange: str
    market: str
    symbols: list[str] = Field(min_length=1)
    timeframes: list[str] = Field(min_length=1)
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


class WalkForwardConfig(StrictModel):
    """RUN-4 — параметры walk-forward (E5).

    Окна заданы **в днях**, а не «в периодах», намеренно: ``generate_walk_forward_folds``
    (WF-1) принимает ``timedelta``, а календарный месяц в ``timedelta`` точно не
    выражается. Поле `train_span: 6` потребовало бы молча считать месяц за 30 дней —
    то есть вернуть в конфиг скрытое допущение. В днях допущение выбирает автор
    конфига, и оно видно в самом имени поля.
    """

    train_days: int = Field(gt=0)
    test_days: int = Field(gt=0)
    step_days: int = Field(gt=0)
    select_by: str = "total_return_pct"
    # True — train расширяется от начала (expanding); False — скользящее окно.
    anchored: bool = False
    warmup_bars: int = Field(default=0, ge=0)

    @field_validator("select_by")
    @classmethod
    def _known_metric(cls, value: str) -> str:
        if value not in NUMERIC_METRICS:
            raise ValueError(
                f"select_by must be a numeric Metrics field; "
                f"got {value!r}, expected one of {sorted(NUMERIC_METRICS)}"
            )
        return value


class HoldoutConfig(StrictModel):
    """RUN-4 — политика untouched holdout (WF-3).

    ``start`` — начало резерва: всё от этого момента и позже в research не попадает.
    ``symbols`` — символы, целиком уходящие в резерв. Нужно хотя бы одно из двух,
    иначе резерв пуст и holdout бессмыслен.
    """

    start: datetime | None = None
    symbols: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _non_empty(self) -> HoldoutConfig:
        if self.start is None and not self.symbols:
            raise ValueError("holdout requires 'start' and/or non-empty 'symbols'")
        return self


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
    # EXP-2: непрерывный прогон с переносом баланса (дефолт) либо независимые окна.
    slice_mode: SliceMode = SliceMode.CONTINUOUS
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    # RUN-4: обе секции опциональны — `run-experiment` их не требует, а
    # `walk-forward` требует только `walk_forward`.
    walk_forward: WalkForwardConfig | None = None
    holdout: HoldoutConfig | None = None

    @field_validator("schema_version")
    @classmethod
    def _supported_version(cls, value: int) -> int:
        """Версия формата проверяется, а не просто хранится.

        До RUN-6 поле существовало, но ни с чем не сверялось — то есть было
        декоративным. Теперь несовпадение останавливает загрузку: v1-конфиг со
        скалярным ``timeframe`` получает и ошибку о неизвестном поле, и внятное
        объяснение, почему он больше не подходит.
        """
        if value != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {value}; this build expects {SCHEMA_VERSION}. "
                "v1 → v2: data.timeframe (скаляр) заменён на data.timeframes (список)"
            )
        return value


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Загрузить и провалидировать YAML-конфиг эксперимента."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment config must be a mapping, got {type(raw).__name__}")
    return ExperimentConfig.model_validate(raw)
