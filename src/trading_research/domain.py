"""INF-2 — доменные типы (pydantic v2).

Ключевые сущности исследования. ``StrategyParams`` отделены от ``ExecutionModel``
намеренно — их нельзя смешивать.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1


class FrozenModel(BaseModel):
    """Базовая иммутабельная модель."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class TimeSliceKind(StrEnum):
    YEAR = "year"
    QUARTER = "quarter"
    MONTH = "month"
    WEEK = "week"
    CUSTOM = "custom"
    REGIME = "regime"


class TimeSliceRole(StrEnum):
    RESEARCH = "research"
    TRAIN = "train"
    TEST = "test"
    HOLDOUT = "holdout"


class PositionSizeType(StrEnum):
    FIXED = "fixed"
    PERCENT_BALANCE = "percent_balance"
    RISK_BASED = "risk_based"


class FillOn(StrEnum):
    NEXT_OPEN = "next_open"
    CURRENT_CLOSE = "current_close"  # допускается только для отладки


class TimeSlice(FrozenModel):
    """Временное окно — сущность первого класса.

    ``warmup_bars`` — сколько баров до ``start`` нужно подгрузить для прогрева
    индикаторов. Сделки и метрики считаются только начиная со ``start``.
    """

    id: str
    start: datetime
    end: datetime
    label: str
    kind: TimeSliceKind = TimeSliceKind.CUSTOM
    role: TimeSliceRole = TimeSliceRole.RESEARCH
    warmup_bars: int = Field(default=0, ge=0)
    min_bars: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_order(self) -> TimeSlice:
        if self.end <= self.start:
            raise ValueError(f"TimeSlice end must be after start: {self.start} >= {self.end}")
        return self


class DatasetRef(FrozenModel):
    """Ссылка на версионированный датасет (см. каталог DATA-4)."""

    id: str
    symbol: str
    timeframe: str
    exchange: str
    market: str
    start_at: datetime
    end_at: datetime
    data_path: str
    data_hash: str


class ExecutionModel(FrozenModel):
    """Параметры исполнения. Хранятся отдельно от параметров стратегии."""

    initial_balance: float = Field(gt=0)
    commission_rate: float = Field(ge=0)
    slippage_pct: float = Field(ge=0)
    leverage: int = Field(default=1, ge=1)
    position_size_type: PositionSizeType = PositionSizeType.PERCENT_BALANCE
    position_size_value: float = Field(gt=0)
    allow_short: bool = True
    allow_pyramiding: bool = False
    close_on_reverse_signal: bool = True
    signal_lag: int = Field(default=1, ge=0)
    fill_on: FillOn = FillOn.NEXT_OPEN
    # BT-5 — риск-слой. Все доли выражены как относительная дистанция от цены
    # входа (``0.02`` = 2%). ``None`` — риск-выход отключён.
    stop_loss_pct: float | None = Field(default=None, gt=0)
    take_profit_pct: float | None = Field(default=None, gt=0)
    trailing_stop_pct: float | None = Field(default=None, gt=0)
    # Ставка поддерживающей маржи для модели ликвидации (доля нотионала).
    maintenance_margin_rate: float = Field(default=0.005, ge=0, lt=1)
