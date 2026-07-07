"""EXP-3 — run_hash, dedupe и resume.

Каждый атомарный юнит исследования — пара ``(комбинация параметров × TimeSlice)`` —
идентифицируется детерминированным ``run_hash`` (INF-3, ``stable_hash``) от полного
набора входов: стратегия, параметры, символ, таймфрейм, слайс, ``dataset_hash``,
модель исполнения и git-commit кода. Идентичные входы → тот же хэш; изменение
любого входа → другой хэш.

``RunLedger`` — durable множество завершённых ``run_hash`` на диске (по маркеру на
ран, запись атомарна через ``os.replace``). На нём строятся два свойства:

- **dedupe** — повторный запуск пропускает уже посчитанные раны;
- **resume** — маркер пишется только *после* успешного ``compute``, поэтому падение
  процесса не оставляет «полузавершённых» ранов: рестарт продолжает с места
  остановки.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_research.domain import ExecutionModel, TimeSlice
from trading_research.hashing import stable_hash


@dataclass(frozen=True)
class RunSpec:
    """Полная идентичность одного рана ``(параметры × слайс)``."""

    strategy: str
    params: dict[str, Any]
    symbol: str
    timeframe: str
    slice_id: str
    dataset_hash: str
    execution: ExecutionModel
    git_commit: str | None = None

    def run_hash(self, *, length: int = 16) -> str:
        """Детерминированный хэш всех входов рана."""
        return stable_hash(
            {
                "strategy": self.strategy,
                "params": self.params,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "slice_id": self.slice_id,
                "dataset_hash": self.dataset_hash,
                "execution": self.execution,
                "git_commit": self.git_commit,
            },
            length=length,
        )


def build_run_specs(
    *,
    strategy: str,
    symbol: str,
    timeframe: str,
    dataset_hash: str,
    execution: ExecutionModel,
    param_combos: Sequence[Mapping[str, Any]],
    slices: Sequence[TimeSlice],
    git_commit: str | None = None,
) -> list[RunSpec]:
    """Развернуть ``(комбинации × слайсы)`` в детерминированный список ``RunSpec``."""
    return [
        RunSpec(
            strategy=strategy,
            params=dict(params),
            symbol=symbol,
            timeframe=timeframe,
            slice_id=sl.id,
            dataset_hash=dataset_hash,
            execution=execution,
            git_commit=git_commit,
        )
        for params in param_combos
        for sl in slices
    ]


class RunLedger:
    """Durable множество завершённых ``run_hash`` (маркер-файл на ран)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _marker(self, run_hash: str) -> Path:
        return self.root / f"{run_hash}.json"

    def is_done(self, run_hash: str) -> bool:
        return self._marker(run_hash).exists()

    def mark_done(self, run_hash: str, meta: Mapping[str, Any] | None = None) -> None:
        """Атомарно пометить ран завершённым (запись в tmp + ``os.replace``)."""
        payload = {
            "run_hash": run_hash,
            "completed_at": datetime.now(UTC).isoformat(),
            "meta": dict(meta) if meta else {},
        }
        tmp = self.root / f".{run_hash}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._marker(run_hash))

    def completed(self) -> set[str]:
        return {p.stem for p in self.root.glob("*.json")}


@dataclass(frozen=True)
class RunSummary:
    total: int
    executed: int
    skipped: int
    executed_hashes: list[str]


def execute_runs(
    specs: Iterable[RunSpec],
    ledger: RunLedger,
    compute: Callable[[RunSpec], None],
) -> RunSummary:
    """Выполнить раны с dedupe и resume.

    Готовые (по ``ledger``) раны пропускаются. Для остальных вызывается
    ``compute(spec)``; ран помечается завершённым **только после** успешного
    возврата ``compute`` — если он бросит исключение, ран останется незавершённым
    и будет повторён при следующем запуске (resume).
    """
    executed_hashes: list[str] = []
    skipped = 0
    total = 0
    for spec in specs:
        total += 1
        run_hash = spec.run_hash()
        if ledger.is_done(run_hash):
            skipped += 1
            continue
        compute(spec)
        ledger.mark_done(run_hash, meta={"strategy": spec.strategy, "slice_id": spec.slice_id})
        executed_hashes.append(run_hash)
    return RunSummary(
        total=total,
        executed=len(executed_hashes),
        skipped=skipped,
        executed_hashes=executed_hashes,
    )
