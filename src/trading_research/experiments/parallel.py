"""EXP-5 — параллельное исполнение прогонов.

Распределяет вычисление ранов по пулу процессов. Сохраняет семантику EXP-3
(dedupe + resume) и опирается на то, что ``compute`` сам персистит результат
(EXP-4: ``ResultsStore`` — свой Parquet-партишн на ран + SQLite-каталог в WAL,
безопасный при конкурентной записи нескольких процессов).

Гарантии:

- **dedupe** — раны, уже отмеченные в ``RunLedger``, не отправляются в пул.
- **resume** — маркер завершения пишется в *главном* процессе только после
  успешного возврата воркера; падение воркера (исключение в ``compute`` или смерть
  процесса) не помечает ран завершённым, поэтому повторный запуск доделает его.
- **изоляция сбоев** — исключение одного рана не отменяет остальные: оно
  фиксируется в ``failed_hashes``, прочие раны доводятся до конца.

``compute`` должен быть picklable (функция уровня модуля или ``functools.partial``
над ней), так как уходит в дочерний процесс.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

from trading_research.experiments.cache import RunLedger, RunSpec


@dataclass(frozen=True)
class ParallelSummary:
    total: int
    executed: int
    skipped: int
    failed: int
    executed_hashes: list[str] = field(default_factory=list)
    failed_hashes: list[str] = field(default_factory=list)


def execute_runs_parallel(
    specs: Iterable[RunSpec],
    ledger: RunLedger,
    compute: Callable[[RunSpec], None],
    *,
    max_workers: int | None = None,
) -> ParallelSummary:
    """Выполнить раны в пуле процессов с dedupe/resume и изоляцией сбоев.

    ``max_workers=None`` → ``os.cpu_count()`` (дефолт ``ProcessPoolExecutor``).
    """
    all_specs = list(specs)
    pending: list[RunSpec] = []
    skipped = 0
    for spec in all_specs:
        if ledger.is_done(spec.run_hash()):
            skipped += 1
        else:
            pending.append(spec)

    executed_hashes: list[str] = []
    failed_hashes: list[str] = []

    if pending:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_spec = {pool.submit(compute, spec): spec for spec in pending}
            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                run_hash = spec.run_hash()
                try:
                    future.result()
                except Exception:  # noqa: BLE001 — сбой одного рана не рушит эксперимент
                    failed_hashes.append(run_hash)
                    continue
                # Маркер — только после успеха воркера (resume при падении).
                ledger.mark_done(
                    run_hash, meta={"strategy": spec.strategy, "slice_id": spec.slice_id}
                )
                executed_hashes.append(run_hash)

    return ParallelSummary(
        total=len(all_specs),
        executed=len(executed_hashes),
        skipped=skipped,
        failed=len(failed_hashes),
        executed_hashes=executed_hashes,
        failed_hashes=failed_hashes,
    )


__all__ = ["ParallelSummary", "execute_runs_parallel"]
