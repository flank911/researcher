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

**Гранулярность (RUN-1).** ``execute_runs_parallel`` считает один ``RunSpec``
независимой единицей — это верно только для ``SliceMode.INDEPENDENT``. В дефолтном
``CONTINUOUS`` (EXP-2) на комбинацию параметров делается **один** прогон по всей
истории, а метрики окна берут базой equity на баре перед его началом, поэтому
слайсы одной комбинации не независимы. Для таких случаев есть
``execute_run_groups_parallel``: единица работы — *группа* ранов (все слайсы одной
комбинации), которую воркер считает за один прогон.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

from trading_research.experiments.cache import RunLedger, RunSpec

# Пул процессов создаётся только через spawn. На Linux дефолт — fork (до 3.14),
# а форк процесса, в котором уже работал polars, насмерть вешает ребёнка: пул
# потоков rayon в дочерний процесс не переносится, и первый же `collect()` виснет
# в `LockLatch::wait_and_reset` навсегда. На macOS дефолт spawn, поэтому локально
# это не воспроизводится — висит только CI.
#
# Имя публичное: тот же контекст нужен отбору на train в walk-forward (RUN-8), а
# вторая копия этой строки означала бы второй шанс однажды получить fork и
# неотлаживаемый зависший CI.
MP_CONTEXT = multiprocessing.get_context("spawn")


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
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=MP_CONTEXT) as pool:
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


def execute_run_groups_parallel(
    groups: Iterable[Sequence[RunSpec]],
    ledger: RunLedger,
    compute_group: Callable[[Sequence[RunSpec]], None],
    *,
    max_workers: int | None = None,
) -> ParallelSummary:
    """Выполнить ран-**группы** в пуле процессов с dedupe/resume и изоляцией сбоев.

    Группа — набор ранов, считающихся за один прогон движка (в ``CONTINUOUS`` это
    все слайсы одной комбинации параметров). ``compute_group`` получает всю группу
    и сам персистит результат каждого рана.

    Отличия от ``execute_runs_parallel``:

    - **dedupe на уровне группы** — группа пропускается, только если готовы *все*
      её раны. Частично готовая группа считается заново: в ``CONTINUOUS`` отдельный
      слайс и нельзя пересчитать без полного прогона, а повторное сохранение
      идемпотентно (``ResultsStore.save_run``).
    - **resume** — маркеры всех ранов группы пишутся в главном процессе только
      после успешного возврата воркера.

    ``max_workers=1`` исполняет группы в текущем процессе (без пула) — семантика
    та же, удобно для отладки и детерминированного сравнения с параллельным
    прогоном.
    """
    all_groups = [list(g) for g in groups]
    total = sum(len(g) for g in all_groups)

    pending: list[list[RunSpec]] = []
    skipped = 0
    for group in all_groups:
        if group and all(ledger.is_done(s.run_hash()) for s in group):
            skipped += len(group)
        elif group:
            pending.append(group)

    executed_hashes: list[str] = []
    failed_hashes: list[str] = []

    def _mark(group: Sequence[RunSpec]) -> None:
        for spec in group:
            ledger.mark_done(
                spec.run_hash(), meta={"strategy": spec.strategy, "slice_id": spec.slice_id}
            )
            executed_hashes.append(spec.run_hash())

    if pending and max_workers == 1:
        for group in pending:
            try:
                compute_group(group)
            except Exception:  # noqa: BLE001 — сбой группы не рушит эксперимент
                failed_hashes.extend(s.run_hash() for s in group)
                continue
            _mark(group)
    elif pending:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=MP_CONTEXT) as pool:
            future_to_group = {pool.submit(compute_group, g): g for g in pending}
            for future in as_completed(future_to_group):
                group = future_to_group[future]
                try:
                    future.result()
                except Exception:  # noqa: BLE001 — сбой группы не рушит эксперимент
                    failed_hashes.extend(s.run_hash() for s in group)
                    continue
                _mark(group)

    return ParallelSummary(
        total=total,
        executed=len(executed_hashes),
        skipped=skipped,
        failed=len(failed_hashes),
        executed_hashes=executed_hashes,
        failed_hashes=failed_hashes,
    )


__all__ = [
    "MP_CONTEXT",
    "ParallelSummary",
    "execute_run_groups_parallel",
    "execute_runs_parallel",
]
