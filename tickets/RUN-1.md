# RUN-1 — Подключить resume/dedupe и параллелизм к CLI-прогону

- **Эпик:** E7 — First real research run
- **Статус:** todo
- **Зависимости:** EXP-3, EXP-4, EXP-5, EXP-6
- **Критический путь:** да

## Описание

`RunCache` (EXP-3), `run_parallel` (EXP-5) и `ResultsStore` (EXP-4) существуют,
покрыты тестами, но **не подключены ни к одному продакшн-пути**. Подключить их к
`cli/run_experiment.py`, чтобы реальный прогон был возобновляемым и многопроцессным.

## Контекст

`grep` по `src/` показывает: вне собственных модулей `experiments/cache.py`,
`experiments/parallel.py` и `experiments/results_store.py` не импортируются нигде,
кроме единственной ссылки `queries.performance_from_store`.

Фактический путь команды `run-experiment`
(`run_experiment_from_config`, [cli/run_experiment.py](../src/trading_research/cli/run_experiment.py)):

```
config → DuckDBStore.read_window → expand_grid
       → orchestrator.run_experiment   (последовательно, без кэша)
       → build_parameter_performance   (из records в памяти)
       → save_performance_table        (performance.parquet напрямую)
```

То есть реальный многочасовой прогон сегодня:

- **однопроцессный** — `parallel.py` не задействован, хотя EXP-5 закрыт;
- **без resume** — падение на середине сетки теряет весь прогон, `run_hash` не
  считается, dedupe не работает;
- **в обход каталога** — `ResultsStore` не наполняется, артефакты ранов (trades,
  equity curve) не сохраняются, хотя `output.save_trades` и
  `output.save_equity_curve` в конфиге стоят в `true` и сейчас ни на что не влияют.

Последнее особенно важно: три закрытых тикета (EXP-3/4/5) дают нулевой эффект на
реальном прогоне, и обнаруживается это только при попытке его сделать.

## Решение

- `run_experiment_from_config` считает `run_hash` на комбинацию и спрашивает
  `RunCache` перед прогоном (resume + dedupe).
- Исполнение сетки — через `run_parallel` с опцией `--workers` (дефолт:
  `os.cpu_count()`, `1` — последовательно, для отладки).
- Результаты пишутся в `ResultsStore`, таблица AN-1 строится через
  `performance_from_store`; `output.save_trades` / `save_equity_curve` начинают
  реально управлять сохранением артефактов.
- Флаг `--resume/--no-resume` в CLI, прогресс в stderr (количество
  выполнено/из кэша/осталось).

## Критерии приёмки (DoD)

- [ ] `run-experiment` переиспользует результаты из кэша при повторном запуске
      (тест: два прогона, второй не вызывает движок).
- [ ] Прерванный прогон доходит до конца при повторном запуске с `--resume`.
- [ ] `--workers N` даёт тот же результат, что `--workers 1` (детерминизм
      не зависит от числа процессов).
- [ ] `ResultsStore` наполняется, `output.save_trades`/`save_equity_curve`
      управляют артефактами.
- [ ] Таблица AN-1 строится через `performance_from_store` и совпадает с прежней
      по содержимому (регрессионный тест на эквивалентность).
