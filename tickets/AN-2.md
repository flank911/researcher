# AN-2 — Ранжирование лучших параметров по слайсу

- **Эпик:** E4 — Analytics
- **Статус:** done
- **Зависимости:** AN-1

## Описание

Запросы и CLI для топ-параметров.

- `analytics/queries.py`: top-N по метрике в разрезе `time_slice` / symbol / tf.
- `cli/analyze_results.py`: `analyze-results --experiment ... --metric ... --top N`.

## Критерии приёмки (DoD)

- [x] CLI выдаёт top-N по заданной метрике (`analyze-results --metric --top`).
- [x] Корректная сортировка и фильтрация по символу/таймфрейму.
- [x] Тест на примере результатов.

## Примечание

- `analytics/queries.py` → `rank_params(table, metric=, top=, by=, symbol=,
  timeframe=, descending=)`: глобальный top-N или top-N на группу (`by` =
  time/symbol/timeframe); фильтры symbol/timeframe; усреднение метрики по
  `(param_hash × ось)`, ранг 1 — лучший. Плюс `save/load_performance_table`.
- `cli/analyze_results.py` + команда `analyze-results --experiment --metric --top
  [--by --symbol --timeframe --results-root]`; читает таблицу эксперимента.
- Пайплайн связан: `run-experiment` теперь пишет каноническую таблицу AN-1
  (`{results_root}/{name}/performance.parquet`), которую читает `analyze-results`.
- `stability_score` (AN-3) подключится как ещё одна метрика/колонка той же таблицы.
