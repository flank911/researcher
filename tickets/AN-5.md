# AN-5 — Экспорт CSV/Parquet + ноутбук

- **Эпик:** E4 — Analytics
- **Статус:** done
- **Зависимости:** AN-1

## Описание

Экспорт результатов и исследовательский ноутбук.

- Экспорт сводных таблиц в CSV/Parquet.
- `notebooks/001_explore_results.ipynb`: открытие результатов и базовые графики.

## Критерии приёмки (DoD)

- [x] Экспорт-команда создаёт корректный CSV/Parquet.
- [x] Ноутбук открывает результаты и строит базовые графики без правок кода.

## Примечание

- `cli/export_results.py` + команда `export-results --experiment --kind
  (performance/ranking/stability) --format (csv/parquet) [--metric --top --out
  --results-root]`.
- `notebooks/001_explore_results.ipynb`: загружает таблицу из `TR_RESULTS`
  (если задан) либо использует встроенную синтетику → исполняется сверху донизу
  без правок; показывает рейтинг (AN-2), стабильность (AN-3), heatmap (AN-4).
- Тест `test_notebook.py` исполняет код-ячейки ноутбука (на синтетике) и
  проверяет построение PNG; `nbformat` добавлен в dev-deps.
