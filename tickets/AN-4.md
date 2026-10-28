# AN-4 — Heatmap и плоты

- **Эпик:** E4 — Analytics
- **Статус:** done
- **Зависимости:** AN-1

## Описание

Визуализация карты параметров.

- `analytics/plots.py`: heatmap (X/Y параметры, цвет = median метрика/stability).
- Срезы: по всему периоду / по годам / по месяцам / по символам / по режимам.
- `cli/plot_heatmap` (или подкоманда): `--x ... --y ... --metric ...`.

## Критерии приёмки (DoD)

- [x] Генерится PNG для пары параметров (`x × y`, напр. `fast_ma × slow_ma`) по
      выбранной метрике.
- [x] Поддержаны срезы по году/символу (фильтры `symbol`/`timeframe`/`time_slice`).
- [x] Smoke-тест построения.

## Примечание

- `analytics/plots.py`: `build_heatmap(table, x=, y=, metric=, agg=, symbol=,
  timeframe=, time_slice=)` → `HeatmapData` (оси + 2D-сетка; значения параметров
  берутся из `params_json`); `plot_heatmap(data, out)` рендерит PNG через
  matplotlib (Agg, headless); `heatmap_to_png(...)` — всё в один вызов.
- CLI `plot-heatmap --experiment --x --y --metric [--agg --symbol --timeframe
  --time-slice --out --results-root]` (`cli/plot_heatmap.py`).
- matplotlib добавлен в зависимости; для mypy — override `ignore_missing_imports`.
- Общий x/y: heatmap строится по любой паре ключей параметров (пример `atr_*` —
  иллюстративный для другой стратегии; MVP тестируется на `fast_ma × slow_ma`).
