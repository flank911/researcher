# AN-1 — Таблица parameter_performance_by_slice

- **Эпик:** E4 — Analytics
- **Статус:** done
- **Зависимости:** EXP-4
- **Реализует улучшение:** №17 (центральная аналитическая таблица)

## Описание

Материализация центральной таблицы анализа (раздел 17 плана).

- `analytics/queries.py`: DuckDB-вью / Parquet с колонками experiment_id,
  strategy_name, symbol, timeframe, time_slice_*, param_hash, params_json,
  метрики (return/dd/sharpe/...), и слотами под фичи режима
  (market_return_pct, volatility, trend_strength, funding_avg, oi_change_pct,
  ls_ratio_avg, regime_label).
- Возможность среза `params ↔ time / volatility / trend / funding / OI / symbol / tf`.

## Критерии приёмки (DoD)

- [x] Таблица строится из результатов экспериментов.
- [x] Запросы `params↔time`, `params↔symbol`, `params↔tf` возвращают данные.
- [x] Тест на соответствие схеме.

## Примечание

- `analytics/queries.py`: каноническая `PERFORMANCE_SCHEMA` (experiment_id,
  strategy_name, symbol, timeframe, time_slice_*, param_hash, params_json, 13
  метрик + 7 null-слотов режима market_return_pct/volatility/trend_strength/
  funding_avg/oi_change_pct/ls_ratio_avg/regime_label).
- `build_parameter_performance(records)` — из `PerformanceRecord`;
  `performance_from_store(store, slices, experiment_id=)` — из каталога
  `ResultsStore` (EXP-4; добавлен `all_runs()`).
- Срезы `params ↔ time/symbol/timeframe` через DuckDB-группировку по
  `(param_hash × ось)`: `params_by_time/symbol/timeframe` (+ общий
  `slice_param_performance(axis=, metric=)`; оси/метрики из whitelist).
- Слоты режима — плейсхолдеры под E6/REG; `param_hash` стабилен (INF-3).
