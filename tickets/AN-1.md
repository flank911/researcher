# AN-1 — Таблица parameter_performance_by_slice

- **Эпик:** E4 — Analytics
- **Статус:** todo
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

- [ ] Таблица строится из результатов экспериментов.
- [ ] Запросы `params↔time`, `params↔symbol`, `params↔tf` возвращают данные.
- [ ] Тест на соответствие схеме.
