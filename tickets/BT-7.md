# BT-7 — MetricsCalculator

- **Эпик:** E2 — Backtest Core
- **Статус:** done
- **Зависимости:** BT-2
- **Реализует улучшение:** №6 (надёжные метрики на коротких окнах)

## Описание

Расчёт метрик результата прогона.

- `backtest/metrics.py`: total_return_pct, max_drawdown_pct, Sharpe, Sortino,
  profit_factor, win_rate, trades_count, avg_trade_pct, expectancy, recovery_factor.
- **Явный** annualization factor по timeframe; **явный** ряд доходностей
  (bar-returns vs trade-returns — задокументировать выбор).
- Фильтр `min_trades`: метрики, ненадёжные на малой выборке, помечать/занулять.

## Критерии приёмки (DoD)

- [x] Метрики совпадают с эталонным ручным расчётом на фикстуре.
- [x] Sharpe сравним между разными timeframes (корректный annualization).
- [x] При `trades_count < min_trades` метрики помечаются как ненадёжные.
- [x] `test_metrics.py` покрывает граничные случаи (0 сделок, без просадки).
