# BT-6 — Funding в PnL

- **Эпик:** E2 — Backtest Core
- **Статус:** done
- **Зависимости:** BT-4, DATA-2 (funding)
- **Реализует улучшение:** №10 (funding для futures)

## Описание

Учёт funding-платежей в PnL для futures.

- Применение funding rate к открытым позициям в моменты funding-расчётов.
- Корректный знак для long/short.
- Funding как издержка/доход, отдельно учитываемая в equity.

## Критерии приёмки (DoD)

- [x] На датасете с funding итоговый PnL отличается от безфандингового на
      ожидаемую величину (юнит-тест).
- [x] Знак funding корректен для long и short.
- [x] Отсутствие funding-данных не ломает прогон (graceful fallback).

## Примечание

- `backtest/funding.py` (`bucket_funding`): раскладывает ряд funding-ставок по
  барам (ставка → содержащий её бар; несколько ставок на бар суммируются).
- `Portfolio.apply_funding(rate, mark_price)`: знак биржи (rate>0 → long платит,
  short получает); влияет на баланс, трекается в `funding_total` и `Trade.funding`.
- `BacktestEngine.run(signals, execution, funding=None)`: funding применяется на
  границе бара (по `open` как цене момента расчёта), до внутрибарных выходов;
  результат отдаёт `BacktestResult.funding_total`. `funding=None`/пусто → прогон
  без funding (fallback).
- Входной ряд funding (колонки `open_time`, `funding_rate`) может загружаться из
  `ParquetStore.read_series("funding", ...)` (DATA-2); движок принимает готовый
  DataFrame, не завязываясь на сторадж.
