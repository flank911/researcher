# BT-6 — Funding в PnL

- **Эпик:** E2 — Backtest Core
- **Статус:** todo
- **Зависимости:** BT-4, DATA-2 (funding)
- **Реализует улучшение:** №10 (funding для futures)

## Описание

Учёт funding-платежей в PnL для futures.

- Применение funding rate к открытым позициям в моменты funding-расчётов.
- Корректный знак для long/short.
- Funding как издержка/доход, отдельно учитываемая в equity.

## Критерии приёмки (DoD)

- [ ] На датасете с funding итоговый PnL отличается от безфандингового на
      ожидаемую величину (юнит-тест).
- [ ] Знак funding корректен для long и short.
- [ ] Отсутствие funding-данных не ломает прогон (graceful fallback).
