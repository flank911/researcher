# BT-4 — Fees, slippage, sizing

- **Эпик:** E2 — Backtest Core
- **Статус:** todo
- **Зависимости:** BT-2

## Описание

Модели издержек и размера позиции.

- `backtest/fees.py`: maker/taker комиссии.
- `backtest/slippage.py`: модель проскальзывания (`slippage_pct`).
- Sizing: `fixed` / `percent_balance` / `risk_based`; учёт `leverage`.

## Критерии приёмки (DoD)

- [ ] Комиссии и слиппедж корректно уменьшают PnL (юнит-тесты).
- [ ] Каждый режим sizing даёт ожидаемый размер позиции на тесте.
- [ ] Leverage корректно масштабирует экспозицию.
