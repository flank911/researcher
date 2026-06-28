# BT-4 — Fees, slippage, sizing

- **Эпик:** E2 — Backtest Core
- **Статус:** done
- **Зависимости:** BT-2

## Описание

Модели издержек и размера позиции.

- `backtest/fees.py`: maker/taker комиссии.
- `backtest/slippage.py`: модель проскальзывания (`slippage_pct`).
- Sizing: `fixed` / `percent_balance` / `risk_based`; учёт `leverage`.

## Критерии приёмки (DoD)

- [x] Комиссии и слиппедж корректно уменьшают PnL (юнит-тесты).
- [x] Каждый режим sizing даёт ожидаемый размер позиции на тесте
      (`fixed`, `percent_balance`; `risk_based` отложен в BT-5 — требует дистанции до стопа).
- [x] Leverage корректно масштабирует экспозицию.

## Примечание

- Исполнения рыночные (taker) → единая `commission_rate`; разделение maker/taker
  потребует лимитных ордеров (вне scope BT-4).
- `risk_based` sizing зависит от риск-слоя (стоп-дистанция) и реализуется в BT-5.
