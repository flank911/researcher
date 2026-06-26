# BT-1 — Strategy Protocol + MA Cross

- **Эпик:** E2 — Backtest Core
- **Статус:** done
- **Зависимости:** INF-2

## Описание

Чистый интерфейс стратегии и первая реализация.

- `strategies/base.py`: Protocol `Strategy` с `name` и
  `generate_signals(data, params) -> signals` (без I/O, без знания о комиссиях/
  размере позиции/сохранении).
- `strategies/ma_cross.py`: MA Cross на Polars (fast/slow MA, long/exit сигналы).
- Стратегия объявляет требуемый `warmup_bars` (макс. период индикатора).

## Критерии приёмки (DoD)

- [ ] Сигналы детерминированы и не используют будущее внутри `generate_signals`.
- [ ] `test_strategy_signals.py`: на синтетике пересечения MA дают ожидаемые сигналы.
- [ ] Стратегия не выполняет никакого I/O.
