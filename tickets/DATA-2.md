# DATA-2 — Parquet store (партиционирование)

- **Эпик:** E1 — Data Core
- **Статус:** todo
- **Зависимости:** DATA-1

## Описание

Запись/чтение OHLCV в партиционированный Parquet.

- `data/storage/parquet_store.py`.
- Схема путей:
  `raw/candles/exchange=.../market=.../symbol=.../timeframe=.../year=.../month=.../candles.parquet`.
- Идемпотентная дозапись (upsert по времени, без дублей).
- Аналогичные сторы-плейсхолдеры для funding/open_interest/long_short_ratio.

## Критерии приёмки (DoD)

- [ ] Повторная запись того же окна не плодит дубли (round-trip тест).
- [ ] Чтение диапазона, пересекающего несколько партиций, склеивает корректно.
- [ ] Запись сохраняет UTC и типы колонок без потерь.
