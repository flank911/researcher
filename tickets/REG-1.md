# REG-1 — Доп. данные: funding/OI/LS ratio

- **Эпик:** E6 — Market Regimes
- **Статус:** done
- **Зависимости:** DATA-2

## Описание

Загрузка и хранение внешних рыночных данных.

- Загрузчики funding rate, open interest, long/short ratio
  (`data/ingestion/...`).
- Parquet-сторы по аналогии с OHLCV.
- Валидация (повторяет подход DATA-3).

## Критерии приёмки (DoD)

- [x] Данные funding/OI/LS грузятся и сохраняются в Parquet.
- [x] Валидация находит пропуски/дубликаты.
- [x] Интеграционные тесты с мок-ответами.

## Примечание

- `data/ingestion/market_data_client.py` (`BinanceMarketDataClient`):
  `fetch_funding_rate` / `fetch_open_interest` / `fetch_long_short_ratio` — по
  образцу DATA-1 (инжектируемый `Transport`, retry на 418/429/5xx, пагинация по
  времени, нормализация в UTC-мс с ключом `open_time`).
- Канонические схемы рядов в `data/schema.py` (`FUNDING_SCHEMA`,
  `OPEN_INTEREST_SCHEMA`, `LONG_SHORT_RATIO_SCHEMA`); хранение — через
  `ParquetStore.write_series`/`read_series` (DATA-2), идемпотентно.
- Валидация: `validators.validate_series(df, timeframe=, value_columns=,
  non_negative_columns=, ...)` — подход DATA-3 (пропуски по шагу, дубликаты,
  монотонность) + NaN/inf и отрицательные значения (funding может быть
  отрицательным — не кладём его в `non_negative_columns`).
- Тесты — на мок-транспорте (без сети): канонизация, round-trip Parquet, retry
  на 429, поиск пропуска/дубликата/аномалий.
