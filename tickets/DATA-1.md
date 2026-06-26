# DATA-1 — Клиент загрузки OHLCV (Binance futures)

- **Эпик:** E1 — Data Core
- **Статус:** done
- **Зависимости:** INF-1

## Описание

Клиент загрузки исторических свечей с Binance USDT-M futures.

- `data/ingestion/binance_client.py`: пагинация по времени, обработка rate-limit,
  retry с backoff.
- Нормализация в единую схему: UTC, timestamp в ms, явная конвенция времени бара
  (open_time как ключ), типы колонок (open/high/low/close/volume + quote_volume,
  trades, taker_base/quote).
- Параметры: exchange, market, symbol, timeframe, start, end.

## Критерии приёмки (DoD)

- [ ] Загрузка BTCUSDT 1h за заданный диапазон возвращает непрерывный ряд.
- [ ] Интеграционный тест с мок-ответом API.
- [ ] Корректная обработка пустого диапазона / периода до листинга.
- [ ] Rate-limit и retry покрыты тестом (мок 429/5xx).
