# RUN-2 — CLI загрузки рынок-данных (funding / OI / LS ratio)

- **Эпик:** E7 — First real research run
- **Статус:** todo
- **Зависимости:** REG-1, EXP-6

## Описание

Команда `fetch-market-data` поверх `BinanceMarketDataClient` (REG-1) и
`ParquetStore.write_series` (DATA-2). Без неё funding/OI/LS невозможно загрузить
иначе как из Python, и весь эпик E6 на реальном прогоне остаётся без данных.

## Контекст

`fetch-data` ([cli/fetch_data.py](../src/trading_research/cli/fetch_data.py))
загружает **только OHLCV** — она жёстко завязана на `KlineFetcher`/`fetch_klines` и
`write_candles`. При этом REG-1 уже реализовал три метода загрузки
(`fetch_funding_rate`, `fetch_open_interest`, `fetch_long_short_ratio`), а DATA-2 —
парную запись/чтение рядов (`write_series` / `read_series`). Связки между ними нет.

Следствие: BT-6 (funding в PnL) на реальном прогоне тоже остаётся без входных
данных — движок принимает `funding=None` и молча считает PnL без funding-платежей.

## Решение

Команда с выбором ряда и переиспользованием диапазона/путей из `fetch-data`:

```
trading-research fetch-market-data --symbol BTCUSDT --series funding \
  --start 2021-01-01 --end 2026-10-01 [--data-root data/raw]
```

- `--series funding|open_interest|long_short_ratio|all`;
- идемпотентная запись через `write_series` (upsert, как у свечей);
- бизнес-логика в `cli/fetch_market_data.py` — тестируется без сети через
  подставной клиент, как сделано для `KlineFetcher`.

## Критерии приёмки (DoD)

- [ ] Три ряда загружаются и читаются обратно через `read_series`.
- [ ] `--series all` загружает все три за один вызов.
- [ ] Повторный запуск идемпотентен (upsert, без дублей).
- [ ] Тесты без сети (подставной клиент), как в `test_cli.py`.
- [ ] Интервалы OI/LS, не поддерживаемые Binance, дают понятную ошибку, а не
      пустой результат.
