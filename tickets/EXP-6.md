# EXP-6 — CLI: fetch-data / run-experiment

- **Эпик:** E3 — Experiment Engine
- **Статус:** done
- **Зависимости:** DATA-1, EXP-5

## Описание

Командный интерфейс для загрузки данных и запуска экспериментов.

- `cli/main.py`, `cli/fetch_data.py`, `cli/run_experiment.py`.
- Команды как в разделе 13 плана:
  - `fetch-data --exchange ... --symbol ... --timeframe ... --start ... --end ...`
  - `run-experiment --config configs/experiments/<name>.yaml`

## Критерии приёмки (DoD)

- [x] `fetch-data` загружает BTCUSDT 1h и сохраняет в Parquet.
- [x] `run-experiment` прогоняет MA Cross по конфигу и пишет результаты.
- [x] Smoke-тест обеих команд.

## Примечание

- `cli/main.py` — Typer-приложение с командами `fetch-data` и `run-experiment`;
  бизнес-логика вынесена в `cli/fetch_data.py` и `cli/run_experiment.py`
  (тестируется без CLI и без сети через подставной `KlineFetcher`).
- `fetch-data`: Binance-клиент (DATA-1) → `ParquetStore` (DATA-2).
- `run-experiment`: конфиг (INF-2) → данные из `DuckDBStore` (DATA-5) → стратегия
  (BT-1) → сетка (EXP-1) → слайсы (DATA-6) → `run_experiment` (EXP-2); пишет
  таблицу пер-слайс метрик `{results_root}/{name}/metrics.parquet` (вход для AN-1).
- Для CLI добавлен per-file ignore `B008` (Typer требует вызовов в дефолтах).
