# EXP-6 — CLI: fetch-data / run-experiment

- **Эпик:** E3 — Experiment Engine
- **Статус:** todo
- **Зависимости:** DATA-1, EXP-5

## Описание

Командный интерфейс для загрузки данных и запуска экспериментов.

- `cli/main.py`, `cli/fetch_data.py`, `cli/run_experiment.py`.
- Команды как в разделе 13 плана:
  - `fetch-data --exchange ... --symbol ... --timeframe ... --start ... --end ...`
  - `run-experiment --config configs/experiments/<name>.yaml`

## Критерии приёмки (DoD)

- [ ] `fetch-data` загружает BTCUSDT 1h и сохраняет в Parquet.
- [ ] `run-experiment` прогоняет MA Cross по конфигу и пишет результаты.
- [ ] Smoke-тест обеих команд.
