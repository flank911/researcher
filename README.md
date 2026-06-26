# trading-research

Исследовательская платформа для поиска и бэктеста оптимальных конфигураций
торговых стратегий **в разрезе временных окон и рыночных режимов**.

Единица исследования — не «бэктест стратегии», а:

```
ExperimentRun = Strategy + StrategyParams + Symbol + Timeframe
              + TimeSlice + DataVersion + ExecutionModel
```

## Принципы

1. `TimeSlice` — сущность первого класса.
2. Raw data immutable.
3. Strategy отделена от Backtest Engine.
4. `StrategyParams` отделены от `ExecutionModel`.
5. Каждый run воспроизводим (детерминированный `run_hash`).
6. Результаты хранятся по каждому временному окну, не агрегатом.
7. Ищем стабильную область параметров, а не переоптимизированный пик.
8. Финальная проверка — только walk-forward.

## Зафиксированные дизайн-решения

- Движок: событийный bar-by-bar.
- Research-режим: непрерывный прогон на всей истории + нарезка equity по `TimeSlice`.
- Первая стратегия: MA Cross.

## Стек

Python 3.11+ · Polars · DuckDB · Parquet · SQLite · pydantic · Typer (CLI).

## Структура

```
configs/      YAML-конфиги экспериментов и источников данных
data/         raw / processed / features / results (вне git)
src/trading_research/
  data/       ingestion, storage, catalog, dataset_ref, validators
  features/   indicators, market_regime, funding/oi/ls features
  strategies/ base, ma_cross, ...
  backtest/   engine, broker, portfolio, orders, position, fees, slippage, metrics
  experiments/config, orchestrator, grid/random/optuna search, time_splitters, runner, cache
  analytics/  queries, stability, correlations, reports, plots
  cli/        main, fetch_data, run_experiment, analyze_results, build_dataset
notebooks/    исследовательские ноутбуки
tests/        тесты
tickets/      план работ (по эпикам E0..E6)
```

## Установка (dev)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Проверки

```bash
ruff check .
mypy
pytest
```

## План работ

См. [`tickets/README.md`](tickets/README.md) — реестр тикетов по эпикам и
критический путь. MVP: `INF-1..3 → DATA-1..6 → BT-1..8 → EXP-1..6 → AN-1..4`.
