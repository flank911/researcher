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

## Состояние

Платформа счётно готова и проверена на реальной истории: 450 тестов, `ruff`, `mypy`
зелёные. Первое исследование проведено — BTCUSDT / ETHUSDT / SOLUSDT × 1h / 4h,
2022-01-01 … 2026-10-01, отчёт в
[`notebooks/004_first_research.ipynb`](notebooks/004_first_research.ipynb).

В `data/` лежит 156 060 свечей и 31 362 строки funding за весь диапазон. Свечи и
funding можно наполнять дальше уже сейчас — запись идемпотентна, валидатор на этой
истории не нашёл дефектов. Open interest и long/short ratio тоже грузятся целиком
(`REG-5` закрыт), но Binance отдаёт их лишь за ~30 дней, поэтому их историю создаёт
только регулярный сбор — [OPS-1](tickets/OPS-1.md). Подробно —
[`tickets/README.md`](tickets/README.md).

## Стек

Python 3.11+ · Polars · DuckDB · Parquet · SQLite · pydantic · Typer (CLI).

## Структура

```
configs/      YAML-конфиги экспериментов и источников данных
data/         raw / features / results (вне git)
src/trading_research/
  config.py   конфиг эксперимента (pydantic) + загрузчик YAML
  domain.py   доменные типы (TimeSlice, ExecutionModel, SliceMode)
  hashing.py  канонический хэш и воспроизводимость
  data/       ingestion/, storage/, catalog, schema, validators
  features/   indicators, market_regime, funding/oi/ls_ratio features
  strategies/ base, ma_cross
  backtest/   engine, portfolio, position, orders, risk, fees, slippage,
              sizing, funding, metrics
  experiments/orchestrator, runner, grid/random search, time_splitters,
              cache, parallel, results_store, walk_forward, holdout
  analytics/  queries, stability, correlations, plots
  cli/        main, fetch_data, fetch_market_data, build_features,
              run_experiment, walk_forward, analyze_results, analyze_regimes,
              export_results, plot_heatmap
notebooks/    исследовательские ноутбуки
tests/        тесты
tickets/      план работ (по эпикам E0..E8)
```

## Пайплайн

Полный прогон исследования — восемь шагов, все из корня репозитория в
активированном venv. Числа в комментариях — замеры прогона RUN-7 (Apple Silicon,
14 ядер, конфиг `ma_cross_multi_1h_4h`).

```bash
# 1. OHLCV — 3 символа × 2 таймфрейма (87 с, 156 060 свечей)
for s in BTCUSDT ETHUSDT SOLUSDT; do for tf in 1h 4h; do
  trading-research fetch-data --symbol $s --timeframe $tf \
    --start 2022-01-01 --end 2026-10-01
done; done

# 2. Рыночные ряды. funding доступен за всю историю; OI/LS Binance отдаёт
#    только за последние ~30 дней, поэтому они качаются отдельным окном (REG-4).
#    Неполное покрытие — код возврата 1, а не «wrote N rows» (REG-5).
for s in BTCUSDT ETHUSDT SOLUSDT; do for tf in 1h 4h; do
  trading-research fetch-market-data --symbol $s --timeframe $tf \
    --start 2022-01-01 --end 2026-10-01 --series funding
  for ser in open_interest long_short_ratio; do
    trading-research fetch-market-data --symbol $s --timeframe $tf \
      --start 2026-10-02 --end 2026-10-30 --series $ser
  done
done; done

# 3. Фичи режима по слайсам
trading-research build-features --config configs/experiments/ma_cross_multi_1h_4h.yaml

# 4. Сетка по слайсам (4.7 с на 14 воркерах, resume/dedupe по run_hash)
trading-research run-experiment --config configs/experiments/ma_cross_multi_1h_4h.yaml

# 5. Ранжирование и stability
trading-research analyze-results --experiment ma_cross_multi --metric sharpe --top 6
trading-research export-results --experiment ma_cross_multi --kind stability --metric sharpe

# 6. Карты параметров
trading-research plot-heatmap --experiment ma_cross_multi --x fast_ma --y slow_ma \
  --metric sharpe --agg median

# 7. Честный OOS + untouched holdout (13 с, 270 фолдов)
trading-research walk-forward --config configs/experiments/ma_cross_multi_1h_4h.yaml

# 8. Корреляции параметры ↔ режим
trading-research analyze-regimes --experiment ma_cross_multi --metric sharpe
```

Результаты шага 4 и далее складываются в `data/results/<experiment>/`:
`performance.parquet` (пер-слайс метрики), `catalog.sqlite` (runs + run_metrics),
`wf/<symbol>/<timeframe>/` (выборы по фолдам, OOS equity, `holdout_report.json`).

Шаги 5–8 читают уже посчитанное, поэтому их можно гонять повторно без пересчёта.
Шаг 7 считает всё сам и в кэш результатов не смотрит — это осознанно: honest OOS
не имеет права видеть сетку из шага 4.

## Установка (dev)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Проверки

```bash
ruff check .
ruff format --check .
mypy
pytest --cov=trading_research
```

## CI и код-ревью

Каждый PR в `main` проходит два автоматических пайплайна (см. `.github/workflows/`):

- **Quality gate** (`quality.yml`) — `ruff check`, `ruff format --check`, `mypy`,
  `pytest` с coverage на матрице Python 3.11–3.13. Падение любого шага блокирует merge.
- **Claude review** (`claude-review.yml`) — AI-ревьюер комментирует PR по двум осям:
  *качество кода* (читаемость, дублирование, типобезопасность, тесты, архитектурные
  принципы) и *корректность задачи* (сверка диффа с DoD связанного тикета). ID тикета
  берётся из тела PR или имени ветки, поэтому **указание тикета в PR обязательно**
  (см. [PR template](.github/pull_request_template.md)).

Что требуется для merge в `main` (branch protection):

- зелёный **Quality gate** на всех версиях Python из матрицы;
- пройденное ревью (нет неразрешённых блокеров от Claude review / ревьюеров);
- запрещён прямой push в `main` — только через PR.

Настройка (одноразово, админом репозитория):

- Секрет `ANTHROPIC_API_KEY` в *Settings → Secrets and variables → Actions*
  (нужен для Claude review).
- *Settings → Branches → Add branch ruleset* для `main`: require status checks
  (выбрать джобы `checks` из Quality gate), require pull request before merging,
  block direct pushes.

## План работ

См. [`tickets/README.md`](tickets/README.md) — реестр тикетов по эпикам,
критический путь и порядок дальнейших работ.
MVP: `INF-1..3 → DATA-1..6 → BT-1..8 → EXP-1..6 → AN-1..4`; далее
E5 (walk-forward) → E6 (market regimes) → E7 (первый реальный прогон)
→ E8 (эксплуатация данных).

Текущий фронт: `OPS-1` (ежедневный сбор OI/LS, окно retention необратимо) →
`REG-4`, `DATA-8`, `DATA-7` (проверяемое наполнение базы) → `RUN-9`, `RUN-8`.
