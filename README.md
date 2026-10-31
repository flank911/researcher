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

Платформа счётно готова и проверена на реальной истории: 558 тестов, `ruff`, `mypy`
зелёные локально. CI на `main` красный по py3.11 — один тест сверяет `sharpe`
побитово с эталоном, снятым на py3.13 (`INF-8`); продукт это не затрагивает.
Первое исследование проведено — BTCUSDT / ETHUSDT / SOLUSDT × 1h / 4h,
2022-01-01 … 2026-10-01, отчёт в
[`notebooks/004_first_research.ipynb`](notebooks/004_first_research.ipynb).

В `data/` лежит 158 790 свечей и 31 902 строки funding за весь диапазон. Свечи и
funding наполняются дальше одной командой: `update-data` (`OPS-2`) догружает
только хвост от последнего сохранённого бара, а качество проверяется из CLI
командой `validate-data` (`DATA-8`): на всей загруженной истории дефектов нет. Open interest и long/short ratio грузятся целиком
(`REG-5` закрыт), а поскольку Binance отдаёт их лишь за 30 дней, их историю
накапливает регулярный сбор — см. «Регулярный сбор рыночных рядов». Подробно —
[`tickets/README.md`](tickets/README.md).

## Стек

Python 3.11+ · Polars · DuckDB · Parquet · SQLite · pydantic · Typer (CLI).

## Структура

```
configs/      YAML-конфиги экспериментов и источников данных
data/         raw / features / results / reports (вне git)
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
  cli/        main, fetch_data, fetch_market_data, update_data,
              update_market_data, validate_data, build_features,
              run_experiment, walk_forward,
              analyze_results, analyze_regimes, export_results, plot_heatmap,
              text (общее форматирование вывода)
notebooks/    исследовательские ноутбуки
ops/          скрипт и launchd-агент регулярного сбора (OPS-1)
tests/        тесты
tickets/      план работ (по эпикам E0..E8)
```

## Пайплайн

Полный прогон исследования — десять шагов, все из корня репозитория в
активированном venv. Числа в комментариях — замеры на Apple Silicon, 14 ядер,
конфиг `ma_cross_multi_1h_4h` (прогон RUN-7; шаг 9 — после ускорения RUN-8).

```bash
# 1. OHLCV — 3 символа × 2 таймфрейма (87 с, 156 060 свечей). Это первая
#    загрузка; добивать базу до сегодня — шагом 2.
for s in BTCUSDT ETHUSDT SOLUSDT; do for tf in 1h 4h; do
  trading-research fetch-data --symbol $s --timeframe $tf \
    --start 2022-01-01 --end 2026-10-01
done; done

# 2. Догрузка свечей от последнего сохранённого бара до текущего (OPS-2).
#    Запрашивается только хвост: месяц по 6 парам — 4,4 с против 87 с полной
#    перекачки; повторный прогон без новых данных в API не ходит вовсе.
trading-research update-data --config configs/experiments/ma_cross_multi_1h_4h.yaml

# 3. Рыночные ряды. funding доступен за всю историю; OI/LS Binance отдаёт
#    только за последние 30 дней: диапазон за их пределами урезается с явным
#    сообщением (REG-4), а неполное покрытие даёт код возврата 1 (REG-5).
#    Историю OI/LS накапливает регулярный сбор — см. раздел ниже.
for s in BTCUSDT ETHUSDT SOLUSDT; do for tf in 1h 4h; do
  trading-research fetch-market-data --symbol $s --timeframe $tf \
    --start 2022-01-01 --end 2026-10-01 --series funding
  for ser in open_interest long_short_ratio; do
    trading-research fetch-market-data --symbol $s --timeframe $tf \
      --start 2026-10-02 --end 2026-10-30 --series $ser
  done
done; done

# 4. Проверка качества загруженного: пропуски, дубли, монотонность, OHLC,
#    выбросы — по свечам и по рядам funding/OI/LS. JSON-отчёты в data/reports/.
#    --strict даёт код 1 при дефектах: так шаг можно ставить перед прогоном.
trading-research validate-data --config configs/experiments/ma_cross_multi_1h_4h.yaml \
  --strict

# 5. Фичи режима по слайсам
trading-research build-features --config configs/experiments/ma_cross_multi_1h_4h.yaml

# 6. Сетка по слайсам (4.7 с на 14 воркерах, resume/dedupe по run_hash)
trading-research run-experiment --config configs/experiments/ma_cross_multi_1h_4h.yaml

# 7. Ранжирование и stability
trading-research analyze-results --experiment ma_cross_multi --metric sharpe --top 6
trading-research export-results --experiment ma_cross_multi --kind stability --metric sharpe

# 8. Карты параметров
trading-research plot-heatmap --experiment ma_cross_multi --x fast_ma --y slow_ma \
  --metric sharpe --agg median

# 9. Честный OOS + untouched holdout (3.3 с на 8 воркерах, 270 фолдов). По
#    каждой паре — строка holdout: research vs резерв, деградация, сделки и
#    пометка ненадёжного результата; отсутствие holdout объяснено, а не
#    умолчано (RUN-9). --workers 0 берёт число ядер; результат от числа
#    воркеров не зависит (RUN-8).
trading-research walk-forward --config configs/experiments/ma_cross_multi_1h_4h.yaml \
  --workers 8

# 10. Корреляции параметры ↔ режим
trading-research analyze-regimes --experiment ma_cross_multi --metric sharpe
```

Результаты шага 6 и далее складываются в `data/results/<experiment>/`:
`performance.parquet` (пер-слайс метрики), `catalog.sqlite` (runs + run_metrics),
`wf/<symbol>/<timeframe>/` (выборы по фолдам, OOS equity, `holdout_report.json`).

Шаги 6–9 читают уже посчитанное, поэтому их можно гонять повторно без пересчёта.
Шаг 8 считает всё сам и в кэш результатов не смотрит — это осознанно: honest OOS
не имеет права видеть сетку из шага 6.

## Регулярный сбор рыночных рядов

Свечи и funding доступны у Binance за всю историю — их можно догрузить когда
угодно (`update-data` для свечей, `update-market-data --series funding`). Open interest и long/short ratio биржа отдаёт **только за последние
30 дней** (замерено: 29 дней назад эндпоинт отвечает, 30 — уже `-1130`). Их
история поэтому не загружается, а *создаётся* регулярным сбором: каждый день без
сбора — день, которого в датасете не будет никогда.

Одна итерация — догрузка всех рядов от последней сохранённой точки:

```bash
trading-research update-market-data --config configs/experiments/ma_cross_multi_1h_4h.yaml
```

Команда берёт набор пар из конфига (менять состав сбора правкой конфига, а не
расписания), пишет идемпотентно и печатает строку на каждую пару. Невосполнимый
пропуск и неполное покрытие дают код возврата 1 — чтобы провал итерации не
выглядел успехом.

### Поставить по расписанию (launchd, macOS)

Агент уже загружен в этой системе (2026-10-30). Команды ниже — для переустановки
или для другой машины.

```bash
cp ops/com.trading-research.update-market-data.plist ~/Library/LaunchAgents/
```

```bash
launchctl load ~/Library/LaunchAgents/com.trading-research.update-market-data.plist
```

Интервал — 6 часов (`StartInterval`, а не календарное время: интервальную задачу
launchd запускает при пробуждении ноутбука, календарную можно проспать). При окне
биржи в 30 дней это большой запас — чтобы потерять данные, расписание должно
молчать месяц.

Агент собирает только рыночные ряды, потому что срочны именно они. Свечи по
расписанию не нужны: они доступны за всю историю, и `update-data` догонит любой
пропуск в любой момент. Если захочется и их — та же команда в том же plist.

### Как проверить, что расписание живо

Главная проверка — не лог, а сами данные. Запустите команду руками:

```bash
trading-research update-market-data --config configs/experiments/ma_cross_multi_1h_4h.yaml
```

Если сбор работает, OI и LS отвечают «актуален до …, догружать нечего». Если
расписание умерло, та же команда покажет разрыв и дописанные точки за всё
пропущенное время — а при молчании дольше 30 дней ещё и `НЕВОСПОЛНИМЫЙ ПРОПУСК`.

Остальное — для разбора причины:

```bash
launchctl list com.trading-research.update-market-data
```

```bash
tail -20 data/logs/update-market-data.log
```

У `launchctl list` смотрят `LastExitStatus`; в логе каждая итерация заканчивается
строкой `--- exit=…`.

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
→ E8 (эксплуатация данных) → E9 (расширение исследования).

Текущий фронт: `INF-8` (вернуть CI в зелёное) → `INF-9` (автоматическое ревью
выключено) → эпик E9 «Расширение исследования»: `RES-1` вторая стратегия как
разделяющий эксперимент, `RES-2` сетка в сотни комбинаций, `RES-3` второй сквозной
прогон. Первые девять эпиков закрыты полностью — все 53 тикета.
