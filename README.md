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
- Вторая стратегия: возврат к среднему на RSI (`RES-1`) — взята не для
  ассортимента, а как разделяющий эксперимент: трендследящая и
  контртрендовая логики не могут ошибаться одинаково, поэтому их совпадающий
  вывод указывает не на стратегию, а на платформу.

## Состояние

Платформа счётно готова и проверена на реальной истории: 591 тест, `ruff`, `mypy`
зелёные. Первое исследование проведено — BTCUSDT / ETHUSDT / SOLUSDT × 1h / 4h,
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
  strategies/ base, ma_cross, rsi_reversion
  backtest/   engine, portfolio, position, orders, risk, fees, slippage,
              sizing, funding, metrics
  experiments/orchestrator, runner, grid/random search, time_splitters,
              cache, parallel, results_store, walk_forward, holdout
  analytics/  queries, stability, correlations, plots
  admin/      состояние платформы как данные (read-model) и веб-страницы
              только для чтения (templates/)
  cli/        main, fetch_data, fetch_market_data, update_data,
              update_market_data, validate_data, build_features,
              run_experiment, walk_forward, status, admin,
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
#    воркеров не зависит (RUN-8). Прогрев окна задаёт стратегия, а не конфиг
#    (WF-4): warmup_bars в секции walk_forward — нижняя граница, запас сверху.
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

Командой `status` (`ADM-2`) — она читает store и лог, не обращаясь к API:

```bash
trading-research status --config configs/experiments/ma_cross_multi_1h_4h.yaml
```

Вердикт о сборе дают сами данные: если OI/LS свежи, сбор работает, а блок
«Расписание» объясняет, когда была последняя итерация и чем она кончилась. Повод
вмешаться команда называет вслух — в блоке «Внимание» и в итоговой строке:

```
Расписание
  последняя итерация 2026-11-03 13:54, 5 ч 51 мин назад, exit=0
Внимание
  данные: BTCUSDT 1h candles — отстаёт на 84 периода (3 суток 12 ч)
  валидация: вердикт без даты у 24 отчётов: покрытие текущих данных не определить
итого: требуется вмешательство — 7 поводов (валидация, данные)
```

`--strict` даёт код возврата 1, когда вмешательство требуется — так команду можно
ставить перед прогоном; `--json` отдаёт то же состояние машинно.

### То же состояние, но глазами

Там, где `status` отвечает «всё ли живо», а углубиться не даёт, есть локальная
админка (`ADM-3`) — две страницы, обе только для чтения:

```bash
pip install -e ".[admin]"      # веб-зависимости в отдельной экстре
trading-research admin --config configs/experiments/ma_cross_multi_1h_4h.yaml
# админка (только чтение) -> http://127.0.0.1:8787
```

«Обзор» повторяет блоки `status` и добавляет таблицы рядов и walk-forward,
«Данные» показывает каждую пару по рядам: покрытие, отставание, запас retention,
когда проверяли и вердикт. `/api/status` отдаёт то же, что `status --json`.

Админка слушает только `127.0.0.1`, аутентификации у неё нет, и флага `--host`
нет намеренно: доступ с другой машины — это SSH-туннель, то есть осознанное
решение. Менять она не умеет ничего — ни одного `POST`; запуск шагов пайплайна —
отдельный тикет (`ADM-6`).

Отставание OI/LS на часы поводом не считается: их создаёт расписание раз в шесть
часов, и на 1h они отстают по построению. Поводы у этих рядов другие — запас до
потери истории меньше недели и тишина в логе дольше суток.

Починить, а не проверить, — это по-прежнему запуск сбора руками:

```bash
trading-research update-market-data --config configs/experiments/ma_cross_multi_1h_4h.yaml
```

Для разбора причины остаётся `launchctl list com.trading-research.update-market-data`
(смотрят `LastExitStatus`) и `tail -20 data/logs/update-market-data.log`.

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

На каждый PR в `main` работает один автоматический пайплайн
(см. `.github/workflows/`):

- **Quality gate** (`quality.yml`) — `ruff check`, `ruff format --check`, `mypy`,
  `pytest` с coverage на матрице Python 3.11–3.13.

**Автоматического AI-ревью нет.** Воркфлоу `claude-review.yml` удалён в
[INF-9](tickets/INF-9.md): он лежал выключенным, не давал ни чека, ни
предупреждения, и 14 PR подряд прошли без ревью так, что заметить это можно было
только через `gh workflow list`. Ревью на PR — ручное. Указание тикета в теле PR
остаётся обязательным (см. [PR template](.github/pull_request_template.md)): по нему
сверяется DoD — теперь человеком, а не воркфлоу.

### Чего в репозитории не настроено

Branch protection для `main` **не включена**: прямой push не запрещён, и красный
Quality gate merge не блокирует. Это не теория — именно так красный py3.11 прожил три
мержа (#39 → #41, см. [INF-8](tickets/INF-8.md)). Что нужно сделать владельцу
репозитория: *Settings → Branches → Add branch ruleset* для `main` — require status
checks (выбрать джобы из Quality gate), require pull request before merging, block
direct pushes.

## План работ

См. [`tickets/README.md`](tickets/README.md) — реестр тикетов по эпикам,
критический путь и порядок дальнейших работ.
MVP: `INF-1..3 → DATA-1..6 → BT-1..8 → EXP-1..6 → AN-1..4`; далее
E5 (walk-forward) → E6 (market regimes) → E7 (первый реальный прогон)
→ E8 (эксплуатация данных) → E9 (расширение исследования).

Текущий фронт — два эпика. E9 «Расширение исследования»: `RES-1` закрыт (вторая
стратегия как разделяющий эксперимент), остались `RES-2` сетка в сотни комбинаций и
`RES-3` второй сквозной прогон. E10 «Админка»: закрыты `ADM-1` (состояние платформы
как данные), `ADM-2` (команда `status`) и `ADM-3` (веб-страницы только для чтения),
дальше страницы прогонов и walk-forward, а контроль — после них.
Первые девять эпиков закрыты полностью — все 53 тикета.
