# Tickets — Trading Research Platform

Исследовательская платформа для поиска и бэктеста оптимальных конфигураций
торговых стратегий в разрезе временных окон и рыночных режимов.

## Зафиксированные дизайн-решения

- **Движок:** событийный bar-by-bar (корректно для trailing/TP/SL/Chandelier).
- **Research-режим метрик:** непрерывный прогон на всей истории, нарезка equity по `TimeSlice`
  (баланс переносится между окнами). Решает проблему прогрева индикаторов.
- **Первая стратегия MVP:** MA Cross.
- **Стек:** Python + Polars + DuckDB + Parquet + SQLite + YAML/pydantic configs.

## Критический путь (нельзя срезать)

`INF-3` (стабильный хэш) → `BT-3` (fill timing) → `BT-5` (intrabar/ликвидация) →
`BT-8` (golden test) → `EXP-2` (warmup + нарезка). Тихая ошибка в любом из них
обесценивает всю последующую аналитику.

## MVP-граница

`INF-1..3 → DATA-1..6 → BT-1..8 → EXP-1..6 → AN-1..4`.
После MVP: E5 (walk-forward) → E6 (market regimes) → E7 (первый реальный прогон).

## Статус: E0–E6 закрыты

Код платформы готов и покрыт тестами (312 тестов), но **ни одного прогона на
реальных данных не делалось**. E7 закрывает разрыв между «библиотека работает» и
«исследование проведено». Главное, что вскрылось при подготовке E7: три закрытых
тикета (`EXP-3` resume/dedupe, `EXP-4` ResultsStore, `EXP-5` параллелизм) и два
эпика (`E5`, `E6`) существуют только как библиотеки — в CLI-пути они не подключены,
см. [RUN-1](RUN-1.md), [RUN-4](RUN-4.md), [RUN-5](RUN-5.md).

## Легенда статусов

`todo` · `in_progress` · `done` · `blocked`

## Реестр

### E0 — Scaffolding & инфраструктура
| ID | Заголовок | Статус | Зависимости |
|----|-----------|--------|-------------|
| [INF-1](INF-1.md) | Каркас проекта и тулинг | done | — |
| [INF-2](INF-2.md) | Доменные типы и конфиги (pydantic) | done | INF-1 |
| [INF-3](INF-3.md) | Канонический хэш и воспроизводимость | done | INF-1 |
| [INF-4](INF-4.md) | CI-пайплайн для код-ревью merge requests | done | INF-1 |
| [INF-5](INF-5.md) | Детерминизм канонизации (set/bytes/объекты) | done | INF-3 |
| [INF-6](INF-6.md) | Совместимость с duckdb 2.0 | done | DATA-5, AN-1 |

### E1 — Data Core
| ID | Заголовок | Статус | Зависимости |
|----|-----------|--------|-------------|
| [DATA-1](DATA-1.md) | Клиент загрузки OHLCV (Binance futures) | done | INF-1 |
| [DATA-2](DATA-2.md) | Parquet store (партиционирование) | done | DATA-1 |
| [DATA-3](DATA-3.md) | Валидация данных | done | DATA-2 |
| [DATA-4](DATA-4.md) | Каталог датасетов + dataset_hash | done | DATA-2, INF-3 |
| [DATA-5](DATA-5.md) | DuckDB/Polars слой доступа (+warmup) | done | DATA-2 |
| [DATA-6](DATA-6.md) | Генератор TimeSlice | done | INF-2 |

### E2 — Backtest Core
| ID | Заголовок | Статус | Зависимости |
|----|-----------|--------|-------------|
| [BT-1](BT-1.md) | Strategy Protocol + MA Cross | done | INF-2 |
| [BT-2](BT-2.md) | Событийный движок: позиции/ордера | done | BT-1 |
| [BT-3](BT-3.md) | Fill timing / анти-look-ahead | done | BT-2 |
| [BT-4](BT-4.md) | Fees, slippage, sizing | done | BT-2 |
| [BT-5](BT-5.md) | Risk: маржа/ликвидация, intrabar TP/SL | done | BT-4 |
| [BT-6](BT-6.md) | Funding в PnL | done | BT-4, DATA-2 |
| [BT-7](BT-7.md) | MetricsCalculator | done | BT-2 |
| [BT-8](BT-8.md) | Golden test против внешнего движка | done | BT-7 |

### E3 — Experiment Engine
| ID | Заголовок | Статус | Зависимости |
|----|-----------|--------|-------------|
| [EXP-1](EXP-1.md) | Grid/Random search | done | INF-2 |
| [EXP-2](EXP-2.md) | Непрерывный прогон + нарезка по слайсам | done | BT-7, DATA-6, EXP-1 |
| [EXP-3](EXP-3.md) | run_hash, dedupe, resume | done | INF-3, EXP-2 |
| [EXP-4](EXP-4.md) | Хранилище результатов (Parquet + каталог) | done | EXP-2, DATA-4 |
| [EXP-5](EXP-5.md) | Параллельное исполнение | done | EXP-4 |
| [EXP-6](EXP-6.md) | CLI: fetch-data / run-experiment | done | DATA-1, EXP-5 |

### E4 — Analytics
| ID | Заголовок | Статус | Зависимости |
|----|-----------|--------|-------------|
| [AN-1](AN-1.md) | Таблица parameter_performance_by_slice | done | EXP-4 |
| [AN-2](AN-2.md) | Ранжирование лучших параметров по слайсу | done | AN-1 |
| [AN-3](AN-3.md) | Stability score (robust, вектор) | done | AN-1 |
| [AN-4](AN-4.md) | Heatmap и плоты | done | AN-1 |
| [AN-5](AN-5.md) | Экспорт CSV/Parquet + ноутбук | done | AN-1 |

### E5 — Walk-forward
| ID | Заголовок | Статус | Зависимости |
|----|-----------|--------|-------------|
| [WF-1](WF-1.md) | Train/test фолды | done | DATA-6 |
| [WF-2](WF-2.md) | WF-оркестратор (out-of-sample) | done | WF-1, EXP-2, AN-3 |
| [WF-3](WF-3.md) | Untouched holdout / контроль переоптимизации | done | WF-2 |

### E6 — Market Regimes
| ID | Заголовок | Статус | Зависимости |
|----|-----------|--------|-------------|
| [REG-1](REG-1.md) | Доп. данные: funding/OI/LS ratio | done | DATA-2 |
| [REG-2](REG-2.md) | Фичи режима + time_slice_features | done | REG-1, DATA-6 |
| [REG-3](REG-3.md) | Корреляции параметры↔режим | done | REG-2, AN-1 |

### E7 — First real research run
| ID | Заголовок | Статус | Зависимости |
|----|-----------|--------|-------------|
| [RUN-1](RUN-1.md) | Resume/dedupe и параллелизм в CLI-прогоне | done | EXP-3, EXP-4, EXP-5, EXP-6 |
| [RUN-2](RUN-2.md) | CLI загрузки funding/OI/LS | done | REG-1, EXP-6 |
| [RUN-3](RUN-3.md) | CLI построения `time_slice_features` | todo | REG-2, RUN-2 |
| [RUN-4](RUN-4.md) | CLI walk-forward и holdout + конфиг | todo | WF-2, WF-3, EXP-6 |
| [RUN-5](RUN-5.md) | Режимные слоты AN-1 + CLI `analyze-regimes` | todo | REG-3, RUN-3 |
| [RUN-6](RUN-6.md) | Мультитаймфрейм, актуальные конфиги, README | todo | EXP-6 |
| [RUN-7](RUN-7.md) | Первое реальное исследование end-to-end | todo | RUN-1..6 |
| [RUN-8](RUN-8.md) | Производительность walk-forward | todo | RUN-7 |
