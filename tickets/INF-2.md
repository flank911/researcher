# INF-2 — Доменные типы и конфиги (pydantic)

- **Эпик:** E0 — Scaffolding & инфраструктура
- **Статус:** done
- **Зависимости:** INF-1
- **Реализует улучшение:** №15 (валидация конфигов вместо сырого YAML-dict)

## Описание

Доменные модели как pydantic-классы с валидацией и версией схемы.

- `TimeSlice` (id, start, end, label, kind, warmup_bars, min_bars, role).
- `DatasetRef` (symbol, timeframe, exchange, market, range, data_path, data_hash).
- `ExecutionModel` (initial_balance, commission_rate, slippage_pct, leverage,
  position_size_type/value, allow_short, allow_pyramiding, close_on_reverse_signal,
  signal_lag/fill_on).
- `StrategyParams` (typed-обёртка / dict с валидацией на уровне стратегии).
- `ExperimentConfig` (data, time_slices, strategy, params_grid, execution, metrics, output).
- Загрузчик YAML → модель с понятными ошибками валидации.
- Поле версии схемы конфига.

## Критерии приёмки (DoD)

- [ ] Невалидный YAML даёт читаемую ошибку с указанием поля.
- [ ] Round-trip YAML ↔ модель покрыт тестом.
- [ ] Пример конфига из плана (раздел 12) парсится без ошибок.
- [ ] Все модели — `frozen`/immutable где уместно.
