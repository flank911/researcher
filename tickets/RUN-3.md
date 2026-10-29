# RUN-3 — CLI построения режимных фич (`time_slice_features`)

- **Эпик:** E7 — First real research run
- **Статус:** todo
- **Зависимости:** REG-2, RUN-2

## Описание

Команда `build-features`: `build_time_slice_features` (REG-2) → персистентная
таблица `time_slice_features` в Parquet, пригодная как вход для REG-3 (RUN-5).

## Контекст

REG-2 реализовал `compute_slice_features` / `build_time_slice_features` и схему
`TIME_SLICE_FEATURE_COLUMNS`, но вызвать их можно только из Python, и результат
никуда не сохраняется. Для реального прогона нужна таблица на диске: фичи считаются
один раз по истории и затем многократно джойнятся с результатами экспериментов.

## Решение

```
trading-research build-features --config configs/experiments/<...>.yaml \
  [--data-root data/raw] [--features-root data/features]
```

Слайсы и диапазон берутся из того же конфига эксперимента, что и прогон, — иначе
`time_slice_id` в фичах и в таблице AN-1 разойдутся и join в RUN-5 даст пустоту.

Выход: `{features_root}/{experiment}/time_slice_features.parquet`.

## Критерии приёмки (DoD)

- [ ] Таблица строится по конфигу и сохраняется в Parquet со схемой REG-2.
- [ ] `time_slice_id` совпадает с теми, что генерирует `run-experiment` на том же
      конфиге (тест на согласованность ключей join).
- [ ] Отсутствие funding/OI/LS не ломает построение — соответствующие колонки
      остаются `null` (как и задумано в `SliceFeatures`).
- [ ] Повторный запуск перезаписывает таблицу идемпотентно.
