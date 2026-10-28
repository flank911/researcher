# WF-1 — Train/test фолды

- **Эпик:** E5 — Walk-forward
- **Статус:** done
- **Зависимости:** DATA-6

## Описание

Генерация пар окон для walk-forward.

- Расширение сплиттера (`time_splitters.py`) на rolling train/test пары.
- Роли слайсов `train` / `test`; связь между парными окнами (fold id).
- Параметры: длина train, длина test, шаг сдвига.

## Критерии приёмки (DoD)

- [x] Генерация фолдов: rolling и anchored (expanding) пары train/test.
- [x] Корректные границы train/test без перекрытия (`test.start == train.end`).
- [x] Юнит-тесты на границы и шаг.

## Примечание

- `time_splitters.py` → `generate_walk_forward_folds(start, end, *, train, test,
  step, warmup_bars=, anchored=)` → `list[WalkForwardFold]` (пара
  `train`/`test` TimeSlice + `fold`-индекс; роли `TRAIN`/`TEST`, kind `CUSTOM`).
- Длины train/test/шаг — `timedelta`. `anchored=False` — скользящее окно
  фиксированной длины; `anchored=True` — расширяющийся train от `start`.
- Тест всегда строго после train (`test.start == train.end`, без перекрытия);
  включаются только фолды, чей `test` целиком помещается в `[start, end)`.
- «Раздел 10 плана» в репозитории отсутствует — реализована общепринятая
  rolling/anchored схема; параметры заданы длительностями для совместимости с
  календарным сплиттером DATA-6.
