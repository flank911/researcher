# WF-3 — Untouched holdout / контроль переоптимизации

- **Эпик:** E5 — Walk-forward
- **Статус:** done
- **Зависимости:** WF-2
- **Реализует улучшение:** №8 (multiple testing / data dredging)

## Описание

Защита от подгонки под множество гипотез.

- Резерв untouched holdout: часть символов и/или хвост периода, недоступные
  на research-фазе.
- Финальная проверка найденных закономерностей только на holdout.
- (Опц. на будущее) deflated Sharpe / White's Reality Check.

## Критерии приёмки (DoD)

- [x] Holdout не участвует в research-запросах (проверка на уровне кода).
- [x] Отчёт по holdout формируется отдельно.
- [x] Документирована политика: holdout смотрится один раз, в конце.

## Примечание

- `experiments/holdout.py`:
  - `HoldoutSplit(holdout_start?, holdout_symbols?)` — резерв хвостом периода и/или
    целыми символами; `research_data()`/`holdout_data()`/`is_holdout()`.
  - `assert_research_only(data, split, symbol=)` — страховка уровня кода: бросает,
    если во фрейме есть holdout-бары (research обязан ходить через `research_data`).
  - `evaluate_holdout(...)` + `build_holdout_report(...)` → `HoldoutReport`
    (значение метрики research vs holdout + `degradation` = research − holdout);
    `save_holdout_report()` пишет отдельный JSON.
- Политика (в докстринге модуля): holdout недоступен на research-фазе и
  смотрится ровно один раз, в самом конце. Deflated Sharpe / White's Reality
  Check отмечены как опциональное будущее (не реализованы).
