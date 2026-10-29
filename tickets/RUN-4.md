# RUN-4 — CLI walk-forward и holdout + секции конфига

- **Эпик:** E7 — First real research run
- **Статус:** todo
- **Зависимости:** WF-2, WF-3, EXP-6

## Описание

Весь эпик E5 доступен только из Python. Добавить команду `walk-forward` и секции
`walk_forward` / `holdout` в `ExperimentConfig`, чтобы финальная проверка стратегии
(принцип №8 из README — «финальная проверка только walk-forward») была исполнима из
CLI.

## Контекст

`run_walk_forward` (WF-2), `generate_walk_forward_folds` (WF-1) и весь модуль
`holdout.py` (WF-3) не упоминаются ни в `cli/`, ни в `config.py`.
[ExperimentConfig](../src/trading_research/config.py) содержит `data`,
`time_slices`, `strategy`, `params_grid`, `execution`, `slice_mode`, `metrics`,
`output` — ни одного поля про фолды или holdout.

То есть сейчас платформа умеет только то, что README называет переоптимизированным
поиском (сетка по слайсам), а честный out-of-sample из CLI недостижим.

## Решение

Секции конфига:

```yaml
walk_forward:
  train_span: 6          # периодов train
  test_span: 1           # периодов test
  step: 1
  select_by: sharpe       # метрика отбора на train

holdout:
  end: "2026-01-01"       # всё после — untouched, в research не попадает
```

Команда:

```
trading-research walk-forward --config <...> [--data-root] [--results-root]
```

Выход: `selections` по фолдам, склеенная OOS-кривая, OOS-метрики и
`HoldoutReport` (через `save_holdout_report`) в каталоге эксперимента.

`assert_research_only` вызывается **до** прогона: если research-данные залезают в
holdout-окно, команда падает, а не предупреждает.

## Критерии приёмки (DoD)

- [ ] Секции `walk_forward` / `holdout` валидируются pydantic с читаемыми ошибками.
- [ ] `walk-forward` печатает выбор по фолдам и OOS-метрики, сохраняет OOS-кривую.
- [ ] `HoldoutReport` сохраняется в каталог эксперимента.
- [ ] Попытка прогнать research на данных, пересекающих holdout, падает
      (`assert_research_only`).
- [ ] Отсутствие секции `holdout` допустимо — тогда holdout-шаг пропускается.
