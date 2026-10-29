# RUN-5 — Заполнение режимных слотов AN-1 + CLI `analyze-regimes`

- **Эпик:** E7 — First real research run
- **Статус:** todo
- **Зависимости:** REG-3, RUN-3

## Описание

Довести REG-3 до исполнимости из CLI и закрыть расхождение набора фич между AN-1 и
REG-2.

## Контекст

Два разрыва.

**1. Слоты AN-1 никогда не заполняются.**
[REGIME_SLOTS](../src/trading_research/analytics/queries.py) зарезервированы в схеме
`parameter_performance_by_slice` как плейсхолдеры, но `run_experiment_from_config`
их не наполняет — в `performance.parquet` они остаются `null`. Заполнять их должен
join с `time_slice_features` (RUN-3) через `attach_regime_features`, но вызвать его
из CLI сейчас нечем.

**2. Три фичи REG-2 выпадают из корреляций.**
`SliceFeatures` (REG-2) считает **9** числовых фич, а `REGIME_SLOTS` (AN-1)
перечисляет только **6**. Вне списка остались `adx`, `volume_zscore` и
`drawdown_from_ath`. Через join они в таблицу попадают (`attach_regime_features`
берёт все неключевые колонки), но `NUMERIC_REGIME_FEATURES` в REG-3 выводится из
`REGIME_SLOTS`, поэтому **по умолчанию корреляции считаются без них** — молча, без
ошибки. Передать их можно только явным `features=`.

Это ровно тот класс тихого расхождения, от которого страхует критический путь:
аналитик получит матрицу корреляций, не заметив, что треть фич режима в неё не
вошла.

## Решение

- Привести `REGIME_SLOTS` в соответствие с `SliceFeatures`: либо расширить до всех
  9 числовых фич, либо оставить 6 и **явно** задокументировать, что остальные —
  join-only, с тестом, фиксирующим это решение. Предпочтительно первое:
  расхождение двух списков истины — источник будущих ошибок.
- Команда:

  ```
  trading-research analyze-regimes --experiment <name> \
    [--metric total_return_pct] [--regime bull_trend] [--top 20]
  ```

  печатает матрицу `regime_correlations`, а с `--regime` — `best_params_in_regime`.
- Экспорт матрицы в CSV/Parquet по аналогии с `export-results`.

## Критерии приёмки (DoD)

- [ ] `REGIME_SLOTS` и числовые поля `SliceFeatures` согласованы; несогласованность
      ловится тестом, а не ревью.
- [ ] `analyze-regimes` строит матрицу корреляций по сохранённым таблицам AN-1 + RUN-3.
- [ ] `--regime X` выводит top-N параметров внутри режима.
- [ ] Пустой join (несовпадение `time_slice_id`) даёт явную ошибку, а не таблицу
      из `null`.
- [ ] Матрица экспортируется в CSV/Parquet.
