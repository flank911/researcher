# REG-3 — Корреляции параметры↔режим

- **Эпик:** E6 — Market Regimes
- **Статус:** done
- **Зависимости:** REG-2, AN-1

## Описание

Анализ связей параметров стратегии с рыночными режимами.

- `analytics/correlations.py`: связи params ↔ volatility / trend / funding / OI / LS.
- Объединение `parameter_performance_by_slice` с `time_slice_features`.
- Отчёт + `notebooks/003_market_regimes.ipynb`.

## Критерии приёмки (DoD)

- [x] Матрица корреляций params↔режим строится (`regime_correlations`).
- [x] Отвечает на вопросы раздела 9 (`regime_metric_summary`,
      `best_params_in_regime` — лучшие параметры в заданном режиме).
- [x] Ноутбук воспроизводит анализ (`notebooks/003_market_regimes.ipynb`).

## Примечание

- `analytics/correlations.py`:
  - `attach_regime_features(performance, features)` — left-join AN-1-таблицы с
    `time_slice_features` (REG-2) по `(symbol, timeframe, time_slice_id)`, заполняет
    слоты режима.
  - `regime_correlations(joined, metric=, by="param_hash")` — матрица корреляций
    Пирсона метрики с фичами режима (строка на `param_hash` или глобально).
  - `regime_metric_summary(joined, by=(param_hash, regime_label))` и
    `best_params_in_regime(joined, regime_label=, top=)` — ответы на вопросы
    «какие параметры лучше в режиме X».
- Модуль работает по именам колонок (имена слотов режима согласованы между AN-1 и
  REG-2), без импорта REG-2 — развязка слоёв.
- `notebooks/003_market_regimes.ipynb` самодостаточен (синтетика при отсутствии
  `TR_PERFORMANCE`/`TR_FEATURES`), исполняется без правок; тест гоняет его ячейки.
