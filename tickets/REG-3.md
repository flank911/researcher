# REG-3 — Корреляции параметры↔режим

- **Эпик:** E6 — Market Regimes
- **Статус:** todo
- **Зависимости:** REG-2, AN-1

## Описание

Анализ связей параметров стратегии с рыночными режимами.

- `analytics/correlations.py`: связи params ↔ volatility / trend / funding / OI / LS.
- Объединение `parameter_performance_by_slice` с `time_slice_features`.
- Отчёт + `notebooks/003_market_regimes.ipynb`.

## Критерии приёмки (DoD)

- [ ] Матрица корреляций params↔режим строится.
- [ ] Отвечает на вопросы из раздела 9 плана (какие параметры лучше при высокой
      волатильности и т.п.).
- [ ] Ноутбук воспроизводит анализ.
