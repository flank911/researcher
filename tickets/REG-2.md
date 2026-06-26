# REG-2 — Фичи режима + time_slice_features

- **Эпик:** E6 — Market Regimes
- **Статус:** todo
- **Зависимости:** REG-1, DATA-6
- **Реализует улучшение:** №9 (нет утечки будущего в режимах)

## Описание

Расчёт фич рыночного режима на слайс.

- `features/market_regime.py`, `funding_features.py`, `oi_features.py`,
  `ls_ratio_features.py`, `indicators.py`.
- Метрики на слайс: market_return_pct, volatility, trend_strength, ADX,
  funding_avg, oi_change_pct, long_short_ratio_avg, volume_zscore, drawdown_from_ath.
- Таблица `time_slice_features` + `regime_label`
  (bull_trend / bear_trend / sideways_low_vol / sideways_high_vol / panic_dump /
  recovery / overheated_funding).
- **Важно:** для walk-forward режим определяется только из прошлого (документировать
  и разделить research-режим vs торговый).

## Критерии приёмки (DoD)

- [ ] Фичи считаются на каждый слайс и пишутся в `time_slice_features`.
- [ ] `regime_label` присваивается по правилам.
- [ ] Задокументировано: в WF режим строится без знания будущего внутри окна.
- [ ] Юнит-тесты фич.
