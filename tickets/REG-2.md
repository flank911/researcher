# REG-2 — Фичи режима + time_slice_features

- **Эпик:** E6 — Market Regimes
- **Статус:** done
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

- [x] Фичи считаются на каждый слайс и пишутся в `time_slice_features`.
- [x] `regime_label` присваивается по правилам.
- [x] Задокументировано: в WF режим строится без знания будущего внутри окна.
- [x] Юнит-тесты фич.

## Примечание

- Пакет `features/`: `indicators.py` (market_return_pct, volatility, trend_strength
  = корреляция Пирсона, ADX по Уайлдеру, volume_zscore, drawdown_from_ath),
  `funding_features.py`, `oi_features.py`, `ls_ratio_features.py`,
  `market_regime.py` (оркестратор).
- `build_time_slice_features(candles, slices, …)` → таблица `time_slice_features`
  (строка на слайс); колонки режима совпадают по именам со слотами AN-1
  (market_return_pct/volatility/trend_strength/funding_avg/oi_change_pct/
  ls_ratio_avg/regime_label) → прямой вход для REG-3.
- `classify_regime` по порогам `RegimeThresholds` →
  bull_trend / bear_trend / sideways_low_vol / sideways_high_vol / panic_dump /
  recovery / overheated_funding.
- **Без утечки будущего (№9):** фичи слайса считаются только из баров
  `[slice.start, slice.end)`; в докстринге разделены research- и торговый режимы
  (функция будущее не читает, различие — лишь в доступных вызывающему слайсах).
