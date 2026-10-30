"""RUN-3 — бизнес-логика команды ``build-features``.

REG-2 умеет считать вектор фич режима (`build_time_slice_features`), но вызвать его
можно было только из Python, и результат никуда не сохранялся. Для реального прогона
нужна таблица на диске: фичи считаются один раз по истории, а затем многократно
джойнятся с результатами экспериментов (REG-3 / RUN-5).

**Слайсы берутся из того же конфига и той же функцией, что и прогон**
(`generate_slices_for` из `cli/run_experiment.py`). Это не стилистика: если
`time_slice_id` в фичах и в таблице AN-1 разойдутся, join в RUN-5 даст пустоту —
причём молча, потому что left-join просто вернёт null во всех режимных колонках.

Отсутствующие ряды (funding/OI/LS ещё не загружены через RUN-2) не ломают
построение: соответствующие фичи остаются ``null``, как и задумано в
``SliceFeatures``. Нюанс реализации — ``read_series`` на пустом store возвращает
DataFrame **без колонок**, а REG-2 фильтрует такой ряд по ``open_time`` и падает с
``ColumnNotFoundError``; поэтому отсутствие данных нормализуется в ``None``.

**REG-4 — покрытие фич.** Устойчивость к отсутствию рядов имеет цену: команда
отрабатывала с кодом 0 и ничего не говорила, когда фича выходила пустой целиком.
На прогоне RUN-7 так и вышло — ``oi_change_pct`` и ``ls_ratio_avg`` были 114 null
из 114, и обнаружилось это только при чтении корреляций. Поэтому ``feature_coverage``
считает заполненность каждой фичи, а команда её печатает: пустая фича должна быть
видна там, где она возникла, а не через два шага пайплайна.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from trading_research.cli.run_experiment import slices_by_timeframe
from trading_research.config import load_experiment_config
from trading_research.data.storage.duckdb_store import DuckDBStore
from trading_research.data.storage.parquet_store import (
    FUNDING,
    LONG_SHORT_RATIO,
    OPEN_INTEREST,
    ParquetStore,
)
from trading_research.features.market_regime import (
    TIME_SLICE_FEATURE_COLUMNS,
    TIME_SLICE_FEATURE_SCHEMA,
    build_time_slice_features,
)

FEATURES_FILE = "time_slice_features.parquet"
_TIME_COL = "open_time"
# Ключи строки таблицы — не фичи, их заполненность считать нечего.
_KEY_COLUMNS = ("symbol", "timeframe", "time_slice_id")


def build_features_from_config(
    config_path: str | Path,
    *,
    data_root: str | Path,
    features_root: str | Path,
) -> Path:
    """Построить таблицу ``time_slice_features`` по конфигу эксперимента.

    Возвращает путь к ``{features_root}/{name}/time_slice_features.parquet``.
    Повторный запуск перезаписывает файл целиком (идемпотентно).
    """
    cfg = load_experiment_config(config_path)
    all_slices = slices_by_timeframe(cfg)
    if not any(all_slices.values()):
        raise ValueError(
            "config produced no time slices; check data range and time_slices.min_bars"
        )

    candle_store = DuckDBStore(data_root)
    series_store = ParquetStore(data_root)
    frames: list[pl.DataFrame] = []

    for symbol in cfg.data.symbols:
        for timeframe in cfg.data.timeframes:
            slices = all_slices[timeframe]
            if not slices:
                continue
            candles = candle_store.read_window(
                exchange=cfg.data.exchange,
                market=cfg.data.market,
                symbol=symbol,
                timeframe=timeframe,
                start=cfg.data.start,
                end=cfg.data.end,
            )
            if candles.is_empty():
                continue
            keys = {
                "exchange": cfg.data.exchange,
                "market": cfg.data.market,
                "symbol": symbol,
                "timeframe": timeframe,
            }
            frames.append(
                build_time_slice_features(
                    candles,
                    slices,
                    symbol=symbol,
                    timeframe=timeframe,
                    funding=read_series_or_none(series_store, FUNDING, **keys),
                    oi=read_series_or_none(series_store, OPEN_INTEREST, **keys),
                    ls=read_series_or_none(series_store, LONG_SHORT_RATIO, **keys),
                )
            )

    table = (
        pl.concat(frames, how="vertical").select(TIME_SLICE_FEATURE_COLUMNS)
        if frames
        else pl.DataFrame(schema=TIME_SLICE_FEATURE_SCHEMA)
    )

    out = Path(features_root) / cfg.name / FEATURES_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out)
    return out


def read_series_or_none(
    store: ParquetStore,
    dataset: str,
    *,
    exchange: str,
    market: str,
    symbol: str,
    timeframe: str,
) -> pl.DataFrame | None:
    """Прочитать ряд RUN-2 или вернуть ``None``, если его нет.

    ``None``, а не пустой DataFrame: ``read_series`` на отсутствующем датасете
    отдаёт фрейм **без колонок**, и REG-2, фильтруя его по ``open_time``, упал бы с
    ``ColumnNotFoundError``. Контракт ``SliceFeatures`` же ожидает именно ``None``
    как «данных нет» — тогда фича остаётся ``null``.
    """
    frame = store.read_series(
        dataset, exchange=exchange, market=market, symbol=symbol, timeframe=timeframe
    )
    if frame.is_empty() or _TIME_COL not in frame.columns:
        return None
    return frame


@dataclass(frozen=True)
class FeatureCoverage:
    """Заполненность одной фичи режима по таблице ``time_slice_features``."""

    feature: str
    filled: int
    total: int

    @property
    def is_empty(self) -> bool:
        """Фича пуста целиком — данных для неё не было ни по одному слайсу."""
        return self.filled == 0

    @property
    def is_partial(self) -> bool:
        return 0 < self.filled < self.total

    @property
    def pct(self) -> float:
        return 100.0 * self.filled / self.total if self.total else 0.0


def feature_coverage(table: pl.DataFrame) -> list[FeatureCoverage]:
    """Сколько слайсов заполнено по каждой фиче режима.

    Считается по всем колонкам схемы, кроме ключей (``symbol``, ``timeframe``,
    ``time_slice_id``): пустым может оказаться и ``regime_label``, если не хватило
    входных данных для классификации.
    """
    features = [c for c in TIME_SLICE_FEATURE_COLUMNS if c not in _KEY_COLUMNS]
    total = table.height
    return [
        FeatureCoverage(feature=name, filled=total - table[name].null_count(), total=total)
        for name in features
        if name in table.columns
    ]


def describe_feature_coverage(coverage: list[FeatureCoverage]) -> list[str]:
    """Строки отчёта о покрытии — пустые и частичные фичи названы отдельно."""
    # Пустая таблица — один отказ, а не десять пустых фич: перечислять их значило
    # бы выдать одну проблему за десять.
    if not coverage or coverage[0].total == 0:
        return ["покрытие фич: таблица пуста — ни одного слайса"]

    lines = [
        f"  {c.feature}: {c.filled}/{c.total} слайсов ({c.pct:.0f}%)"
        + ("  ← ПУСТО" if c.is_empty else "")
        for c in coverage
    ]
    empty = [c.feature for c in coverage if c.is_empty]
    header = f"покрытие фич ({coverage[0].total} слайсов):"
    if empty:
        header += f" пустых — {len(empty)}: {', '.join(empty)}"
    return [header, *lines]


def features_path(features_root: str | Path, experiment: str) -> Path:
    """Путь к таблице фич эксперимента (для читателей — RUN-5)."""
    return Path(features_root) / experiment / FEATURES_FILE


__all__ = [
    "FEATURES_FILE",
    "FeatureCoverage",
    "build_features_from_config",
    "describe_feature_coverage",
    "feature_coverage",
    "features_path",
    "read_series_or_none",
]
