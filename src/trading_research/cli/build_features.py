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
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from trading_research.cli.run_experiment import generate_slices_for
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
    slices = generate_slices_for(cfg)
    if not slices:
        raise ValueError(
            "config produced no time slices; check data range and time_slices.min_bars"
        )

    candle_store = DuckDBStore(data_root)
    series_store = ParquetStore(data_root)
    frames: list[pl.DataFrame] = []

    for symbol in cfg.data.symbols:
        candles = candle_store.read_window(
            exchange=cfg.data.exchange,
            market=cfg.data.market,
            symbol=symbol,
            timeframe=cfg.data.timeframe,
            start=cfg.data.start,
            end=cfg.data.end,
        )
        if candles.is_empty():
            continue
        keys = {
            "exchange": cfg.data.exchange,
            "market": cfg.data.market,
            "symbol": symbol,
            "timeframe": cfg.data.timeframe,
        }
        frames.append(
            build_time_slice_features(
                candles,
                slices,
                symbol=symbol,
                timeframe=cfg.data.timeframe,
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


def features_path(features_root: str | Path, experiment: str) -> Path:
    """Путь к таблице фич эксперимента (для читателей — RUN-5)."""
    return Path(features_root) / experiment / FEATURES_FILE


__all__ = [
    "FEATURES_FILE",
    "build_features_from_config",
    "features_path",
    "read_series_or_none",
]
