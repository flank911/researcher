"""Tests for DATA-4 — dataset catalog + content data_hash."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from trading_research.data.catalog import DatasetCatalog, dataset_hash
from trading_research.data.schema import CANDLE_SCHEMA

REF = {
    "name": "btc_1h",
    "exchange": "binance",
    "market": "futures",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "data_path": "data/raw/candles/...",
}


def _candles(n: int, start: datetime | None = None) -> pl.DataFrame:
    start = start or datetime(2024, 1, 1, tzinfo=UTC)
    times = [start + timedelta(hours=i) for i in range(n)]
    return pl.DataFrame(
        {
            "open_time": times,
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [10.0 + i for i in range(n)],
            "close_time": [t + timedelta(hours=1) for t in times],
            "quote_volume": [1000.0 + i for i in range(n)],
            "trades": [5 + i for i in range(n)],
            "taker_buy_base": [4.0 + i for i in range(n)],
            "taker_buy_quote": [400.0 + i for i in range(n)],
        },
        schema=CANDLE_SCHEMA,
    )


# --- content hash stability -------------------------------------------------


def test_same_data_same_hash() -> None:
    assert dataset_hash(_candles(24)) == dataset_hash(_candles(24))


def test_hash_independent_of_row_order() -> None:
    df = _candles(24)
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=7)
    assert dataset_hash(shuffled) == dataset_hash(df)


def test_different_data_different_hash() -> None:
    assert dataset_hash(_candles(24)) != dataset_hash(_candles(48))


def test_empty_candles_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        dataset_hash(pl.DataFrame(schema=CANDLE_SCHEMA))


# --- register / resolve -----------------------------------------------------


def test_register_returns_ref_and_resolves_by_id(tmp_path: Path) -> None:
    catalog = DatasetCatalog(tmp_path / "catalog.sqlite")
    ref = catalog.register(_candles(24), **REF)

    assert ref.data_hash == dataset_hash(_candles(24))
    assert ref.symbol == "BTCUSDT"
    assert ref.start_at == datetime(2024, 1, 1, tzinfo=UTC)
    assert ref.end_at == datetime(2024, 1, 1, 23, tzinfo=UTC)

    resolved = catalog.get(ref.id)
    assert resolved is not None
    assert resolved == ref


def test_resolve_by_hash(tmp_path: Path) -> None:
    catalog = DatasetCatalog(tmp_path / "catalog.sqlite")
    ref = catalog.register(_candles(24), **REF)
    by_hash = catalog.get_by_hash(ref.data_hash)
    assert by_hash is not None
    assert by_hash.id == ref.id


def test_register_is_idempotent_for_identical_content(tmp_path: Path) -> None:
    catalog = DatasetCatalog(tmp_path / "catalog.sqlite")
    a = catalog.register(_candles(24), **REF)
    b = catalog.register(_candles(24), **REF)
    assert a == b
    assert len(catalog.versions("BTCUSDT", "1h")) == 1


def test_get_missing_returns_none(tmp_path: Path) -> None:
    catalog = DatasetCatalog(tmp_path / "catalog.sqlite")
    assert catalog.get("nope") is None
    assert catalog.get_by_hash("nope") is None
    assert catalog.latest("BTCUSDT", "1h") is None


# --- versioning: refetch creates a new version, old stays -------------------


def test_refetch_creates_new_version_without_breaking_old(tmp_path: Path) -> None:
    catalog = DatasetCatalog(tmp_path / "catalog.sqlite")
    v1 = catalog.register(_candles(24), **REF)
    # Докачка: те же 24 бара + ещё 24 → другое содержимое → новая версия.
    v2 = catalog.register(_candles(48), **REF)

    assert v1.id != v2.id
    assert v1.data_hash != v2.data_hash

    versions = catalog.versions("BTCUSDT", "1h")
    assert len(versions) == 2
    assert catalog.latest("BTCUSDT", "1h") == v2

    # Ссылка старого прогона (по id и по хэшу) остаётся валидной.
    assert catalog.get(v1.id) == v1
    assert catalog.get_by_hash(v1.data_hash) == v1


def test_hash_stable_across_processes(tmp_path: Path) -> None:
    # Хэш детерминирован — повторная регистрация тех же данных даёт ту же id.
    catalog = DatasetCatalog(tmp_path / "catalog.sqlite")
    ref = catalog.register(_candles(24), **REF)
    reopened = DatasetCatalog(tmp_path / "catalog.sqlite")
    assert reopened.register(_candles(24), **REF).id == ref.id
    assert len(reopened.versions("BTCUSDT", "1h")) == 1
