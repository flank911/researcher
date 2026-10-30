"""Tests for DATA-8 — CLI-команда ``validate-data``.

Всё на временном store, без сети. Отдельно закреплены два места, где команда
легко соврала бы: шаг funding (его задаёт биржа, а не путь хранения) и окно
сверки для OI/LS (его урезает retention).
"""

from __future__ import annotations

import json
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from trading_research.cli.main import app
from trading_research.cli.validate_data import (
    describe_pair,
    describe_total,
    has_defects,
    infer_step,
    validate_data_from_config,
)
from trading_research.data.schema import (
    CANDLE_SCHEMA,
    FUNDING_SCHEMA,
    LONG_SHORT_RATIO_SCHEMA,
    OPEN_INTEREST_SCHEMA,
)
from trading_research.data.storage.parquet_store import (
    CANDLES,
    FUNDING,
    LONG_SHORT_RATIO,
    OPEN_INTEREST,
    ParquetStore,
)

runner = CliRunner()

T0 = datetime(2024, 1, 1, tzinfo=UTC)
HOURS = 24 * 60  # конфиг ниже просит ровно 60 дней
END = T0 + timedelta(hours=HOURS)
EXP = "validate_exp"


def _keys(symbol: str = "BTCUSDT") -> dict[str, str]:
    return {"exchange": "binance", "market": "futures", "symbol": symbol, "timeframe": "1h"}


def _candles(times: list[datetime]) -> pl.DataFrame:
    close = [100.0 + i % 7 for i in range(len(times))]
    return pl.DataFrame(
        {
            "open_time": times,
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [10.0] * len(times),
            "close_time": [t + timedelta(hours=1) for t in times],
            "quote_volume": [1000.0] * len(times),
            "trades": [5] * len(times),
            "taker_buy_base": [4.0] * len(times),
            "taker_buy_quote": [400.0] * len(times),
        },
        schema=CANDLE_SCHEMA,
    )


def _hours(n: int = HOURS) -> list[datetime]:
    return [T0 + timedelta(hours=i) for i in range(n)]


def _seed_candles(
    root: Path, *, symbols: tuple[str, ...] = ("BTCUSDT",), skip: range | None = None
) -> None:
    times = [t for i, t in enumerate(_hours()) if skip is None or i not in skip]
    store = ParquetStore(root)
    for symbol in symbols:
        store.write_candles(_candles(times), **_keys(symbol))


def _seed_series(root: Path, *, symbol: str = "BTCUSDT", funding_drift: bool = False) -> None:
    store = ParquetStore(root)
    times = _hours()

    # Времена выплат биржа отдаёт с миллисекундным дрейфом: в реальной истории
    # 8 ч + 1 мс встречается сотнями. Шаг от этого не меняется.
    funding_times = times[::8]
    if funding_drift:
        funding_times = [
            t + timedelta(milliseconds=(i % 3) - 1) for i, t in enumerate(funding_times)
        ]
    store.write_series(
        FUNDING,
        pl.DataFrame(
            {"open_time": funding_times, "funding_rate": [-0.0002] * len(funding_times)},
            schema=FUNDING_SCHEMA,
        ),
        **_keys(symbol),
    )
    store.write_series(
        OPEN_INTEREST,
        pl.DataFrame(
            {
                "open_time": times,
                "open_interest": [1000.0 + i for i in range(len(times))],
                "open_interest_value": [5e6] * len(times),
            },
            schema=OPEN_INTEREST_SCHEMA,
        ),
        **_keys(symbol),
    )
    store.write_series(
        LONG_SHORT_RATIO,
        pl.DataFrame(
            {
                "open_time": times,
                "long_short_ratio": [1.5] * len(times),
                "long_account": [0.6] * len(times),
                "short_account": [0.4] * len(times),
            },
            schema=LONG_SHORT_RATIO_SCHEMA,
        ),
        **_keys(symbol),
    )


def _config(path: Path, *, symbols: str = "[BTCUSDT]", timeframes: str = "[1h]") -> Path:
    path.write_text(
        textwrap.dedent(
            f"""
            schema_version: 2
            name: {EXP}
            data:
              exchange: binance
              market: futures
              symbols: {symbols}
              timeframes: {timeframes}
              start: "{T0:%Y-%m-%d}"
              end: "{END:%Y-%m-%d}"
            time_slices:
              type: month
              min_bars: 50
            strategy:
              name: ma_cross
            params_grid:
              fast_ma: [5]
              slow_ma: [20]
            execution:
              initial_balance: 10000
              commission_rate: 0.0006
              slippage_pct: 0.0003
              leverage: 3
              position_size_type: percent_balance
              position_size_value: 20
              signal_lag: 1
              fill_on: next_open
            """
        ),
        encoding="utf-8",
    )
    return path


def _run(tmp_path: Path, **kwargs):
    """Валидация полного набора данных; ``now`` — конец окна конфига."""
    return validate_data_from_config(
        _config(tmp_path / "cfg.yaml"),
        data_root=tmp_path / "raw",
        reports_root=tmp_path / "reports",
        now=END,
        **kwargs,
    )


def _check(pair, dataset: str):
    return next(c for c in pair.checks if c.dataset == dataset)


# --- чистые данные --------------------------------------------------------


def test_clean_store_has_no_defects(tmp_path: Path) -> None:
    _seed_candles(tmp_path / "raw")
    _seed_series(tmp_path / "raw")

    pairs = _run(tmp_path)

    assert len(pairs) == 1
    assert not has_defects(pairs)
    assert _check(pairs[0], CANDLES).rows == HOURS
    assert "дефектов нет" in describe_total(pairs)


def test_series_are_validated_not_only_candles(tmp_path: Path) -> None:
    """DoD: ``validate_series`` должен вызываться, иначе второй валидатор без CLI."""
    _seed_candles(tmp_path / "raw")
    _seed_series(tmp_path / "raw")

    checks = {c.dataset: c for c in _run(tmp_path)[0].checks}

    assert set(checks) == {CANDLES, FUNDING, OPEN_INTEREST, LONG_SHORT_RATIO}
    for dataset in (FUNDING, OPEN_INTEREST, LONG_SHORT_RATIO):
        assert checks[dataset].report is not None, dataset
        assert checks[dataset].report.total_rows > 0, dataset


def test_reports_are_saved_and_paths_visible(tmp_path: Path) -> None:
    _seed_candles(tmp_path / "raw")
    _seed_series(tmp_path / "raw")

    pairs = _run(tmp_path)
    lines = "\n".join(describe_pair(pairs[0]))

    for check in pairs[0].checks:
        assert check.path is not None
        assert check.path.exists()
        assert str(check.path) in lines
        saved = json.loads(check.path.read_text(encoding="utf-8"))
        assert saved["step_ms"] > 0


# --- шаг ряда -------------------------------------------------------------


def test_funding_step_comes_from_data_not_from_partition(tmp_path: Path) -> None:
    """Партиция funding лежит под 1h, но выплаты идут раз в 8 ч.

    Со «шагом партиции» команда отчиталась бы о пропуске на каждой точке.
    """
    _seed_candles(tmp_path / "raw")
    _seed_series(tmp_path / "raw")

    funding = _check(_run(tmp_path)[0], FUNDING)

    assert funding.step == "8h"
    assert funding.report is not None
    assert funding.report.gaps == []


def test_millisecond_drift_is_not_a_gap(tmp_path: Path) -> None:
    """Регрессия: 8 ч + 1 мс давало ``missing_bars=0``, запрещённый ``Field(gt=0)``."""
    _seed_candles(tmp_path / "raw")
    _seed_series(tmp_path / "raw", funding_drift=True)

    funding = _check(_run(tmp_path)[0], FUNDING)

    assert funding.step == "8h"
    assert funding.report is not None
    assert funding.report.gaps == []
    assert funding.report.is_clean


@pytest.mark.parametrize("step_hours,expected", [(1, "1h"), (4, "4h"), (8, "8h"), (24, "1d")])
def test_infer_step_recognises_supported_timeframes(step_hours: int, expected: str) -> None:
    frame = pl.DataFrame({"open_time": [T0 + timedelta(hours=step_hours * i) for i in range(10)]})
    assert infer_step(frame) == expected


def test_infer_step_gives_none_on_unrecognisable_step() -> None:
    frame = pl.DataFrame({"open_time": [T0 + timedelta(minutes=7 * i) for i in range(10)]})
    assert infer_step(frame) is None


def test_unrecognisable_step_skips_gap_check_instead_of_inventing_one(tmp_path: Path) -> None:
    _seed_candles(tmp_path / "raw")
    ParquetStore(tmp_path / "raw").write_series(
        FUNDING,
        pl.DataFrame(
            {
                "open_time": [T0 + timedelta(minutes=7 * i) for i in range(10)],
                "funding_rate": [0.0001] * 10,
            },
            schema=FUNDING_SCHEMA,
        ),
        **_keys(),
    )

    funding = _check(_run(tmp_path)[0], FUNDING)

    assert funding.report is None
    assert funding.note is not None
    assert "шаг" in funding.note
    assert not funding.is_defect


# --- дефекты и --strict ---------------------------------------------------


def test_gap_in_candles_is_reported(tmp_path: Path) -> None:
    _seed_candles(tmp_path / "raw", skip=range(100, 110))
    _seed_series(tmp_path / "raw")

    pairs = _run(tmp_path)
    candles = _check(pairs[0], CANDLES)

    assert has_defects(pairs)
    assert candles.report is not None
    assert candles.report.missing_bars_total == 10
    assert "пропусков 1 (10 баров)" in describe_pair(pairs[0])[1]


def test_missing_candles_are_a_defect_but_missing_series_are_not(tmp_path: Path) -> None:
    """funding/OI/LS опциональны по контракту ``build_features``, свечи — нет."""
    only_series = tmp_path / "series_only"
    _seed_series(only_series)
    candles = _check(
        validate_data_from_config(
            _config(tmp_path / "cfg.yaml"),
            data_root=only_series,
            reports_root=tmp_path / "reports",
            now=END,
        )[0],
        CANDLES,
    )
    assert candles.is_defect
    assert candles.note == "нет данных"

    only_candles = tmp_path / "candles_only"
    _seed_candles(only_candles)
    pairs = validate_data_from_config(
        _config(tmp_path / "cfg.yaml"),
        data_root=only_candles,
        reports_root=tmp_path / "reports",
        now=END,
    )
    assert not has_defects(pairs)
    assert _check(pairs[0], OPEN_INTEREST).note == "нет данных"


def test_truncated_series_is_a_defect(tmp_path: Path) -> None:
    """Ряд, оборвавшийся в начале окна, — дефект, а не «2 точки, чисто».

    Дубли и немонотонность до CLI дойти не могут: ``write_series`` схлопывает их
    при записи — они проверяются на уровне валидатора (``test_validators``).
    """
    _seed_candles(tmp_path / "raw", symbols=("ETHUSDT",))
    ParquetStore(tmp_path / "raw").write_series(
        OPEN_INTEREST,
        pl.DataFrame(
            {
                "open_time": [T0, T0 + timedelta(hours=1)],
                "open_interest": [1.0, 2.0],
                "open_interest_value": [1.0, 2.0],
            },
            schema=OPEN_INTEREST_SCHEMA,
        ),
        **_keys("ETHUSDT"),
    )

    pairs = validate_data_from_config(
        _config(tmp_path / "cfg.yaml", symbols="[ETHUSDT]"),
        data_root=tmp_path / "raw",
        reports_root=tmp_path / "reports",
        now=END,
    )
    oi = _check(pairs[0], OPEN_INTEREST)

    assert oi.report is not None
    assert oi.report.missing_total == HOURS - 2
    assert oi.is_defect
    assert has_defects(pairs)


# --- окно retention -------------------------------------------------------


def test_window_older_than_retention_is_named_not_reported_as_a_gap(tmp_path: Path) -> None:
    """Истории OI/LS старше ~30 дней у биржи не существует — это не пропуск."""
    _seed_candles(tmp_path / "raw")
    _seed_series(tmp_path / "raw")

    # «Сейчас» на год позже окна конфига: весь диапазон вне retention.
    pairs = validate_data_from_config(
        _config(tmp_path / "cfg.yaml"),
        data_root=tmp_path / "raw",
        reports_root=tmp_path / "reports",
        now=END + timedelta(days=365),
    )
    oi = _check(pairs[0], OPEN_INTEREST)

    assert oi.report is not None
    assert oi.report.gaps == []
    assert oi.note == "окно конфига старше retention"
    assert not has_defects(pairs)


def test_funding_window_is_not_clamped(tmp_path: Path) -> None:
    """У funding ограничения retention нет — окно сверки остаётся полным."""
    _seed_candles(tmp_path / "raw")
    _seed_series(tmp_path / "raw")

    funding = _check(
        validate_data_from_config(
            _config(tmp_path / "cfg.yaml"),
            data_root=tmp_path / "raw",
            reports_root=tmp_path / "reports",
            now=END + timedelta(days=365),
        )[0],
        FUNDING,
    )

    assert funding.note is None
    assert funding.report is not None
    assert funding.report.gaps == []


# --- CLI ------------------------------------------------------------------


def _invoke(tmp_path: Path, *extra: str):
    return runner.invoke(
        app,
        [
            "validate-data",
            "--config",
            str(_config(tmp_path / "cfg.yaml")),
            "--data-root",
            str(tmp_path / "raw"),
            "--out",
            str(tmp_path / "reports"),
            *extra,
        ],
    )


def test_cli_runs_and_prints_report_paths(tmp_path: Path) -> None:
    _seed_candles(tmp_path / "raw")
    _seed_series(tmp_path / "raw")

    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    assert "BTCUSDT 1h:" in result.output
    assert str(tmp_path / "reports" / EXP / "BTCUSDT_1h_candles.json") in result.output
    assert "дефектов нет" in result.output


def test_cli_strict_exits_non_zero_only_with_defects(tmp_path: Path) -> None:
    _seed_candles(tmp_path / "raw", skip=range(100, 110))
    _seed_series(tmp_path / "raw")

    # Без --strict команда остаётся диагностикой: дефекты видны, код 0.
    soft = _invoke(tmp_path)
    assert soft.exit_code == 0, soft.output
    assert "пропусков 1" in soft.output

    strict = _invoke(tmp_path, "--strict")
    assert strict.exit_code == 1, strict.output
    assert "с дефектами: 1" in strict.output


def test_cli_strict_passes_on_clean_data(tmp_path: Path) -> None:
    _seed_candles(tmp_path / "raw")
    _seed_series(tmp_path / "raw")

    result = _invoke(tmp_path, "--strict")

    assert result.exit_code == 0, result.output


def test_cli_narrows_by_symbol_and_timeframe(tmp_path: Path) -> None:
    _seed_candles(tmp_path / "raw", symbols=("BTCUSDT", "ETHUSDT"))
    _seed_series(tmp_path / "raw")

    config = _config(tmp_path / "cfg.yaml", symbols="[BTCUSDT, ETHUSDT]")
    result = runner.invoke(
        app,
        [
            "validate-data",
            "--config",
            str(config),
            "--data-root",
            str(tmp_path / "raw"),
            "--out",
            str(tmp_path / "reports"),
            "--symbol",
            "ETHUSDT",
            "--timeframe",
            "1h",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "ETHUSDT 1h:" in result.output
    assert "BTCUSDT" not in result.output
    assert "итого: 1 пара" in result.output
