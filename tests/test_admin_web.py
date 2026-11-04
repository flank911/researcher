"""Tests for ADM-3 — веб-админка только для чтения.

Веб-зависимости лежат в экстре ``admin``, поэтому модуль пропускается, если её не
ставили — тем же приёмом, что golden-тест BT-8. В CI экстра ставится вместе с dev,
то есть страницы там проверяются, а не пропускаются.

Проверяется ровно то, что добавляет эта страница поверх ADM-1 и ADM-2: что числа на
ней те же, что в CLI, что менять ею ничего нельзя, что состояние читается на каждый
запрос и что значение holdout на страницу не попадает.
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
import textwrap
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import polars as pl
import pytest
from typer.testing import CliRunner

pytest.importorskip("fastapi", reason="веб-админка живёт в экстре admin")
pytest.importorskip("httpx2", reason="starlette.testclient требует httpx2")

from fastapi.testclient import TestClient  # noqa: E402

from trading_research.admin.web import Roots, create_app  # noqa: E402
from trading_research.cli import admin as cli_admin  # noqa: E402
from trading_research.cli.main import app as cli_app  # noqa: E402
from trading_research.cli.status import (  # noqa: E402
    StatusReport,
    describe_results,
    describe_schedule,
    describe_validation,
    status_from_config,
    status_payload,
)
from trading_research.cli.validate_data import validate_data_from_config  # noqa: E402
from trading_research.cli.walk_forward import (  # noqa: E402
    HOLDOUT_REPORT_FILE,
    SELECTIONS_FILE,
    wf_dir,
)
from trading_research.data.schema import CANDLE_SCHEMA, FUNDING_SCHEMA  # noqa: E402
from trading_research.data.storage.parquet_store import FUNDING, ParquetStore  # noqa: E402

runner = CliRunner()

T0 = datetime(2024, 1, 1, tzinfo=UTC)
HOURS = 24 * 40
END = T0 + timedelta(hours=HOURS)
EXP = "web_exp"
HOLDOUT_VALUE = -1.70431373826594

KEYS = {"exchange": "binance", "market": "futures", "symbol": "BTCUSDT", "timeframe": "1h"}


# --- фикстуры ---------------------------------------------------------------


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


def _seed(root: Path, *, hours: int = HOURS, skip: range | None = None) -> None:
    store = ParquetStore(root)
    times = [t for i, t in enumerate(_hours(hours)) if skip is None or i not in skip]
    store.write_candles(_candles(times), **KEYS)
    funding_times = times[::8]
    store.write_series(
        FUNDING,
        pl.DataFrame(
            {"open_time": funding_times, "funding_rate": [-0.0002] * len(funding_times)},
            schema=FUNDING_SCHEMA,
        ),
        **KEYS,
    )


def _config(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            f"""
            schema_version: 2
            name: {EXP}
            data:
              exchange: binance
              market: futures
              symbols: [BTCUSDT]
              timeframes: [1h]
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


def _roots(tmp_path: Path, *, log: Path | None = None) -> Roots:
    return Roots(
        config=_config(tmp_path / "cfg.yaml"),
        data=tmp_path / "raw",
        results=tmp_path / "results",
        reports=tmp_path / "reports",
        log=log,
    )


def _client(tmp_path: Path, *, log: Path | None = None) -> TestClient:
    return TestClient(create_app(_roots(tmp_path, log=log)))


def _validate(tmp_path: Path) -> None:
    validate_data_from_config(
        _config(tmp_path / "cfg.yaml"),
        data_root=tmp_path / "raw",
        reports_root=tmp_path / "reports",
        now=END,
    )


def _report(tmp_path: Path, *, log: Path | None = None) -> StatusReport:
    return status_from_config(
        _config(tmp_path / "cfg.yaml"),
        data_root=tmp_path / "raw",
        results_root=tmp_path / "results",
        reports_root=tmp_path / "reports",
        log_path=log,
    )


# --- страницы отвечают ------------------------------------------------------


def test_pages_render_on_a_seeded_store(tmp_path: Path) -> None:
    _seed(tmp_path / "raw")
    _validate(tmp_path)
    client = _client(tmp_path)

    overview = client.get("/")
    data = client.get("/data")

    assert overview.status_code == 200
    assert data.status_code == 200
    assert EXP in overview.text
    assert "candles" in data.text
    assert f"{HOURS:,}".replace(",", " ") in data.text  # 960 баров, с разрядами


def test_empty_store_renders_instead_of_failing(tmp_path: Path) -> None:
    """Пустой store — состояние, а не ошибка: страница объясняет, чего нет."""
    client = _client(tmp_path)

    overview = client.get("/")
    data = client.get("/data")

    assert overview.status_code == 200
    assert data.status_code == 200
    assert "прогонов ещё не было" in overview.text
    assert "данных нет" in data.text


# --- одна правда: страница и CLI --------------------------------------------


def test_overview_repeats_the_cli_lines(tmp_path: Path) -> None:
    """Блоки страницы — те же строки, что печатает ``status``."""
    _seed(tmp_path / "raw")
    _validate(tmp_path)
    log = tmp_path / "update.log"
    log.write_text("=== 2024-02-09 06:00:01 UTC — cfg.yaml\n--- exit=0\n", encoding="utf-8")

    report = _report(tmp_path, log=log)
    text = _client(tmp_path, log=log).get("/").text

    assert describe_schedule(report.state) in text
    assert describe_validation(report) in text
    assert describe_results(report.state) in text


def test_api_status_matches_the_cli_payload(tmp_path: Path) -> None:
    """``/api/status`` отдаёт то же, что ``status --json`` — без второго счёта."""
    _seed(tmp_path / "raw")
    _validate(tmp_path)

    payload = _client(tmp_path).get("/api/status").json()
    expected = status_payload(_report(tmp_path))

    assert payload["experiment"] == expected["experiment"]
    assert payload["validation"] == expected["validation"]
    assert len(payload["datasets"]) == len(expected["datasets"])
    assert [a["message"] for a in payload["alerts"]] == [a["message"] for a in expected["alerts"]]


def test_defects_are_worded_as_validate_data_words_them(tmp_path: Path) -> None:
    """Дефект, названный в CLI одним словом, а на странице другим, читался бы как два."""
    _seed(tmp_path / "raw", skip=range(10, 20))
    _validate(tmp_path)

    text = _client(tmp_path).get("/data").text
    assert "пропусков 1" in text


def test_undated_verdict_is_not_shown_as_fresh(tmp_path: Path) -> None:
    _seed(tmp_path / "raw")
    _validate(tmp_path)
    for path in (tmp_path / "reports" / EXP).glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("checked_at")
        path.write_text(json.dumps(payload), encoding="utf-8")

    assert "время неизвестно" in _client(tmp_path).get("/data").text


# --- границы: только чтение, свежесть, резерв -------------------------------


def test_every_route_is_read_only(tmp_path: Path) -> None:
    """Контроль — отдельный тикет (ADM-6); до него менять нечем в принципе."""
    app = create_app(_roots(tmp_path))

    methods = {m for route in app.routes for m in getattr(route, "methods", set())}
    assert methods <= {"GET", "HEAD"}


def test_state_is_read_on_every_request(tmp_path: Path) -> None:
    """Кэша нет: страница, показывающая вчерашнюю свежесть, отвечает не на тот вопрос."""
    _seed(tmp_path / "raw", hours=100)
    client = _client(tmp_path)
    before = client.get("/data").text

    _seed(tmp_path / "raw", hours=HOURS)
    after = client.get("/data").text

    assert ">100<" in before
    assert ">100<" not in after
    assert f">{HOURS}<" in after


def test_holdout_value_never_reaches_the_page(tmp_path: Path) -> None:
    """Резерв WF-3 ценен тем, сколько раз в него смотрели: раскрытие — ADM-5."""
    _seed(tmp_path / "raw")
    out = wf_dir(tmp_path / "results", EXP, "BTCUSDT", "1h")
    out.mkdir(parents=True)
    (out / SELECTIONS_FILE).write_text(
        json.dumps({"selections": [{"fold": 0}], "oos_metrics": {}}), encoding="utf-8"
    )
    (out / HOLDOUT_REPORT_FILE).write_text(
        json.dumps({"holdout_value": HOLDOUT_VALUE}), encoding="utf-8"
    )

    text = _client(tmp_path).get("/").text
    assert str(HOLDOUT_VALUE) not in text
    assert "1.70" not in text
    assert "значение скрыто" in text


def test_pages_have_no_external_addresses(tmp_path: Path) -> None:
    """Админка обязана открываться на машине без интернета: ни CDN, ни шрифтов."""
    _seed(tmp_path / "raw")
    client = _client(tmp_path)

    for url in ("/", "/data"):
        assert re.findall(r'(?:src|href)="(https?://[^"]+)"', client.get(url).text) == []


# --- запуск из CLI ----------------------------------------------------------


def test_busy_port_is_explained_not_traced(tmp_path: Path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind((cli_admin.HOST, 0))
        held.listen(1)
        port = held.getsockname()[1]

        assert cli_admin.port_is_free(port) is False
        with pytest.raises(cli_admin.AdminStartupError, match=f"порт {port} занят"):
            cli_admin.serve(
                _config(tmp_path / "cfg.yaml"),
                data_root=tmp_path / "raw",
                results_root=tmp_path / "results",
                reports_root=tmp_path / "reports",
                port=port,
            )


def test_missing_extra_is_explained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Без экстры CLI должен объяснить, что ставить, а не упасть трейсбеком."""

    def _no_module(name: str) -> ModuleType:
        raise ImportError(name)

    monkeypatch.setattr("trading_research.cli.admin.import_module", _no_module)
    with pytest.raises(cli_admin.AdminStartupError, match='pip install -e ".\\[admin\\]"'):
        cli_admin.load_web()


def test_server_binds_loopback_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Флага --host нет намеренно: снаружи админка не открывается одной опцией."""
    calls: dict[str, object] = {}

    def _fake_import(name: str) -> ModuleType:
        if name == "uvicorn":
            fake = SimpleNamespace(run=lambda app, **kwargs: calls.update(kwargs))
            return cast(ModuleType, fake)
        return import_module(name)

    monkeypatch.setattr("trading_research.cli.admin.import_module", _fake_import)
    _seed(tmp_path / "raw")
    cli_admin.serve(
        _config(tmp_path / "cfg.yaml"),
        data_root=tmp_path / "raw",
        results_root=tmp_path / "results",
        reports_root=tmp_path / "reports",
        port=0,
    )

    assert calls["host"] == "127.0.0.1"
    assert "--host" not in str(cli_admin.serve.__doc__)


def test_cli_reports_startup_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise cli_admin.AdminStartupError("порт 8787 занят")

    monkeypatch.setattr("trading_research.cli.main.serve", _boom)
    result = runner.invoke(
        cli_app,
        [
            "admin",
            "--config",
            str(_config(tmp_path / "cfg.yaml")),
            "--data-root",
            str(tmp_path / "raw"),
        ],
    )

    assert result.exit_code == 1
    assert "порт 8787 занят" in result.stdout
    assert "Traceback" not in result.stdout


def test_base_cli_works_without_web_dependencies() -> None:
    """Экстра не должна протечь в базовую установку: `fetch-data` веб-сервер не нужен.

    Проверяется в отдельном процессе, где импорт `fastapi`/`uvicorn`/`jinja2`
    запрещён: так видно именно импорты на уровне модулей, а не кэш ``sys.modules``
    текущего прогона.
    """
    script = textwrap.dedent(
        """
        import sys

        BANNED = {"fastapi", "uvicorn", "jinja2", "starlette"}


        class Blocker:
            def find_module(self, name, path=None):
                return self.find_spec(name, path)

            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BANNED:
                    raise ImportError(f"{name} запрещён в этой проверке")
                return None


        sys.meta_path.insert(0, Blocker())
        from typer.testing import CliRunner

        from trading_research.cli.main import app

        result = CliRunner().invoke(app, ["--help"])
        assert result.exit_code == 0, result.output
        assert "admin" in result.output
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )

    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
