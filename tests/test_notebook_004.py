"""RUN-7 — ноутбук 004 исполняется сверху донизу без артефактов прогона.

Отчёт по первому реальному исследованию читается как текст (числа и выводы в
markdown-ячейках), а код оживает только при наличии `data/`. Тест фиксирует
именно это: без артефактов ноутбук не падает, а сообщает, какие шаги прогнать.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "004_first_research.ipynb"


def test_notebook_exists_and_valid() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    assert any(c.cell_type == "code" for c in nb.cells)


def test_notebook_runs_without_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for var in ("TR_RAW", "TR_RESULTS", "TR_FEATURES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)

    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n\n".join(c.source for c in nb.cells if c.cell_type == "code")
    namespace: dict[str, object] = {"__name__": "__main__"}
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)  # noqa: S102 — доверенный ноутбук

    assert namespace["HAVE_PERFORMANCE"] is False
    out = capsys.readouterr().out
    # Каждая аналитическая ячейка назвала команду, которой её оживить.
    assert "trading-research run-experiment" in out
    assert "trading-research walk-forward" in out
    assert "trading-research build-features" in out
