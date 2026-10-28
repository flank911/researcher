"""AN-5 — ноутбук 001 исполняется сверху донизу без правок (на синтетике)."""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "001_explore_results.ipynb"


def test_notebook_exists_and_valid() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    assert any(c.cell_type == "code" for c in nb.cells)


def test_notebook_runs_without_edits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Без TR_RESULTS ноутбук использует синтетические данные.
    monkeypatch.delenv("TR_RESULTS", raising=False)
    monkeypatch.chdir(tmp_path)

    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n\n".join(c.source for c in nb.cells if c.cell_type == "code")
    namespace: dict[str, object] = {"__name__": "__main__"}
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)  # noqa: S102 — доверенный локальный ноутбук

    # Последняя ячейка строит heatmap и кладёт путь в `out`.
    out = namespace["out"]
    assert isinstance(out, Path)
    assert out.exists()
    assert out.read_bytes()[:4] == b"\x89PNG"
