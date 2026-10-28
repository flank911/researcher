"""AN-5 — бизнес-логика команды ``export-results``.

Экспорт сводных таблиц из результатов эксперимента (таблица AN-1) в CSV/Parquet:
саму таблицу (``performance``), рейтинг параметров (``ranking``, AN-2) или вектор
стабильности (``stability``, AN-3).
"""

from __future__ import annotations

from pathlib import Path

from trading_research.analytics.queries import load_performance_table, rank_params
from trading_research.analytics.stability import stability_vector

FORMATS = ("csv", "parquet")
KINDS = ("performance", "ranking", "stability")


def export_results(
    results_path: str | Path,
    out_path: str | Path,
    *,
    kind: str = "performance",
    fmt: str = "csv",
    metric: str = "total_return_pct",
    top: int = 20,
) -> Path:
    """Экспортировать выбранную сводную таблицу в CSV/Parquet.

    ``kind`` — ``performance`` (вся таблица) / ``ranking`` (top-N по ``metric``) /
    ``stability`` (вектор устойчивости). Возвращает путь к файлу.
    """
    if fmt not in FORMATS:
        raise ValueError(f"fmt {fmt!r} unknown; expected one of {list(FORMATS)}")
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} unknown; expected one of {list(KINDS)}")

    table = load_performance_table(results_path)
    if kind == "performance":
        df = table
    elif kind == "ranking":
        df = rank_params(table, metric=metric, top=top)
    else:
        df = stability_vector(table, return_metric=metric)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.write_csv(out)
    else:
        df.write_parquet(out)
    return out


__all__ = ["FORMATS", "KINDS", "export_results"]
