"""RUN-4 — бизнес-логика команды ``walk-forward``.

Эпик E5 был доступен только из Python: `run_walk_forward` (WF-2),
`generate_walk_forward_folds` (WF-1) и весь `holdout.py` (WF-3) не упоминались ни в
`cli/`, ни в `config.py`. То есть из CLI платформа умела ровно то, что README
называет переоптимизированным поиском (сетка по слайсам), а честный out-of-sample —
принцип №8, «финальная проверка только walk-forward» — был недостижим.

Порядок шагов здесь выражает политику WF-3, а не удобство:

1. **Резерв отрезается первым.** Если задана секция `holdout`, research-данные
   получаются через `HoldoutSplit.research_data`, и сразу после этого
   `assert_research_only` проверяет результат. Проверка избыточна по построению — и
   именно поэтому полезна: она ловит ошибку в самом разрезании, а не доверяет ему.
2. **Фолды строятся только внутри research-диапазона.** Верхняя граница фолдов —
   начало holdout, а не конец данных, иначе test-окно последнего фолда залезло бы
   в резерв.
3. **Holdout считается один раз, в самом конце**, на параметрах, выбранных
   walk-forward (см. `_holdout_params`).

Сравнение в отчёте — `OOS-метрика research` против `holdout`, а не train против
holdout: train-скор заведомо оптимистичен, и сравнивать с ним означало бы измерять
переоптимизацию дважды. Деградация здесь отвечает на вопрос «был ли оптимистичен
даже честный OOS-прогноз».
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from trading_research.config import ExperimentConfig, load_experiment_config
from trading_research.data.storage.duckdb_store import DuckDBStore
from trading_research.experiments.grid_search import expand_grid
from trading_research.experiments.holdout import (
    HoldoutSplit,
    assert_research_only,
    build_holdout_report,
    evaluate_holdout,
    save_holdout_report,
)
from trading_research.experiments.time_splitters import (
    WalkForwardFold,
    generate_walk_forward_folds,
)
from trading_research.experiments.walk_forward import WalkForwardResult, run_walk_forward
from trading_research.strategies import get_strategy

OOS_EQUITY_FILE = "oos_equity.parquet"
SELECTIONS_FILE = "wf_selections.json"
HOLDOUT_REPORT_FILE = "holdout_report.json"


def walk_forward_from_config(
    config_path: str | Path,
    *,
    data_root: str | Path,
    results_root: str | Path,
) -> dict[str, WalkForwardResult]:
    """Прогнать walk-forward (и holdout, если задан) по конфигу.

    Возвращает результат по каждому символу. Артефакты пишутся в
    ``{results_root}/{experiment}/wf/{symbol}/``.

    Raises:
        ValueError: нет секции ``walk_forward``; не получилось ни одного фолда;
            research-выборка пересекается с holdout.
    """
    cfg = load_experiment_config(config_path)
    if cfg.walk_forward is None:
        raise ValueError(
            "config has no 'walk_forward' section; walk-forward requires it "
            "(train_days / test_days / step_days)"
        )

    wf = cfg.walk_forward
    split = build_split(cfg)
    research_end = split.holdout_start if split and split.holdout_start else cfg.data.end

    strategy = get_strategy(cfg.strategy.name)
    combos = expand_grid(cfg.params_grid)
    store = DuckDBStore(data_root)
    out_root = Path(results_root) / cfg.name / "wf"
    results: dict[str, WalkForwardResult] = {}

    for symbol in cfg.data.symbols:
        data = store.read_window(
            exchange=cfg.data.exchange,
            market=cfg.data.market,
            symbol=symbol,
            timeframe=cfg.data.timeframe,
            start=cfg.data.start,
            end=cfg.data.end,
        )
        if data.is_empty():
            continue

        research = split.research_data(data, symbol=symbol) if split else data
        if split is not None:
            # Страховка уровня кода: резерв действительно отрезан (WF-3).
            assert_research_only(research, split, symbol=symbol)
        if research.is_empty():
            # Символ целиком зарезервирован — на research его нет.
            continue

        folds = build_folds(cfg, research_end)
        result = run_walk_forward(
            research,
            strategy,
            combos,
            folds,
            cfg.execution,
            cfg.data.timeframe,
            select_by=wf.select_by,
        )
        results[symbol] = result

        symbol_dir = out_root / symbol
        _save_oos(result, symbol_dir)
        if split is not None:
            _run_holdout(cfg, split, data, symbol=symbol, result=result, out_dir=symbol_dir)

    return results


def build_split(cfg: ExperimentConfig) -> HoldoutSplit | None:
    """Политика резерва из конфига (``None``, если секции ``holdout`` нет)."""
    if cfg.holdout is None:
        return None
    return HoldoutSplit(
        holdout_start=cfg.holdout.start,
        holdout_symbols=frozenset(cfg.holdout.symbols),
    )


def build_folds(cfg: ExperimentConfig, research_end: datetime) -> list[WalkForwardFold]:
    """Фолды в ``[data.start, research_end)`` — выше резерв, туда фолды не заходят."""
    wf = cfg.walk_forward
    if wf is None:  # pragma: no cover — проверено вызывающей стороной
        raise ValueError("config has no 'walk_forward' section")
    folds = generate_walk_forward_folds(
        cfg.data.start,
        research_end,
        train=timedelta(days=wf.train_days),
        test=timedelta(days=wf.test_days),
        step=timedelta(days=wf.step_days),
        warmup_bars=wf.warmup_bars,
        anchored=wf.anchored,
    )
    if not folds:
        raise ValueError(
            f"no walk-forward folds fit into [{cfg.data.start.date()}, {research_end.date()}): "
            f"train={wf.train_days}d + test={wf.test_days}d exceeds the research range"
        )
    return folds


def _holdout_params(result: WalkForwardResult) -> dict[str, object]:
    """Параметры для holdout — выбор **последнего** фолда.

    Это то, что реально «пошло бы в бой»: самый свежий режим рынка. Альтернатива —
    самые частые параметры по фолдам — устойчивее, но отвечает на другой вопрос
    («что работало в среднем»), а holdout проверяет именно вывод исследования.
    """
    return dict(result.selections[-1].params)


def _save_oos(result: WalkForwardResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.oos_equity.write_parquet(out_dir / OOS_EQUITY_FILE)
    payload = [
        {
            "fold": s.fold,
            "params": s.params,
            "train_score": s.train_score,
            "test_metrics": asdict(s.test_metrics),
        }
        for s in result.selections
    ]
    (out_dir / SELECTIONS_FILE).write_text(
        json.dumps(
            {"selections": payload, "oos_metrics": asdict(result.oos_metrics)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_holdout(
    cfg: ExperimentConfig,
    split: HoldoutSplit,
    data: pl.DataFrame,
    *,
    symbol: str,
    result: WalkForwardResult,
    out_dir: Path,
) -> None:
    """Однократная проверка выбранных параметров на резерве (WF-3)."""
    if not result.selections:
        return
    holdout_data = split.holdout_data(data, symbol=symbol)
    if holdout_data.is_empty():
        return

    wf = cfg.walk_forward
    metric = wf.select_by if wf is not None else "total_return_pct"
    params = _holdout_params(result)
    holdout_metrics = evaluate_holdout(
        holdout_data,
        get_strategy(cfg.strategy.name),
        params,
        cfg.execution,
        cfg.data.timeframe,
    )
    research_raw = getattr(result.oos_metrics, metric)
    report = build_holdout_report(
        symbol=symbol,
        params=params,
        research_value=float("nan") if research_raw is None else float(research_raw),
        holdout_metrics=holdout_metrics,
        metric=metric,
    )
    save_holdout_report(report, out_dir / HOLDOUT_REPORT_FILE)


def wf_dir(results_root: str | Path, experiment: str, symbol: str) -> Path:
    """Каталог артефактов walk-forward для символа (для читателей и тестов)."""
    return Path(results_root) / experiment / "wf" / symbol


__all__ = [
    "HOLDOUT_REPORT_FILE",
    "OOS_EQUITY_FILE",
    "SELECTIONS_FILE",
    "build_folds",
    "build_split",
    "walk_forward_from_config",
    "wf_dir",
]
