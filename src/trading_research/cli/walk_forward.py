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

**RUN-9 — результат должен быть виден.** Политика выше была реализована верно, но
holdout считался молча: отчёты писались в
``data/results/<experiment>/wf/<symbol>/<timeframe>/``, а в выводе команды слова
«holdout» не было вовсе. На прогоне RUN-7 это скрыло главный результат
исследования: по BTC 1h research-OOS sharpe 0.07 против holdout −1.70, то есть
даже честный OOS оказался оптимистичным. Поэтому прогон возвращает не только
`WalkForwardResult`, но и `PairOutcome` — с отчётом holdout и, когда его нет, с
причиной (`HoldoutStatus`). Отсутствие holdout выражается строкой, а не тишиной:
раньше «нет секции в конфиге», «резерв пуст» и «символ целиком в резерве»
выглядели одинаково, причём последний случай вовсе убирал пару из вывода.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

import polars as pl

from trading_research.cli.text import plural
from trading_research.config import ExperimentConfig, load_experiment_config
from trading_research.data.storage.duckdb_store import DuckDBStore
from trading_research.experiments.grid_search import expand_grid
from trading_research.experiments.holdout import (
    HoldoutReport,
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


class HoldoutStatus(StrEnum):
    """Почему по паре есть или нет отчёта holdout (RUN-9).

    Три нижних значения раньше выглядели в выводе одинаково — тишиной, — хотя
    означают разное: от «резерв не задан» до «символ целиком зарезервирован, и
    research-данных по нему нет вовсе».
    """

    DONE = "done"
    NO_CONFIG = "no_config"  # в конфиге нет секции holdout
    EMPTY_RESERVE = "empty_reserve"  # резерв по этой паре пуст
    NO_SELECTION = "no_selection"  # walk-forward не выбрал параметров
    SYMBOL_RESERVED = "symbol_reserved"  # символ целиком в резерве
    RESEARCH_EMPTY = "research_empty"  # весь диапазон пары попал в резерв
    NO_DATA = "no_data"  # данных по паре нет в окне конфига


@dataclass(frozen=True)
class PairOutcome:
    """Итог прогона по одной паре символ × таймфрейм.

    ``result`` равен ``None``, когда walk-forward не прогонялся вовсе: раньше такая
    пара просто исчезала из результата, и отличить «её не считали» от «её не было в
    конфиге» было нельзя.
    """

    symbol: str
    timeframe: str
    holdout_status: HoldoutStatus
    result: WalkForwardResult | None = None
    holdout: HoldoutReport | None = None
    artifacts: Path | None = None

    @property
    def was_run(self) -> bool:
        return self.result is not None

    @property
    def is_unreliable_holdout(self) -> bool:
        """Holdout посчитан, но сделок меньше порога — число убедительнее, чем есть."""
        return self.holdout is not None and not self.holdout.holdout_metrics.reliable


def walk_forward_from_config(
    config_path: str | Path,
    *,
    data_root: str | Path,
    results_root: str | Path,
    workers: int = 1,
) -> dict[tuple[str, str], PairOutcome]:
    """Прогнать walk-forward (и holdout, если задан) по конфигу.

    Возвращает итог по каждой паре ``(символ, таймфрейм)``, **включая пары, по
    которым прогон не состоялся** — с причиной в ``holdout_status`` (RUN-9).
    Артефакты пишутся в ``{results_root}/{experiment}/wf/{symbol}/{timeframe}/``.

    ``workers`` — процессов на отбор параметров на train (RUN-8): ``1`` — в текущем
    процессе, ``0`` — по числу ядер. На результат не влияет.

    RUN-6: таймфрейм входит и в ключ результата, и в путь артефактов — иначе прогоны
    одного символа на 1h и 4h перезаписывали бы друг друга.

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
    results: dict[tuple[str, str], PairOutcome] = {}
    no_holdout = HoldoutStatus.NO_CONFIG if split is None else HoldoutStatus.EMPTY_RESERVE

    for symbol in cfg.data.symbols:
        for timeframe in cfg.data.timeframes:
            data = store.read_window(
                exchange=cfg.data.exchange,
                market=cfg.data.market,
                symbol=symbol,
                timeframe=timeframe,
                start=cfg.data.start,
                end=cfg.data.end,
            )
            if data.is_empty():
                results[symbol, timeframe] = PairOutcome(
                    symbol=symbol, timeframe=timeframe, holdout_status=HoldoutStatus.NO_DATA
                )
                continue

            research = split.research_data(data, symbol=symbol) if split else data
            if split is not None:
                # Страховка уровня кода: резерв действительно отрезан (WF-3).
                assert_research_only(research, split, symbol=symbol)
            if research.is_empty():
                # Research-данных нет: символ целиком в резерве либо весь диапазон
                # пары лежит правее holdout_start. Причины разные, и обе должны
                # дойти до вывода — раньше пара просто исчезала.
                reserved = split is not None and symbol in split.holdout_symbols
                results[symbol, timeframe] = PairOutcome(
                    symbol=symbol,
                    timeframe=timeframe,
                    holdout_status=(
                        HoldoutStatus.SYMBOL_RESERVED if reserved else HoldoutStatus.RESEARCH_EMPTY
                    ),
                )
                continue

            folds = build_folds(cfg, research_end)
            result = run_walk_forward(
                research,
                strategy,
                combos,
                folds,
                cfg.execution,
                timeframe,
                select_by=wf.select_by,
                workers=workers,
            )
            out_dir = out_root / symbol / timeframe
            _save_oos(result, out_dir)
            report, status = (
                _run_holdout(
                    cfg,
                    split,
                    data,
                    symbol=symbol,
                    timeframe=timeframe,
                    result=result,
                    out_dir=out_dir,
                )
                if split is not None
                else (None, no_holdout)
            )
            results[symbol, timeframe] = PairOutcome(
                symbol=symbol,
                timeframe=timeframe,
                holdout_status=status,
                result=result,
                holdout=report,
                artifacts=out_dir,
            )

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
    timeframe: str,
    result: WalkForwardResult,
    out_dir: Path,
) -> tuple[HoldoutReport | None, HoldoutStatus]:
    """Однократная проверка выбранных параметров на резерве (WF-3).

    Возвращает отчёт либо причину его отсутствия: раньше оба отказа были тихим
    ``return``, и снаружи их нельзя было отличить от «holdout не задан» (RUN-9).
    """
    if not result.selections:
        return None, HoldoutStatus.NO_SELECTION
    holdout_data = split.holdout_data(data, symbol=symbol)
    if holdout_data.is_empty():
        return None, HoldoutStatus.EMPTY_RESERVE

    wf = cfg.walk_forward
    metric = wf.select_by if wf is not None else "total_return_pct"
    params = _holdout_params(result)
    holdout_metrics = evaluate_holdout(
        holdout_data,
        get_strategy(cfg.strategy.name),
        params,
        cfg.execution,
        timeframe,
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
    return report, HoldoutStatus.DONE


# Почему пара вовсе не прогонялась — текст для вывода.
_SKIP_REASONS: dict[HoldoutStatus, str] = {
    HoldoutStatus.NO_DATA: "нет данных в окне конфига",
    HoldoutStatus.SYMBOL_RESERVED: "символ целиком в резерве holdout, research-данных нет",
    HoldoutStatus.RESEARCH_EMPTY: "весь диапазон пары правее начала holdout, research пуст",
}

# Почему holdout не считался, хотя walk-forward прошёл.
_ABSENT_HOLDOUT: dict[HoldoutStatus, str] = {
    HoldoutStatus.NO_CONFIG: "в конфиге нет секции holdout, резерв не задан",
    HoldoutStatus.EMPTY_RESERVE: "резерв по этой паре пуст",
    HoldoutStatus.NO_SELECTION: "walk-forward не выбрал параметров ни на одном фолде",
}


def describe_outcome(outcome: PairOutcome) -> list[str]:
    """Строки отчёта по одной паре: фолды, OOS и holdout.

    Holdout печатается сразу за OOS, потому что для исследования он важнее: OOS
    отвечает «сколько заработала бы выбранная параметризация», а holdout — «был ли
    оптимистичен сам этот ответ».
    """
    head = f"{outcome.symbol} {outcome.timeframe}"
    if outcome.result is None:
        reason = _SKIP_REASONS.get(outcome.holdout_status, "прогон не состоялся")
        return [f"{head}: пропущен — {reason}"]

    lines = [f"{head}: {len(outcome.result.selections)} folds"]
    lines += [
        f"  fold {sel.fold}: {sel.params} train={sel.train_score:.4f}"
        for sel in outcome.result.selections
    ]
    m = outcome.result.oos_metrics
    lines.append(
        f"  OOS: return={m.total_return_pct:.2f}% dd={m.max_drawdown_pct:.2f}% "
        f"trades={m.trades_count} reliable={m.reliable}"
    )
    lines.append(f"  {describe_holdout(outcome)}")
    if outcome.artifacts is not None:
        lines.append(f"  артефакты -> {outcome.artifacts}")
    return lines


def describe_holdout(outcome: PairOutcome) -> str:
    """Строка holdout — или объяснение, почему его нет."""
    report = outcome.holdout
    if report is None:
        reason = _ABSENT_HOLDOUT.get(outcome.holdout_status, "не задан")
        return f"holdout: не считался — {reason}"

    metrics = report.holdout_metrics
    line = (
        f"holdout: {report.metric} research={_fmt_metric(report.research_value, report.metric)}"
        f" → holdout={_fmt_metric(report.holdout_value, report.metric)}, "
        f"{_fmt_degradation(report.degradation, report.metric)}, "
        f"сделок {metrics.trades_count}, параметры {report.params}"
    )
    # Два лучших holdout-результата прогона RUN-7 получены на 6 сделках: sharpe
    # 2.47 на шести сделках выглядит убедительнее, чем есть.
    if not metrics.reliable:
        line += f"  ← НЕНАДЁЖНО: сделок меньше порога {metrics.min_trades}"
    return line


def describe_total(outcomes: list[PairOutcome]) -> str:
    """Сводка по всем парам — на двадцати парах отдельные строки уже не читаются."""
    if not outcomes:
        return "итого: ни одной пары в конфиге"
    ran = [o for o in outcomes if o.was_run]
    if not ran:
        return f"итого: ни одна из {plural(len(outcomes), 'пары', 'пар', 'пар')} не прогналась"

    parts = [f"итого: прогнано {plural(len(ran), 'пара', 'пары', 'пар')}"]
    if skipped := [o for o in outcomes if not o.was_run]:
        parts.append(f"пропущено {len(skipped)}")

    scored = [o for o in ran if o.holdout is not None]
    if not scored:
        parts.append("holdout не считался ни по одной паре")
        return ", ".join(parts)

    # NaN-деградация (метрика не определена) сюда не попадает: сравнение с NaN ложно.
    worse = [o for o in scored if o.holdout is not None and o.holdout.degradation > 0]
    parts.append(f"holdout посчитан по {len(scored)}")
    parts.append(f"хуже research на {len(worse)} из {len(scored)}")
    if unreliable := [o for o in scored if o.is_unreliable_holdout]:
        parts.append(f"ненадёжных по числу сделок {len(unreliable)}")
    return ", ".join(parts)


def _fmt_metric(value: float, metric: str) -> str:
    """Проценты — с двумя знаками, прочие метрики (sharpe и т. п.) — с четырьмя."""
    if value != value:  # NaN: метрика не определена на этих данных
        return "—"
    return f"{value:.2f}%" if metric.endswith("_pct") else f"{value:.4f}"


def _fmt_degradation(value: float, metric: str) -> str:
    """Знак здесь — главное, что читают, поэтому он назван словом, а не минусом.

    ``degradation = research - holdout``: положительная величина означает ухудшение на
    резерве, отрицательная — что резерв оказался лучше research-OOS.
    """
    if value != value:
        return "деградация неизвестна (метрика не определена)"
    if value > 0:
        return f"деградация {_fmt_metric(value, metric)}"
    if value < 0:
        return f"прирост {_fmt_metric(-value, metric)} (на резерве лучше)"
    return "без изменения"


def wf_dir(results_root: str | Path, experiment: str, symbol: str, timeframe: str) -> Path:
    """Каталог артефактов walk-forward для пары символ × таймфрейм."""
    return Path(results_root) / experiment / "wf" / symbol / timeframe


__all__ = [
    "HOLDOUT_REPORT_FILE",
    "OOS_EQUITY_FILE",
    "SELECTIONS_FILE",
    "HoldoutStatus",
    "PairOutcome",
    "build_folds",
    "build_split",
    "describe_holdout",
    "describe_outcome",
    "describe_total",
    "walk_forward_from_config",
    "wf_dir",
]
