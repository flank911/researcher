"""CLI entry point (EXP-6).

Typer-приложение с командами ``fetch-data`` и ``run-experiment`` (раздел 13 плана).
Бизнес-логика вынесена в ``cli/fetch_data.py`` и ``cli/run_experiment.py``, здесь —
только разбор аргументов и вывод.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import typer

from trading_research import __version__
from trading_research.cli.analyze_regimes import (
    analyze_regimes,
    export_regimes,
    regime_availability_lines,
)
from trading_research.cli.analyze_results import analyze_results, experiment_results_path
from trading_research.cli.build_features import (
    build_features_from_config,
    describe_feature_coverage,
    feature_coverage,
)
from trading_research.cli.export_results import export_results
from trading_research.cli.fetch_data import fetch_data
from trading_research.cli.fetch_market_data import (
    CoverageStatus,
    SeriesKind,
    describe_coverage,
    fetch_market_data,
)
from trading_research.cli.plot_heatmap import plot_experiment_heatmap
from trading_research.cli.run_experiment import run_experiment_from_config
from trading_research.cli.update_data import (
    describe_total as describe_candle_total,
)
from trading_research.cli.update_data import (
    describe_update as describe_candle_update,
)
from trading_research.cli.update_data import (
    has_defects as has_candle_defects,
)
from trading_research.cli.update_data import (
    update_data_from_config,
)
from trading_research.cli.update_market_data import (
    describe_total,
    describe_update,
    has_defects,
    update_market_data_from_config,
)
from trading_research.cli.validate_data import (
    describe_pair,
    validate_data_from_config,
)
from trading_research.cli.validate_data import (
    describe_total as describe_validation_total,
)
from trading_research.cli.validate_data import (
    has_defects as has_validation_defects,
)
from trading_research.cli.walk_forward import (
    describe_outcome,
    walk_forward_from_config,
)
from trading_research.cli.walk_forward import (
    describe_total as describe_wf_total,
)

app = typer.Typer(
    add_completion=False,
    help="Trading research platform CLI.",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Trading research platform."""


@app.command()
def version() -> None:
    """Показать версию пакета."""
    typer.echo(__version__)


@app.command("fetch-data")
def fetch_data_cmd(
    symbol: str = typer.Option(..., help="Тикер, например BTCUSDT."),
    timeframe: str = typer.Option(..., help="Таймфрейм, например 1h."),
    start: datetime = typer.Option(..., formats=["%Y-%m-%d"], help="Начало диапазона (UTC)."),
    end: datetime = typer.Option(
        ..., formats=["%Y-%m-%d"], help="Конец диапазона (UTC, эксклюзивно)."
    ),
    exchange: str = typer.Option("binance", help="Биржа (для пути хранения)."),
    market: str = typer.Option("futures", help="Рынок (для пути хранения)."),
    data_root: Path = typer.Option(Path("data/raw"), help="Корень Parquet-store."),
) -> None:
    """Загрузить OHLCV и сохранить в партиционированный Parquet."""
    written = fetch_data(
        exchange=exchange,
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        data_root=data_root,
    )
    typer.echo(f"wrote {written} candles for {symbol} {timeframe} -> {data_root}")


@app.command("fetch-market-data")
def fetch_market_data_cmd(
    symbol: str = typer.Option(..., help="Тикер, например BTCUSDT."),
    timeframe: str = typer.Option(
        ..., help="Интервал: ключ партиции и период агрегации для OI/LS."
    ),
    start: datetime = typer.Option(..., formats=["%Y-%m-%d"], help="Начало диапазона (UTC)."),
    end: datetime = typer.Option(
        ..., formats=["%Y-%m-%d"], help="Конец диапазона (UTC, эксклюзивно)."
    ),
    series: SeriesKind = typer.Option(
        SeriesKind.ALL, help="Ряд: funding / open_interest / long_short_ratio / all."
    ),
    exchange: str = typer.Option("binance", help="Биржа (для пути хранения)."),
    market: str = typer.Option("futures", help="Рынок (для пути хранения)."),
    data_root: Path = typer.Option(Path("data/raw"), help="Корень Parquet-store."),
) -> None:
    """Загрузить funding / open interest / long-short ratio в Parquet-store."""
    written = fetch_market_data(
        exchange=exchange,
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        series=series,
        data_root=data_root,
    )
    for name, coverage in written.items():
        typer.echo(describe_coverage(name, symbol, timeframe, coverage))
    typer.echo(f"-> {data_root}")

    # Неполное покрытие — не успех: молчаливый код 0 здесь и сделал REG-5
    # незаметным на целом прогоне. Данные при этом записаны, см. вывод выше.
    if any(c.status is CoverageStatus.INCOMPLETE for c in written.values()):
        raise typer.Exit(code=1)


@app.command("update-data")
def update_data_cmd(
    config: Path = typer.Option(..., exists=True, help="YAML-конфиг эксперимента."),
    data_root: Path = typer.Option(Path("data/raw"), help="Корень Parquet-store."),
) -> None:
    """Догрузить свечи от последнего сохранённого бара до текущего (OPS-2)."""
    updates = update_data_from_config(config, data_root=data_root)
    for update in updates:
        typer.echo(describe_candle_update(update))
    typer.echo(describe_candle_total(updates))

    # Недобор на догрузке хвоста — дырка в свежей истории. Код возврата нужен,
    # чтобы расписание не отчитывалось об успехе там, где данных не хватает.
    if has_candle_defects(updates):
        raise typer.Exit(code=1)


@app.command("update-market-data")
def update_market_data_cmd(
    config: Path = typer.Option(..., exists=True, help="YAML-конфиг эксперимента."),
    series: SeriesKind = typer.Option(
        SeriesKind.ALL, help="Ряд: funding / open_interest / long_short_ratio / all."
    ),
    data_root: Path = typer.Option(Path("data/raw"), help="Корень Parquet-store."),
) -> None:
    """Догрузить рыночные ряды от последней сохранённой точки (для расписания)."""
    updates = update_market_data_from_config(config, data_root=data_root, series=series)
    for update in updates:
        typer.echo(describe_update(update))
    typer.echo(describe_total(updates))

    # Пропуск или неполное покрытие не должны выглядеть успешным прогоном
    # расписания: иначе провал сбора обнаружится через месяц, когда окно
    # retention уже съело историю.
    if has_defects(updates):
        raise typer.Exit(code=1)


@app.command("validate-data")
def validate_data_cmd(
    config: Path = typer.Option(..., exists=True, help="YAML-конфиг эксперимента."),
    symbol: list[str] = typer.Option(
        None, "--symbol", help="Проверить только эти тикеры (по умолчанию — все из конфига)."
    ),
    timeframe: list[str] = typer.Option(
        None, "--timeframe", help="Проверить только эти таймфреймы (по умолчанию — все)."
    ),
    data_root: Path = typer.Option(Path("data/raw"), help="Корень Parquet-store."),
    out: Path = typer.Option(Path("data/reports"), help="Корень для JSON-отчётов."),
    strict: bool = typer.Option(
        False, "--strict", help="Ненулевой код возврата при найденных дефектах."
    ),
) -> None:
    """Проверить качество загруженных данных (свечи и ряды funding/OI/LS)."""
    pairs = validate_data_from_config(
        config,
        data_root=data_root,
        reports_root=out,
        symbols=list(symbol) if symbol else None,
        timeframes=list(timeframe) if timeframe else None,
    )
    for pair in pairs:
        for line in describe_pair(pair):
            typer.echo(line)
    typer.echo(describe_validation_total(pairs, out))

    # Без --strict команда остаётся диагностикой: дефекты видны, код 0. С ним её
    # можно ставить в пайплайн перед run-experiment — тогда прогон не стартует
    # на данных с дырами.
    if strict and has_validation_defects(pairs):
        raise typer.Exit(code=1)


@app.command("build-features")
def build_features_cmd(
    config: Path = typer.Option(..., exists=True, help="YAML-конфиг эксперимента."),
    data_root: Path = typer.Option(Path("data/raw"), help="Корень Parquet-store."),
    features_root: Path = typer.Option(Path("data/features"), help="Корень для фич."),
) -> None:
    """Построить таблицу `time_slice_features` (фичи рыночного режима) по конфигу."""
    out = build_features_from_config(config, data_root=data_root, features_root=features_root)
    typer.echo(f"features -> {out}")
    # Пустая фича иначе проходит незаметно и всплывает только в корреляциях —
    # уже как «связи нет» вместо «данных нет» (REG-4).
    for line in describe_feature_coverage(feature_coverage(pl.read_parquet(out))):
        typer.echo(line)


@app.command("run-experiment")
def run_experiment_cmd(
    config: Path = typer.Option(..., exists=True, help="YAML-конфиг эксперимента."),
    data_root: Path = typer.Option(Path("data/raw"), help="Корень Parquet-store."),
    results_root: Path = typer.Option(Path("data/results"), help="Корень для результатов."),
    workers: int = typer.Option(0, help="Процессов для сетки (0 — по числу ядер, 1 — без пула)."),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Продолжить прогон, пропуская готовые раны."
    ),
) -> None:
    """Прогнать эксперимент по конфигу и записать пер-слайс метрики."""
    out = run_experiment_from_config(
        config,
        data_root=data_root,
        results_root=results_root,
        workers=workers or None,
        resume=resume,
    )
    typer.echo(f"results -> {out}")


@app.command("walk-forward")
def walk_forward_cmd(
    config: Path = typer.Option(..., exists=True, help="YAML-конфиг эксперимента."),
    data_root: Path = typer.Option(Path("data/raw"), help="Корень Parquet-store."),
    results_root: Path = typer.Option(Path("data/results"), help="Корень для результатов."),
    workers: int = typer.Option(
        1, help="Процессов на отбор параметров (1 — без пула, 0 — по числу ядер)."
    ),
) -> None:
    """Прогнать walk-forward (честный OOS) и, если задан, holdout."""
    outcomes = walk_forward_from_config(
        config, data_root=data_root, results_root=results_root, workers=workers
    )
    # Holdout печатается вместе с OOS, а его отсутствие — отдельной причиной:
    # главный результат исследования не должен жить только в JSON (RUN-9).
    for outcome in outcomes.values():
        typer.echo("")
        for line in describe_outcome(outcome):
            typer.echo(line)
    typer.echo("")
    typer.echo(describe_wf_total(list(outcomes.values())))


@app.command("analyze-results")
def analyze_results_cmd(
    experiment: str = typer.Option(..., help="Имя эксперимента (имя из конфига)."),
    metric: str = typer.Option("total_return_pct", help="Метрика ранжирования."),
    top: int = typer.Option(20, help="Сколько лучших комбинаций вывести."),
    by: str = typer.Option("", help="Разрез: time / symbol / timeframe (пусто — глобально)."),
    symbol: str = typer.Option("", help="Фильтр по символу."),
    timeframe: str = typer.Option("", help="Фильтр по таймфрейму."),
    results_root: Path = typer.Option(Path("data/results"), help="Корень результатов."),
) -> None:
    """Вывести top-N параметров по метрике из результатов эксперимента."""
    ranked = analyze_results(
        experiment_results_path(results_root, experiment),
        metric=metric,
        top=top,
        by=by or None,
        symbol=symbol or None,
        timeframe=timeframe or None,
    )
    typer.echo(str(ranked))


@app.command("plot-heatmap")
def plot_heatmap_cmd(
    experiment: str = typer.Option(..., help="Имя эксперимента (имя из конфига)."),
    x: str = typer.Option(..., help="Параметр по оси X (ключ из params)."),
    y: str = typer.Option(..., help="Параметр по оси Y (ключ из params)."),
    metric: str = typer.Option("total_return_pct", help="Метрика (цвет ячеек)."),
    agg: str = typer.Option("median", help="Агрегат: median/mean/min/max."),
    symbol: str = typer.Option("", help="Фильтр по символу."),
    timeframe: str = typer.Option("", help="Фильтр по таймфрейму."),
    time_slice: str = typer.Option("", help="Фильтр по time_slice_id (напр. год/месяц)."),
    out: str = typer.Option("", help="Путь к PNG (по умолчанию в каталоге эксперимента)."),
    results_root: Path = typer.Option(Path("data/results"), help="Корень результатов."),
) -> None:
    """Построить PNG-heatmap ``x × y`` по метрике из результатов эксперимента."""
    default_out = results_root / experiment / f"heatmap_{x}_{y}_{metric}.png"
    saved = plot_experiment_heatmap(
        experiment_results_path(results_root, experiment),
        Path(out) if out else default_out,
        x=x,
        y=y,
        metric=metric,
        agg=agg,
        symbol=symbol or None,
        timeframe=timeframe or None,
        time_slice=time_slice or None,
    )
    typer.echo(f"heatmap -> {saved}")


@app.command("export-results")
def export_results_cmd(
    experiment: str = typer.Option(..., help="Имя эксперимента (имя из конфига)."),
    kind: str = typer.Option(
        "performance", help="Что экспортировать: performance/ranking/stability."
    ),
    fmt: str = typer.Option("csv", "--format", help="Формат: csv/parquet."),
    metric: str = typer.Option("total_return_pct", help="Метрика (для ranking/stability)."),
    top: int = typer.Option(20, help="Top-N для ranking."),
    out: str = typer.Option("", help="Путь к файлу (по умолчанию в каталоге эксперимента)."),
    results_root: Path = typer.Option(Path("data/results"), help="Корень результатов."),
) -> None:
    """Экспортировать сводную таблицу эксперимента в CSV/Parquet."""
    default_out = results_root / experiment / f"export_{kind}.{fmt}"
    saved = export_results(
        experiment_results_path(results_root, experiment),
        Path(out) if out else default_out,
        kind=kind,
        fmt=fmt,
        metric=metric,
        top=top,
    )
    typer.echo(f"export -> {saved}")


@app.command("analyze-regimes")
def analyze_regimes_cmd(
    experiment: str = typer.Option(..., help="Имя эксперимента (имя из конфига)."),
    metric: str = typer.Option("total_return_pct", help="Метрика для корреляций."),
    regime: str = typer.Option(
        "", help="Режим (напр. bull_trend) — тогда выводится top-N параметров в нём."
    ),
    top: int = typer.Option(20, help="Top-N для режима."),
    global_only: bool = typer.Option(
        False, "--global", help="Одна строка общей корреляции вместо матрицы по параметрам."
    ),
    out: str = typer.Option("", help="Выгрузить результат в файл (CSV/Parquet)."),
    fmt: str = typer.Option("csv", "--format", help="Формат выгрузки: csv/parquet."),
    results_root: Path = typer.Option(Path("data/results"), help="Корень результатов."),
    features_root: Path = typer.Option(Path("data/features"), help="Корень фич."),
) -> None:
    """Корреляции параметров с рыночным режимом (REG-3) по сохранённым таблицам."""
    table = analyze_regimes(
        results_root,
        features_root,
        experiment,
        metric=metric,
        regime=regime or None,
        top=top,
        by_param=not global_only,
    )
    # Матрица режима — 10 колонок; дефолтный вывод polars их обрезает, а смысл
    # команды именно в том, чтобы показать все фичи сразу.
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        typer.echo(str(table))
    # Без этой пометки `null` от пустой фичи читается как «связи нет» (REG-4).
    if not regime:
        for line in regime_availability_lines(results_root, features_root, experiment):
            typer.echo(line)
    if out:
        saved = export_regimes(
            results_root,
            features_root,
            experiment,
            out,
            metric=metric,
            regime=regime or None,
            top=top,
            fmt=fmt,
        )
        typer.echo(f"export -> {saved}")


if __name__ == "__main__":
    app()
