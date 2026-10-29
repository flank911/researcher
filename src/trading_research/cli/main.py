"""CLI entry point (EXP-6).

Typer-приложение с командами ``fetch-data`` и ``run-experiment`` (раздел 13 плана).
Бизнес-логика вынесена в ``cli/fetch_data.py`` и ``cli/run_experiment.py``, здесь —
только разбор аргументов и вывод.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer

from trading_research import __version__
from trading_research.cli.analyze_results import analyze_results, experiment_results_path
from trading_research.cli.build_features import build_features_from_config
from trading_research.cli.export_results import export_results
from trading_research.cli.fetch_data import fetch_data
from trading_research.cli.fetch_market_data import SeriesKind, fetch_market_data
from trading_research.cli.plot_heatmap import plot_experiment_heatmap
from trading_research.cli.run_experiment import run_experiment_from_config
from trading_research.cli.walk_forward import walk_forward_from_config

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
    for name, rows in written.items():
        typer.echo(f"wrote {rows} rows of {name} for {symbol} {timeframe} -> {data_root}")


@app.command("build-features")
def build_features_cmd(
    config: Path = typer.Option(..., exists=True, help="YAML-конфиг эксперимента."),
    data_root: Path = typer.Option(Path("data/raw"), help="Корень Parquet-store."),
    features_root: Path = typer.Option(Path("data/features"), help="Корень для фич."),
) -> None:
    """Построить таблицу `time_slice_features` (фичи рыночного режима) по конфигу."""
    out = build_features_from_config(config, data_root=data_root, features_root=features_root)
    typer.echo(f"features -> {out}")


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
) -> None:
    """Прогнать walk-forward (честный OOS) и, если задан, holdout."""
    results = walk_forward_from_config(config, data_root=data_root, results_root=results_root)
    if not results:
        typer.echo("no symbols produced walk-forward results")
        return
    for symbol, result in results.items():
        typer.echo(f"\n{symbol}: {len(result.selections)} folds")
        for sel in result.selections:
            typer.echo(f"  fold {sel.fold}: {sel.params} train={sel.train_score:.4f}")
        m = result.oos_metrics
        typer.echo(
            f"  OOS: return={m.total_return_pct:.2f}% dd={m.max_drawdown_pct:.2f}% "
            f"trades={m.trades_count} reliable={m.reliable}"
        )


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


if __name__ == "__main__":
    app()
