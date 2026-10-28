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
from trading_research.cli.fetch_data import fetch_data
from trading_research.cli.run_experiment import run_experiment_from_config

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


@app.command("run-experiment")
def run_experiment_cmd(
    config: Path = typer.Option(..., exists=True, help="YAML-конфиг эксперимента."),
    data_root: Path = typer.Option(Path("data/raw"), help="Корень Parquet-store."),
    results_root: Path = typer.Option(Path("data/results"), help="Корень для результатов."),
) -> None:
    """Прогнать эксперимент по конфигу и записать пер-слайс метрики."""
    out = run_experiment_from_config(config, data_root=data_root, results_root=results_root)
    typer.echo(f"results -> {out}")


if __name__ == "__main__":
    app()
