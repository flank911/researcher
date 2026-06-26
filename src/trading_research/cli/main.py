"""CLI entry point.

Команды реализуются в тикетах E1+ (EXP-6 и далее). Здесь — каркас Typer-приложения,
чтобы entry point ``trading-research`` резолвился сразу после установки.
"""

from __future__ import annotations

import typer

from trading_research import __version__

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


if __name__ == "__main__":
    app()
