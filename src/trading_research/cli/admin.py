"""ADM-3 — запуск веб-админки из CLI.

Два решения, которые важнее механики запуска.

**Веб-зависимости грузятся лениво.** ``fastapi`` и ``uvicorn`` живут в отдельной
экстре ``admin``, и импорт их на уровне модуля сломал бы весь CLI для тех, кто
поставил платформу без неё: ``fetch-data`` и ``run-experiment`` веб-сервер не нужен.
Поэтому приложение импортируется внутри вызова, а отсутствие экстры — это понятное
сообщение, а не трейсбек.

**Адрес не настраивается.** Админка слушает только ``127.0.0.1`` и аутентификации не
имеет — одно следует из другого. Флага ``--host`` нет намеренно: он позволил бы
открыть снаружи интерфейс, где видно всё состояние исследования, одной опцией и без
предупреждения. Нужен доступ с другой машины — это SSH-туннель, то есть решение,
принятое осознанно.
"""

from __future__ import annotations

import socket
from importlib import import_module
from pathlib import Path
from types import ModuleType

HOST = "127.0.0.1"
"""Админка слушает только петлю: аутентификации у неё нет (см. докстринг модуля)."""

DEFAULT_PORT = 8787

WEB_EXTRA_HINT = (
    'веб-зависимости не установлены. Админка живёт в отдельной экстре: pip install -e ".[admin]"'
)


class AdminStartupError(RuntimeError):
    """Админку не удалось запустить, и причина объяснена человеку, а не трейсбеком."""


def load_web() -> ModuleType:
    """Импортировать веб-слой; без экстры — понятная ошибка.

    Raises:
        AdminStartupError: экстра ``admin`` не установлена.
    """
    try:
        return import_module("trading_research.admin.web")
    except ImportError as exc:  # pragma: no cover — ветка проверяется подменой import_module
        raise AdminStartupError(WEB_EXTRA_HINT) from exc


def port_is_free(port: int, *, host: str = HOST) -> bool:
    """Свободен ли порт. Проверяется до старта — иначе uvicorn падает трейсбеком."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def serve(
    config_path: str | Path,
    *,
    data_root: str | Path,
    results_root: str | Path,
    reports_root: str | Path,
    log_path: str | Path | None = None,
    port: int = DEFAULT_PORT,
) -> None:
    """Поднять админку на ``127.0.0.1:<port>``.

    Raises:
        AdminStartupError: нет экстры ``admin`` или порт занят.
    """
    web = load_web()
    if not port_is_free(port):
        raise AdminStartupError(
            f"порт {port} занят — закройте то, что его держит, или задайте --port"
        )

    roots = web.Roots(
        config=Path(config_path),
        data=Path(data_root),
        results=Path(results_root),
        reports=Path(reports_root),
        log=Path(log_path) if log_path is not None else None,
    )
    uvicorn = import_module("uvicorn")
    uvicorn.run(web.create_app(roots), host=HOST, port=port, log_level="warning")


def describe_start(port: int) -> str:
    """Строка, с которой начинается работа: адрес и то, что админка не умеет."""
    return f"админка (только чтение) -> http://{HOST}:{port}"


__all__ = [
    "DEFAULT_PORT",
    "HOST",
    "WEB_EXTRA_HINT",
    "AdminStartupError",
    "describe_start",
    "load_web",
    "port_is_free",
    "serve",
]
