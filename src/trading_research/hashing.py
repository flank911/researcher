"""INF-3 — стабильный детерминированный хэш и утилиты воспроизводимости.

Встроенный ``hash()`` для строк рандомизирован между процессами (``PYTHONHASHSEED``),
поэтому он непригоден для ``run_hash`` / cache / resume. Здесь — ``sha256`` от
канонического JSON с нормализацией float, datetime и порядка ключей.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from typing import Any

_FLOAT_SIG_DIGITS = 12


def _canonicalize(obj: Any) -> Any:
    """Привести объект к стабильной, сериализуемой форме.

    - dict -> отсортированный по ключам dict;
    - float -> нормализованное представление (убирает -0.0 и шум последних разрядов);
    - datetime/date -> ISO-строка;
    - Enum -> его value;
    - объекты с ``model_dump`` (pydantic) / ``__dict__`` -> dict.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        normalized = float(f"{obj:.{_FLOAT_SIG_DIGITS}g}")
        return 0.0 if normalized == 0.0 else normalized
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return _canonicalize(obj.value)
    if isinstance(obj, Mapping):
        return {str(k): _canonicalize(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)) or (
        isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))
    ):
        return [_canonicalize(v) for v in obj]
    # pydantic v2 model
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return _canonicalize(model_dump(mode="python"))
    if hasattr(obj, "__dict__"):
        return _canonicalize(vars(obj))
    return str(obj)


def canonical_json(obj: Any) -> str:
    """Каноническая JSON-строка: отсортированные ключи, без лишних пробелов."""
    return json.dumps(
        _canonicalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def stable_hash(obj: Any, *, length: int = 16) -> str:
    """Стабильный sha256-хэш объекта.

    Args:
        obj: произвольный сериализуемый объект (включая pydantic-модели).
        length: длина возвращаемого hex-префикса (полный хэш при ``length<=0``).
    """
    digest = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    return digest if length <= 0 else digest[:length]


def git_commit_hash(*, short: bool = False) -> str | None:
    """Текущий git commit hash или ``None`` вне git-репозитория."""
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    commit = out.stdout.strip()
    return commit or None
