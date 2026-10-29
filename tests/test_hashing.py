"""Tests for INF-3 — stable hashing, and INF-5 — детерминизм канонизации."""

from __future__ import annotations

import math
import subprocess
import sys
from datetime import datetime
from enum import Enum, StrEnum
from pathlib import Path

import pytest
from pydantic import BaseModel

from trading_research.hashing import canonical_json, git_commit_hash, stable_hash

# Подпроцесс считает один хэш — секунды с запасом на холодный импорт. Лимит нужен,
# чтобы зависший ребёнок падал по таймауту, а не держал весь прогон до лимита джобы
# (INF-7). ``git_commit_hash`` в проде ограничен тем же способом.
_SUBPROCESS_TIMEOUT_S = 60


def _hash_in_subprocess(expr: str, runs: int = 3) -> list[str]:
    """Посчитать ``stable_hash(expr)`` в отдельных процессах (разный PYTHONHASHSEED)."""
    code = f"from trading_research.hashing import stable_hash;print(stable_hash({expr}))"
    return [
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
        ).stdout.strip()
        for _ in range(runs)
    ]


def test_key_order_does_not_matter() -> None:
    a = {"x": 1, "y": 2, "z": 3}
    b = {"z": 3, "y": 2, "x": 1}
    assert stable_hash(a) == stable_hash(b)


def test_different_input_changes_hash() -> None:
    assert stable_hash({"fast_ma": 20}) != stable_hash({"fast_ma": 21})


def test_float_noise_normalized() -> None:
    assert stable_hash({"v": 0.1 + 0.2}) == stable_hash({"v": 0.30000000000000004})
    assert stable_hash({"v": 0.0}) == stable_hash({"v": -0.0})


def test_datetime_canonicalized() -> None:
    payload = {"start": datetime(2024, 1, 1)}
    assert "2024-01-01T00:00:00" in canonical_json(payload)


def test_hash_length() -> None:
    assert len(stable_hash({"a": 1})) == 16
    assert len(stable_hash({"a": 1}, length=0)) == 64


def test_stable_across_processes() -> None:
    """hash() рандомизирован между процессами — наш stable_hash не должен быть."""
    params = {"strategy": "ma_cross", "fast_ma": 20, "slow_ma": 100}
    runs = _hash_in_subprocess(repr(params), runs=2)
    assert runs[0] == runs[1]
    assert runs[0] == stable_hash(params)


# --- INF-3: нормализация особых float ---


def test_nan_normalized() -> None:
    assert canonical_json({"v": math.nan}) == '{"v":"NaN"}'
    # NaN != NaN, но канонизация делает их одинаковыми.
    assert stable_hash({"v": math.nan}) == stable_hash({"v": float("nan")})


def test_infinities_normalized_and_distinct() -> None:
    assert canonical_json({"v": math.inf}) == '{"v":"Infinity"}'
    assert canonical_json({"v": -math.inf}) == '{"v":"-Infinity"}'
    assert stable_hash({"v": math.inf}) != stable_hash({"v": -math.inf})


# --- INF-3: энумы, pydantic, обычные объекты ---


def test_str_enum_serializes_to_value() -> None:
    """StrEnum ловит проверка на str; значение всё равно каноническое."""

    class Kind(StrEnum):
        MONTHLY = "monthly"

    assert canonical_json({"kind": Kind.MONTHLY}) == '{"kind":"monthly"}'
    assert stable_hash({"kind": Kind.MONTHLY}) == stable_hash({"kind": "monthly"})


def test_non_string_enum_uses_enum_branch() -> None:
    """Не-строковый Enum идёт через ветку Enum и канонизируется по value."""

    class Mode(Enum):
        PAIR = (1, 2)

    assert canonical_json({"m": Mode.PAIR}) == '{"m":[1,2]}'


def test_pydantic_model_canonicalized() -> None:
    class Params(BaseModel):
        slow_ma: int
        fast_ma: int

    # Порядок полей модели не влияет — ключи сортируются.
    assert stable_hash(Params(slow_ma=100, fast_ma=20)) == stable_hash(
        {"fast_ma": 20, "slow_ma": 100}
    )


def test_plain_object_canonicalized_via_dict() -> None:
    class Spec:
        def __init__(self) -> None:
            self.fast_ma = 20

    assert stable_hash(Spec()) == stable_hash({"fast_ma": 20})


def test_nested_sequences_canonicalized() -> None:
    """tuple и list неразличимы после канонизации."""
    assert stable_hash({"grid": (1, 2, 3)}) == stable_hash({"grid": [1, 2, 3]})


# --- INF-5: множества ---


def test_set_order_does_not_matter() -> None:
    assert stable_hash({"tags": {"alpha", "beta"}}) == stable_hash({"tags": {"beta", "alpha"}})


def test_set_and_frozenset_agree() -> None:
    assert stable_hash({"t": {"a", "b"}}) == stable_hash({"t": frozenset({"b", "a"})})


def test_set_canonicalized_to_sorted_list() -> None:
    assert canonical_json({"t": {"beta", "alpha"}}) == '{"t":["alpha","beta"]}'


def test_set_of_strings_stable_across_processes() -> None:
    """Регрессия INF-5: порядок итерации set строк зависит от PYTHONHASHSEED.

    До фикса множество уходило в ``str(obj)`` и три процесса давали три разных хэша.
    """
    runs = _hash_in_subprocess("{'tags': {'alpha','beta','gamma','delta'}}")
    assert len(set(runs)) == 1, f"hash unstable across processes: {runs}"
    assert runs[0] == stable_hash({"tags": {"alpha", "beta", "gamma", "delta"}})


def test_mixed_type_set_canonicalized() -> None:
    """Разнотипное множество не падает на сравнении несравнимых типов."""
    assert stable_hash({"t": {1, "a", None}}) == stable_hash({"t": {None, "a", 1}})


# --- INF-5: bytes и громкий отказ ---


def test_bytes_canonicalized_to_hex() -> None:
    assert canonical_json({"b": b"abc"}) == '{"b":"616263"}'
    assert stable_hash({"b": b"abc"}) == stable_hash({"b": bytearray(b"abc")})


def test_object_without_dict_raises() -> None:
    """Регрессия INF-5: раньше уходило в str(obj) с адресом в памяти."""

    class Slotted:
        __slots__ = ("a",)

        def __init__(self) -> None:
            self.a = 1

    with pytest.raises(TypeError, match="cannot canonicalize 'Slotted'"):
        stable_hash({"o": Slotted()})


# --- INF-3: git commit hash ---


def test_git_commit_hash_in_repo() -> None:
    full = git_commit_hash()
    assert full is not None
    assert len(full) == 40
    assert all(c in "0123456789abcdef" for c in full)


def test_git_commit_hash_short() -> None:
    short = git_commit_hash(short=True)
    full = git_commit_hash()
    assert short is not None and full is not None
    assert len(short) < len(full)
    assert full.startswith(short)


def test_git_commit_hash_outside_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Вне git-репозитория — явный None, а не исключение."""
    monkeypatch.chdir(tmp_path)
    assert git_commit_hash() is None
