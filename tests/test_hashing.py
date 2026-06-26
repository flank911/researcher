"""Tests for INF-3 — stable hashing."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime

from trading_research.hashing import canonical_json, stable_hash


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
    code = (
        "from trading_research.hashing import stable_hash;"
        "print(stable_hash({'strategy': 'ma_cross', 'fast_ma': 20, 'slow_ma': 100}))"
    )
    env_runs = [
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for _ in range(2)
    ]
    assert env_runs[0] == env_runs[1]
    assert env_runs[0] == stable_hash(
        {"strategy": "ma_cross", "fast_ma": 20, "slow_ma": 100}
    )
