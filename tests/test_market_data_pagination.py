"""Tests for REG-5 — пагинация `/futures/data/*` (OI/LS) против хвостового ответа.

Ключевая фикстура здесь — `_TailTransport`: он воспроизводит **фактическое**
поведение Binance для `/futures/data/*`, а не удобное для кода. Эндпоинт отдаёт
*хвост* запрошенного окна (последние `limit` точек перед `endTime`), поэтому
курсорная пагинация вперёд молча теряет начало диапазона.

`/fapi/v1/fundingRate` ведёт себя иначе — отдаёт от `startTime` вперёд, — и для
него проверяется, что прежняя логика не сломана (`_ForwardTransport`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from trading_research.data.ingestion.binance_client import HttpResponse
from trading_research.data.ingestion.market_data_client import (
    MAX_LIMIT_DATA,
    PERIOD_MS,
    BinanceMarketDataClient,
)

T0 = datetime(2026, 10, 2, tzinfo=UTC)
H = timedelta(hours=1)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


class _TailTransport:
    """Binance `/futures/data/*`: точки окна по `period`, но **только хвост**.

    Отдаёт последние `limit` точек из `[startTime, endTime]` — именно так, как
    отвечает биржа. Если код предполагает ответ «от `startTime` вперёд», начало
    запрошенного диапазона до него не доедет.
    """

    def __init__(self, period_ms: int = PERIOD_MS["1h"]) -> None:
        self.period_ms = period_ms
        self.calls: list[dict[str, str]] = []

    def get(self, url: str, params: Mapping[str, str]) -> HttpResponse:
        self.calls.append(dict(params))
        start, end = int(params["startTime"]), int(params["endTime"])
        limit = int(params["limit"])

        first = -(-start // self.period_ms) * self.period_ms  # выравнивание вверх
        points = list(range(first, end + 1, self.period_ms))
        tail = points[-limit:] if limit else []

        rows: list[dict[str, Any]] = [
            {
                "symbol": params["symbol"],
                "sumOpenInterest": "1000.0",
                "sumOpenInterestValue": "5000000.0",
                "longShortRatio": "1.5",
                "longAccount": "0.6",
                "shortAccount": "0.4",
                "timestamp": ts,
            }
            for ts in tail
        ]
        return HttpResponse(200, {}, json.dumps(rows).encode())


class _ForwardTransport:
    """Binance `/fapi/v1/fundingRate`: точки **от** `startTime` вперёд, не более `limit`."""

    def __init__(self, step_ms: int = 8 * 3_600_000) -> None:
        self.step_ms = step_ms
        self.calls: list[dict[str, str]] = []

    def get(self, url: str, params: Mapping[str, str]) -> HttpResponse:
        self.calls.append(dict(params))
        start, end = int(params["startTime"]), int(params["endTime"])
        limit = int(params["limit"])

        first = -(-start // self.step_ms) * self.step_ms
        points = list(range(first, end + 1, self.step_ms))[:limit]
        rows = [
            {"symbol": params["symbol"], "fundingTime": ts, "fundingRate": "0.0001"}
            for ts in points
        ]
        return HttpResponse(200, {}, json.dumps(rows).encode())


def _client(transport: object) -> BinanceMarketDataClient:
    return BinanceMarketDataClient(transport=transport, sleep_fn=lambda _s: None)  # type: ignore[arg-type]


# --- окно шире limit: начало диапазона не теряется ---------------------------


@pytest.mark.parametrize("fetch", ["fetch_open_interest", "fetch_long_short_ratio"])
def test_window_wider_than_limit_is_fully_covered(fetch: str) -> None:
    """Профиль RUN-7: 28 дней часовых точек (672) при `limit=500`.

    До исправления записывалось ровно 500 строк — хвост окна, а первые семь дней
    отсутствовали. Проверяем края, а не только число строк: именно край и терялся.
    """
    transport = _TailTransport()
    end = T0 + 28 * 24 * H
    df = getattr(_client(transport), fetch)("BTCUSDT", T0, end, period="1h", limit=500)

    assert df.height == 672, "окно 28 дней при периоде 1h — это 672 точки"
    assert df["open_time"][0] == T0, "потеряно начало окна"
    assert df["open_time"][-1] == end - H, "правая граница эксклюзивна"
    assert df["open_time"].is_sorted()
    assert df["open_time"].n_unique() == df.height, "дубли на стыке окон"
    assert len(transport.calls) == 2, "672 точки при limit=500 — это два окна"


def test_window_exactly_limit_is_single_request() -> None:
    """Ровно `limit` точек — одно окно, без лишнего запроса за пустотой."""
    transport = _TailTransport()
    df = _client(transport).fetch_open_interest("BTCUSDT", T0, T0 + 500 * H, period="1h", limit=500)
    assert df.height == 500
    assert df["open_time"][0] == T0
    assert len(transport.calls) == 1


def test_many_windows_have_no_overlap_or_hole() -> None:
    """Пять окон подряд: стык окон не теряет и не дублирует точки."""
    df = _client(_TailTransport()).fetch_open_interest(
        "BTCUSDT", T0, T0 + 50 * H, period="1h", limit=10
    )
    assert df.height == 50
    deltas = df["open_time"].diff().drop_nulls().unique().to_list()
    assert deltas == [H], f"шаг должен быть ровно 1h, получено {deltas}"


def test_window_boundaries_are_half_open() -> None:
    """Запрошенный `endTime` эксклюзивен: точка на правой границе не берётся дважды."""
    transport = _TailTransport()
    _client(transport).fetch_open_interest("BTCUSDT", T0, T0 + 20 * H, period="1h", limit=10)

    windows = [(int(c["startTime"]), int(c["endTime"])) for c in transport.calls]
    assert windows == [
        (_ms(T0), _ms(T0 + 10 * H) - 1),
        (_ms(T0 + 10 * H), _ms(T0 + 20 * H) - 1),
    ]


@pytest.mark.parametrize("period", list(PERIOD_MS))
def test_every_supported_period_covers_its_window(period: str) -> None:
    """Нарезка считает ширину окна по периоду, поэтому проверяем все периоды."""
    period_ms = PERIOD_MS[period]
    transport = _TailTransport(period_ms=period_ms)
    end = T0 + timedelta(milliseconds=period_ms * 25)
    df = _client(transport).fetch_open_interest("BTCUSDT", T0, end, period=period, limit=10)

    assert df.height == 25
    assert df["open_time"][0] == T0


def test_unknown_period_raises_before_any_request() -> None:
    """Неизвестный период — ошибка до сети: ширину окна по нему не посчитать."""
    transport = _TailTransport()
    with pytest.raises(ValueError, match="unsupported period"):
        _client(transport).fetch_open_interest("BTCUSDT", T0, T0 + 10 * H, period="7h")
    assert transport.calls == []


def test_empty_window_does_not_stop_the_rest() -> None:
    """Дыра в середине диапазона не обрывает загрузку последующих окон."""

    class _HoleTransport(_TailTransport):
        def get(self, url: str, params: Mapping[str, str]) -> HttpResponse:
            if int(params["startTime"]) == _ms(T0 + 10 * H):
                self.calls.append(dict(params))
                return HttpResponse(200, {}, b"[]")
            return super().get(url, params)

    transport = _HoleTransport()
    df = _client(transport).fetch_open_interest("BTCUSDT", T0, T0 + 30 * H, period="1h", limit=10)

    assert len(transport.calls) == 3, "пустое окно не должно прерывать цикл"
    assert df.height == 20, "загружены первое и третье окна"
    assert df["open_time"][-1] == T0 + 29 * H


def test_default_limit_matches_binance_cap() -> None:
    """Ширина окна по умолчанию опирается на лимит эндпоинта, а не на магию."""
    transport = _TailTransport()
    _client(transport).fetch_open_interest("BTCUSDT", T0, T0 + MAX_LIMIT_DATA * H, period="1h")
    assert len(transport.calls) == 1
    assert transport.calls[0]["limit"] == str(MAX_LIMIT_DATA)


# --- регрессия: funding пагинируется вперёд ---------------------------------


def test_funding_paginates_forward_across_batches() -> None:
    """У `/fapi/v1/fundingRate` ответ идёт от `startTime` — курсорная логика жива."""
    transport = _ForwardTransport()
    end = T0 + 100 * 8 * H  # 100 выплат по 8 часов
    df = _client(transport).fetch_funding_rate("BTCUSDT", T0, end, limit=30)

    assert df.height == 100
    assert df["open_time"][0] == T0
    assert df["open_time"][-1] == end - 8 * H
    assert df["open_time"].n_unique() == 100
    assert len(transport.calls) == 4, "100 точек по 30 за батч — четыре запроса"


def test_funding_stops_on_short_batch() -> None:
    """Неполный батч означает конец данных — лишнего запроса не делаем."""
    transport = _ForwardTransport()
    df = _client(transport).fetch_funding_rate("BTCUSDT", T0, T0 + 10 * 8 * H, limit=30)
    assert df.height == 10
    assert len(transport.calls) == 1
