#!/usr/bin/env bash
# OPS-1 — одна итерация сбора рыночных рядов, для запуска по расписанию.
#
# Смысл расписания: Binance отдаёт open interest и long/short ratio только за
# последние 30 дней, поэтому их история не загружается, а создаётся этим сбором.
# Пропущенный месяц — дыра, которую нечем закрыть.
#
# Путь к репозиторию определяется от самого скрипта, чтобы launchd не зависел от
# рабочего каталога. Код возврата команды сохраняется: launchd показывает его в
# `launchctl list`, и именно по нему видно, что итерация прошла с пропуском.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${TR_CONFIG:-$REPO/configs/experiments/ma_cross_multi_1h_4h.yaml}"
LOG="${TR_LOG:-$REPO/data/logs/update-market-data.log}"

mkdir -p "$(dirname "$LOG")"
{
  echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC') — $CONFIG"
  "$REPO/.venv/bin/trading-research" update-market-data --config "$CONFIG"
  echo "--- exit=$?"
} >>"$LOG" 2>&1
