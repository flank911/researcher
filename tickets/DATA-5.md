# DATA-5 — DuckDB/Polars слой доступа (+warmup)

- **Эпик:** E1 — Data Core
- **Статус:** done
- **Зависимости:** DATA-2
- **Реализует улучшение:** №1 (warmup-буфер на границах слайса)

## Описание

Слой доступа к данным поверх Parquet.

- `data/storage/duckdb_store.py`: обёртки над `read_parquet`/`scan_parquet`.
- Polars lazy `scan_parquet` для ленивого чтения без загрузки всего файла.
- Выдача непрерывного ряда с **warmup-буфером**: запрос окна `[start, end]`
  фактически читает `[start - warmup_bars, end]`, помечая warmup-часть.

## Критерии приёмки (DoD)

- [x] Запрос окна возвращает прогретый ряд (warmup-бары присутствуют и помечены).
- [x] Ленивое сканирование не грузит весь файл в память (проверка на большом фикстуре).
- [x] SQL-запрос через DuckDB поверх Parquet возвращает корректные данные.

## Примечание

- `data/storage/duckdb_store.py` (`DuckDBStore`): `scan_candles` → `pl.LazyFrame`
  (`scan_parquet` по партициям, pushdown фильтров/проекций); `read_window(start,
  end, warmup_bars)` → `pl.DataFrame` с колонкой `is_warmup`; `query(sql)` —
  DuckDB SQL поверх тех же файлов (view `candles`).
- Warmup: нижняя граница скана = `start − warmup_bars` шагов timeframe → читается
  только окно+прогрев. При пропусках баров прогрев best-effort (сколько есть).
- Layout путей переиспользует схему `ParquetStore` (DATA-2).
