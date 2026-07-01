# INF-4 — CI-пайплайн для код-ревью merge requests

- **Эпик:** E0 — Scaffolding & инфраструктура
- **Статус:** done
- **Зависимости:** INF-1

## Описание

CI-пайплайн, который на каждый pull/merge request в `main` проверяет
**качество кода** и **корректность реализации задачи** (соответствие тикету).

### 1. Quality gate (обязательные проверки)

- GitHub Actions workflow на `pull_request` (target: `main`).
- Шаги: `ruff check` + `ruff format --check`, `mypy`, `pytest` (с coverage).
- Кэш зависимостей, матрица по Python (минимум 3.13).
- Падение любого шага блокирует merge.

### 2. Автоматическое код-ревью

- Автоматический ревьюер (AI review, напр. Cursor Bugbot или review-агент),
  который комментирует PR по двум осям:
  - **качество кода:** читаемость, дублирование, типобезопасность, тесты,
    нарушения архитектурных принципов из `README`/`tickets`;
  - **корректность задачи:** реализованы ли критерии приёмки (DoD) связанного
    тикета; нет ли расхождений с описанием.
- В PR указывается ID тикета (например `BT-3`); ревьюер сверяет изменения с DoD
  этого тикета.

### 3. Процессная обвязка

- PR template: ссылка на тикет, чек-лист DoD, краткое описание изменений.
- Branch protection для `main`: требуются зелёный quality gate и пройденное ревью
  до merge; запрет прямого push в `main`.

## Критерии приёмки (DoD)

- [x] Workflow запускается на каждый PR в `main` и гоняет ruff/mypy/pytest
      (`.github/workflows/quality.yml`, матрица Python 3.11–3.13, pytest+coverage).
- [x] Красный quality gate блокирует merge — механизм готов; ruleset `main`
      включается админом репозитория (инструкция в `README` → «CI и код-ревью»).
- [x] На PR появляется автоматический ревью-комментарий с оценкой качества кода
      и соответствия DoD тикета (`.github/workflows/claude-review.yml`;
      требует секрет `ANTHROPIC_API_KEY`).
- [x] Есть PR template с полем «связанный тикет» и чек-листом DoD
      (`.github/pull_request_template.md`).
- [x] Документация в `README`: как проходит ревью и что требуется для merge.

## Примечание

- CI использует `pip install -e ".[dev]"` (без lock-файла) с кэшем pip; версия
  `ruff` зафиксирована (`==0.15.20`) в dev-deps и pre-commit, чтобы `ruff format
  --check` был детерминирован между локалью и CI. В рамках задачи репозиторий
  единожды приведён к формату этой версии.
- Ручные шаги админа репозитория (вне git-диффа): добавить секрет
  `ANTHROPIC_API_KEY` и включить branch ruleset для `main` (required status
  checks = джоб `checks`, require PR before merge, block direct pushes).
