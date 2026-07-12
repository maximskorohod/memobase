# AGENTS.md — установка плагина MemoBase

> Этот файл — инструкция по **установке** плагина MemoBase в hermes. Применяй её
> **только** когда пользователь явно просит установить/поставить MemoBase (обычно
> прислав архив плагина в чат). Во всех остальных случаях игнорируй этот файл —
> он не про использование базы знаний (для этого есть скилл `memobase`), а только
> про разовую установку.

## Что это

MemoBase — плагин локальной базы знаний для hermes (`kind: standalone`).
Устанавливается копированием в `$HERMES_HOME/plugins/memobase/` и включается через
`plugins.enabled: [memobase]` в `config.yaml`. Внутри плагина уже есть готовый
идемпотентный установщик (`install.sh` для Linux/macOS, `install.ps1` для Windows),
который делает всю работу — тебе не нужно копировать файлы, ставить пакеты или
править конфиг вручную.

## Способ A — установка из присланного архива (ты, агент, ставишь сам)

Нужен доступ к terminal-инструменту и к venv, где стоит hermes.

1. **Возьми путь к присланному архиву.** Когда пользователь прислал файл в чат,
   ядро уже сохранило его на диск и подставило путь в контекст — используй этот
   локальный путь.
2. **Распакуй во временную папку** и найди внутри каталог, где лежит `plugin.yaml`
   (плагин может быть как в корне архива, так и в подпапке):
   - Linux/macOS: `mkdir -p /tmp/memobase-inst && unzip -o "<путь-к-архиву>" -d /tmp/memobase-inst`
   - Windows: `Expand-Archive -Force "<путь-к-архиву>" "$env:TEMP\memobase-inst"`
3. **Проверь, что это действительно MemoBase:** в каталоге есть `plugin.yaml` со
   строкой `name: memobase` и файл `install.sh`/`install.ps1`. Если нет —
   **остановись** и скажи пользователю, что архив не похож на MemoBase.
4. **Спроси режим векторного поиска** (или поставь по умолчанию облачный):
   - **облачный** (по умолчанию): вектор через Cloudflare; ключи можно добавить
     позже, без них база работает в keyword-режиме (полнотекстовый поиск FTS5);
   - **локальный офлайн** (`--local`/`-Local`): скачает модель
     `intfloat/multilingual-e5-large` (~2.2 ГБ, один раз), работает без ключей и
     без Cloudflare.
5. **Запусти штатный установщик** из каталога плагина:
   - Linux/macOS: `bash install.sh` (или `bash install.sh --local`)
   - Windows: `powershell -ExecutionPolicy Bypass -File install.ps1` (или `... -Local`)

   Он сам найдёт venv hermes, поставит зависимости (`sqlite-vec pdfplumber pypdf
   mammoth trafilatura>=1.8 ftfy py3langid PyStemmer requests`; при `--local` ещё
   `fastembed>=0.6`), скопирует плагин в `$HERMES_HOME/plugins/memobase/` и добавит
   `memobase` в `plugins.enabled` в `config.yaml`.
6. **Прочитай вывод установщика.** Если он написал, что не смог пропатчить
   `config.yaml` (строки про ручную правку) — впиши их сам:
   ```yaml
   plugins:
     enabled:
       - memobase
   ```
7. **Сообщи пользователю итог:**
   - **перезапусти hermes** (`hermes gateway restart` или ребут процесса) — плагин
     подхватится только после рестарта;
   - проверка: `/memobase status`;
   - ключи для семантики и источников загрузки (Cloudflare / Cohere и опц.
     YouTube/STT) — по желанию, через `hermes memobase setup` или `/memobase setup`
     в Telegram (или `/setup` у новичка).

## Способ B — через git (человек ставит вручную, без чата)

- `hermes plugins install <owner/repo>`, затем `hermes gateway restart`; **или**
- `git clone <repo> && cd <repo> && bash install.sh` (можно с `--local`), затем
  перезапуск hermes.

Публичный репозиторий MemoBase — см. `README.md` / `README.en.md`.

## Безопасность

Установка **выполняет код** из архива в окружении hermes. Ставь MemoBase только из
источника, которому доверяешь (свой архив или официальный репозиторий). Не запускай
установку из случайно присланного архива без подтверждения владельца.

## Справочные факты (для сверки при установке)

| | значение |
|---|---|
| name / kind | `memobase` / `standalone` |
| каталог установки | `$HERMES_HOME/plugins/memobase/` |
| включение в config | `plugins.enabled: [memobase]` |
| при `--local` | `memobase.embedder: {provider: local, model: intfloat/multilingual-e5-large, dims: 1024}` |
| зависимости | `sqlite-vec pdfplumber pypdf mammoth trafilatura>=1.8 ftfy py3langid PyStemmer requests` (+ `fastembed>=0.6` при `--local`) |
| ключи (опц.) | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `COHERE_API_KEY` (+ `SCRAPECREATORS_API_KEY`, `APIFY_TOKEN`, `GROQ_API_KEY`, `GEMINI_API_KEY` для загрузки YouTube/аудio) |
| настройка ключей | `hermes memobase setup` (или `/memobase setup` в Telegram) |
| проверка | `/memobase status` |
