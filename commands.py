"""``/memobase`` slash command for MemoBase — works identically in CLI and
gateway sessions (``register_command`` handler signature
``fn(raw_args: str) -> str | None``, no chat/user identity
available).

    /memobase <question>                          -> memobase_ask against the default
                                                (or session-bound) collection
    /memobase status [collection]                 -> memobase_status
    /memobase list                                -> memobase_list
    /memobase ingest <source> <type> [collection] -> memobase_ingest
    /memobase selfcheck <collection>              -> memobase_selfcheck
    /memobase delete <collection>                 -> memobase_delete
    /memobase help                                -> this text

Deliberately thin: every subcommand calls straight into ``tools.py``'s
plain module-level handler functions (NOT ``tools.registry.dispatch`` — no
tool-call machinery needed for a slash command) with ``session_id=None``,
since a slash command has no session-binding concept of its own (binding is
a delegated-subagent isolation mechanism, see ``tools.py``'s module
docstring — a top-level chat session invoking ``/memobase`` is always
"unbound"/privileged).

ВАЖНО: новые slash-подкоманды подхватываются gateway только после «hermes
gateway restart» — до рестарта работает CLI-путь (hermes memobase ...).
"""

from __future__ import annotations

from typing import Optional

from . import tools

HELP_TEXT = (
    "/memobase <вопрос> — спросить базу знаний (коллекция по умолчанию)\n"
    "/memobase status [коллекция] — статус базы знаний\n"
    "/memobase list — список коллекций\n"
    "/memobase list-documents [коллекция] — документы внутри коллекции (заголовок, источник, число фрагментов)\n"
    "/memobase ingest <источник> <тип> [коллекция] — загрузить файл/URL\n"
    "/memobase selfcheck <коллекция> — проверить качество индексации\n"
    "/memobase delete <коллекция> — удалить коллекцию целиком\n"
    "/memobase map [коллекция] — карта коллекции (mermaid)\n"
    "/memobase create-for <user_id> <коллекция> — создать личную коллекцию гостю (владелец)\n"
    "/memobase share <коллекция> <user_id> [read|write] — выдать доступ (владелец)\n"
    "/memobase share-revoke <коллекция> <user_id> — отозвать доступ (владелец)\n"
    "/memobase quarantine [коллекция] — очередь гостевых загрузок на проверку (владелец)\n"
    "/memobase youtube-list <канал> [сколько] — свежие видео канала (владелец)\n"
    "/memobase youtube-transcript <видео> — таймкод-транскрипт по субтитрам (владелец)\n"
    "/memobase youtube-audio <видео> — аудио+STT для видео без субтитров, платно (владелец)\n"
    "/memobase setup — мастер настройки (в Telegram — по одному сообщению за раз; в CLI см. `hermes memobase setup`)\n"
    "/memobase help — эта справка"
)

_SOURCE_TYPES = {"pdf", "docx", "html", "url", "md", "txt", "csv"}


def handle_kb_command(raw_args: str) -> Optional[str]:
    text = (raw_args or "").strip()
    if not text:
        return HELP_TEXT

    first_word, _, rest = text.partition(" ")
    sub = first_word.lower()
    rest = rest.strip()

    if sub in ("help", "?"):
        return HELP_TEXT

    if sub == "status":
        args = {"collection": rest} if rest else {}
        return tools.memobase_status(args, session_id=None)

    if sub == "list":
        return tools.memobase_list({}, session_id=None)

    if sub == "list-documents":
        args = {"collection": rest} if rest else {}
        return tools.memobase_list_documents(args, session_id=None)

    if sub == "ingest":
        parts = rest.split()
        if len(parts) < 2:
            return "Использование: /memobase ingest <источник> <тип> [коллекция]"
        source, source_type = parts[0], parts[1]
        if source_type not in _SOURCE_TYPES:
            return f"Неизвестный тип источника «{source_type}». Допустимые: {', '.join(sorted(_SOURCE_TYPES))}."
        args = {"source": source, "source_type": source_type}
        if len(parts) > 2:
            args["collection"] = parts[2]
        return tools.memobase_ingest(args, session_id=None)

    if sub == "selfcheck":
        if not rest:
            return "Использование: /memobase selfcheck <коллекция>"
        return tools.memobase_selfcheck({"collection": rest}, session_id=None)

    if sub == "delete":
        if not rest:
            return "Использование: /memobase delete <коллекция>"
        return tools.memobase_delete({"collection": rest}, session_id=None)

    if sub == "map":
        args = {"collection": rest} if rest else {}
        return tools.memobase_map(args, session_id=None)

    if sub == "create-for":
        parts = rest.split(maxsplit=1)
        if len(parts) < 2:
            return "Использование: /memobase create-for <user_id> <коллекция>"
        return tools.memobase_create_for({"user_id": parts[0], "collection": parts[1]}, session_id=None)

    if sub == "share":
        parts = rest.split()
        if len(parts) < 2:
            return "Использование: /memobase share <коллекция> <user_id> [read|write]"
        args = {"collection": parts[0], "user_id": parts[1]}
        if len(parts) > 2:
            args["permission"] = parts[2]
        return tools.memobase_share(args, session_id=None)

    if sub in ("share-revoke", "revoke"):
        parts = rest.split()
        if len(parts) < 2:
            return "Использование: /memobase share-revoke <коллекция> <user_id>"
        return tools.memobase_share_revoke({"collection": parts[0], "user_id": parts[1]}, session_id=None)

    if sub == "quarantine":
        args = {"collection": rest} if rest else {}
        return tools.memobase_quarantine_list(args, session_id=None)

    if sub == "youtube-list":
        parts = rest.split()
        if not parts:
            return "Использование: /memobase youtube-list <канал> [сколько]"
        args = {"channel": parts[0]}
        if len(parts) > 1:
            try:
                args["limit"] = int(parts[1])
            except ValueError:
                return f"«{parts[1]}» — не число (второй параметр — сколько видео вернуть)."
        return tools.memobase_youtube_list_channel(args, session_id=None)

    if sub == "youtube-transcript":
        if not rest:
            return "Использование: /memobase youtube-transcript <видео>"
        return tools.memobase_youtube_get_transcript({"video": rest}, session_id=None)

    if sub == "youtube-audio":
        if not rest:
            return "Использование: /memobase youtube-audio <видео>"
        return tools.memobase_youtube_dl_audio({"video": rest}, session_id=None)

    if sub == "setup":
        return (
            "Мастер настройки в CLI: `hermes memobase setup`. В Telegram отправьте боту «/memobase setup» напрямую — "
            "там мастер ведёт диалог по одному сообщению (эта команда, вызванная через /memobase, не имеет доступа "
            "к вашей переписке в чате)."
        )

    # Default: the whole text is a question against the default (or
    # session-bound) collection — matches DESIGN_v1.md's "/memobase <вопрос>".
    return tools.memobase_ask({"query": text}, session_id=None)


def register(ctx) -> None:
    ctx.register_command(
        "memobase",
        handler=handle_kb_command,
        description="Спросить базу знаний MemoBase или управлять коллекциями.",
        args_hint="<вопрос> | status|list|ingest|selfcheck|delete ...",
    )
