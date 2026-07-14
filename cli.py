"""``hermes memobase ingest|list|reindex|status`` — CLI entry point for
MemoBase (``register_cli_command``).

Thin wrapper: every subcommand calls the same ``tools.py`` handler
functions the tool-call and slash-command paths use, then prints the RU
result to stdout. ``reindex`` is the one CLI-only operation (no tool/slash
equivalent — shadow-table migration is a deliberate, potentially-costly
operation an operator should trigger
explicitly from a terminal, not something a model can call as a tool).
"""

from __future__ import annotations

from . import config as kb_config
from . import db
from . import embed as embed_mod
from . import tools

_SOURCE_TYPES = ("pdf", "docx", "html", "url", "md", "txt", "csv")


def _setup(subparser) -> None:
    sub = subparser.add_subparsers(dest="memobase_subcommand", required=True)

    p_ingest = sub.add_parser("ingest", help="Загрузить файл/URL в коллекцию")
    p_ingest.add_argument("source", help="Путь к файлу или URL")
    p_ingest.add_argument("source_type", choices=_SOURCE_TYPES, help="Тип источника")
    p_ingest.add_argument("--collection", default=None, help="Имя коллекции (по умолчанию — default)")
    p_ingest.add_argument("--confirm", action="store_true", help="Подтвердить загрузку сверх порога")

    sub.add_parser("list", help="Список коллекций")

    p_status = sub.add_parser("status", help="Статус базы знаний")
    p_status.add_argument("--collection", default=None, help="Ограничить одной коллекцией")

    p_reindex = sub.add_parser(
        "reindex", help="Пере-эмбеддинг коллекции в теневую таблицу (после смены embedder в конфиге)"
    )
    p_reindex.add_argument("collection", help="Имя коллекции для переиндексации")

    p_map = sub.add_parser("map", help="Построить mermaid-карту коллекции")
    p_map.add_argument("collection", nargs="?", default=None, help="Имя коллекции (по умолчанию — default)")

    # --- MULTIUSER (owner-only; CLI is always the privileged operator) ----
    p_create_for = sub.add_parser("create-for", help="Создать личную коллекцию для гостя")
    p_create_for.add_argument("collection", help="Имя новой коллекции")
    p_create_for.add_argument("user_id", help="Идентификатор гостя (из шлюза)")

    p_share = sub.add_parser("share", help="Выдать пользователю доступ к коллекции")
    p_share.add_argument("collection")
    p_share.add_argument("user_id")
    p_share.add_argument("permission", nargs="?", default="read", choices=("read", "write"))

    p_revoke = sub.add_parser("share-revoke", help="Отозвать доступ пользователя к коллекции")
    p_revoke.add_argument("collection")
    p_revoke.add_argument("user_id")

    p_quota = sub.add_parser("set-guest-quota", help="Настроить квоту гостя")
    p_quota.add_argument("user_id")
    p_quota.add_argument("--max-mb", type=float, default=None)
    p_quota.add_argument("--max-chunks", type=int, default=None)
    p_quota.add_argument("--daily-upload-mb", type=float, default=None)
    p_quota.add_argument("--daily-budget-usd", type=float, default=None)
    p_quota.add_argument("--daily-calls", type=int, default=None)

    p_qlist = sub.add_parser("quarantine-list", help="Показать очередь гостевых загрузок на проверку")
    p_qlist.add_argument("--collection", default=None)

    p_qreview = sub.add_parser("quarantine-review", help="Одобрить/отклонить запись из очереди проверки")
    p_qreview.add_argument("quarantine_id", type=int)
    p_qreview.add_argument("action", choices=("approve", "reject"))

    # --- OPS -----------------------------------------------------------
    p_backup = sub.add_parser("backup-run", help="Запустить ночной доктор вручную (снимок БД + ротация + диск)")
    p_backup.add_argument("--dir", default=None, help="Каталог для снимков (по умолчанию memobase/backups)")

    sub.add_parser("setup", help="Интерактивный мастер настройки (эмбеддинги/ключи/Obsidian/первая загрузка)")


def _handle(args) -> None:
    sub = getattr(args, "memobase_subcommand", None)

    if sub == "ingest":
        result = tools.memobase_ingest(
            {
                "source": args.source,
                "source_type": args.source_type,
                "collection": args.collection,
                "confirm": args.confirm,
            },
            session_id=None,
        )
        print(result)
        return

    if sub == "list":
        print(tools.memobase_list({}, session_id=None))
        return

    if sub == "status":
        status_args = {"collection": args.collection} if args.collection else {}
        print(tools.memobase_status(status_args, session_id=None))
        return

    if sub == "reindex":
        conn = db.get_connection()
        try:
            row = db.get_collection_by_name(conn, args.collection)
            if row is None:
                print(f"Коллекция «{args.collection}» не найдена.")
                return
            memobase_cfg = kb_config.get_memobase_config_readonly()
            new_cfg = kb_config.get_collection_cfg(row, memobase_cfg=memobase_cfg)
            print(f"Переиндексация коллекции «{args.collection}» начата (embed_signature: {embed_mod.embedding_signature(new_cfg)})...")
            result = embed_mod.reembed_collection_shadow(conn, row, new_cfg)
            print(
                f"Переиндексация завершена: {result.get('chunks_embedded', 0)} фрагмент(ов) переэмбеддено, "
                f"векторный индекс готов: {result.get('vector_index_ready')}."
            )
        except (embed_mod.EmbedError, db.DbError) as exc:
            print(f"Ошибка переиндексации: {exc}")
        finally:
            conn.close()
        return

    if sub == "map":
        print(tools.memobase_map({"collection": args.collection} if args.collection else {}, session_id=None))
        return

    if sub == "create-for":
        print(tools.memobase_create_for({"collection": args.collection, "user_id": args.user_id}, session_id=None))
        return

    if sub == "share":
        print(tools.memobase_share(
            {"collection": args.collection, "user_id": args.user_id, "permission": args.permission},
            session_id=None,
        ))
        return

    if sub == "share-revoke":
        print(tools.memobase_share_revoke({"collection": args.collection, "user_id": args.user_id}, session_id=None))
        return

    if sub == "set-guest-quota":
        fields = {
            "max_mb": args.max_mb, "max_chunks": args.max_chunks, "daily_upload_mb": args.daily_upload_mb,
            "daily_budget_usd": args.daily_budget_usd, "daily_calls": args.daily_calls,
        }
        fields = {k: v for k, v in fields.items() if v is not None}
        print(tools.memobase_set_guest_quota({"user_id": args.user_id, **fields}, session_id=None))
        return

    if sub == "quarantine-list":
        q_args = {"collection": args.collection} if args.collection else {}
        print(tools.memobase_quarantine_list(q_args, session_id=None))
        return

    if sub == "quarantine-review":
        print(tools.memobase_quarantine_review(
            {"quarantine_id": args.quarantine_id, "action": args.action}, session_id=None,
        ))
        return

    if sub == "backup-run":
        from . import backup as backup_mod

        memobase_cfg = kb_config.get_memobase_config_readonly()
        backup_dir = None
        if args.dir:
            from pathlib import Path as _Path

            backup_dir = _Path(args.dir)
        result = backup_mod.run_doctor(memobase_cfg, backup_dir=backup_dir)
        print(backup_mod.format_report(result))
        return

    if sub == "setup":
        _run_cli_setup_wizard()
        return

    print(f"Неизвестная подкоманда: {sub}")


def _run_cli_setup_wizard() -> None:
    """Terminal ``hermes memobase setup`` — drives ``setup_core.py``'s shared
    onboarding logic via plain ``input()``/``print()``.

    Same RU explanations (with metaphors), same key-format validation +
    wrong-key-type detection, same masking, and the same live provider probe
    as the Telegram ``/memobase setup`` wizard (wizard.py) — both call into
    ``setup_core`` so the two entry points cannot drift out of sync. A
    terminal has neither wizard.py's "one message at a time, must survive a
    restart" constraint nor its chat identity, so this is a plain linear
    prompt sequence instead of a persisted per-chat state machine.
    """
    from . import setup_core

    print(setup_core.format_dependency_report(setup_core.check_dependencies()))
    print()

    ram = setup_core.detect_ram_gb()
    while True:
        print(setup_core.embedder_question(ram))
        choice = input("> ").strip()[:1]

        if choice == "1":  # local embedder
            print()
            print(setup_core.local_confirm_question())
            ans = input("> ").strip()
            if setup_core.is_back(ans):
                print()
                continue  # back to the start of the embedder choice
            if not setup_core.is_affirmative(ans):
                print("Не понял ответ — возвращаюсь к выбору.\n")
                continue
            print("\nСтавлю локальный движок и качаю модель (~2.2 ГБ, один раз)…")
            ok, msg = setup_core.install_local_embedder()
            print(msg)
            if ok:
                kb_config.set_memobase_value("embedder.provider", "local")
                kb_config.set_memobase_value("embedder.model", setup_core.local_embedder_model())
                kb_config.set_memobase_value("embedder.dims", 1024)
            else:
                print("Локальный режим не настроен — можно повторить позже или выбрать облако.")
            break

        if choice == "2":  # cloud embedder
            print()
            print(setup_core.cloud_provider_question())
            provider_choice = input("> ").strip()
            provider = setup_core.CLOUD_PROVIDERS.get(provider_choice)
            if provider is None:
                print("Не понял выбор провайдера — оставляю текущие настройки эмбеддера без изменений.")
                break
            kb_config.set_memobase_value("embedder.provider", provider)
            env_var = setup_core.CLOUD_KEY_ENV.get(provider, "API_KEY")
            print()
            print(setup_core.cloud_key_question(provider))
            _MAX_ATTEMPTS = 3
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                key = input("> ").strip()
                if not key:
                    print("Пустой ввод — пропускаю сохранение ключа. Настроить его можно позже вручную в .env.")
                    break
                ok, hint = setup_core.validate_key_format(provider, key)
                if ok:
                    setup_core.write_env_secret(env_var, key)
                    _ok_live, msg = setup_core.validate_provider_key(provider)
                    print(f"Ключ ({setup_core.mask_secret(key)}) сохранён — {msg}")
                    break
                print(hint)
                if attempt == _MAX_ATTEMPTS:
                    print("Не получилось получить ключ подходящего формата — настройте его позже вручную в .env.")
            break

        print("Не понял выбор — напишите 1 (локально) или 2 (облако).\n")

    print()
    print(setup_core.detect_obsidian_message())
    print("Готово. Дальше: `hermes memobase ingest <файл> <тип>`, `hermes memobase list`, `hermes memobase status`.")
    print(setup_core.restart_hint())


def register(ctx) -> None:
    ctx.register_cli_command(
        "memobase",
        help="Управление базой знаний MemoBase (ingest/list/status/reindex)",
        setup_fn=_setup,
        handler_fn=_handle,
        description="MemoBase: локальная база знаний с гибридным поиском и цитируемыми ответами.",
    )
