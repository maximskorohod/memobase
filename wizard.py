"""``/memobase setup`` onboarding wizard for MemoBase.

Deterministic per-chat state machine driven entirely through the
``pre_gateway_dispatch`` hook — wizard steps
are answered with plain text/numbered choices that must NEVER reach the LLM
(fast, free, zero "interpretation" risk per the spec), and a chat's identity
(``chat_id``/``user_id``/``platform``) is only ever available at that hook,
never inside a ``register_command`` handler (no chat identity there at all)
nor inside a tool handler (only ``session_id``/``task_id``).

State machine (steps, in order — see ``_STEPS``):

    embedder -> [cloud_provider -> cloud_key] -> obsidian -> first_ingest
    -> control_question -> done

Resumable: state is persisted to ``<HERMES_HOME>/memobase/wizard_state.json``
after every mutation (small, human-readable) so a hermes restart mid-wizard
picks up exactly where the chat left off. Owner-only: the very first call
(from ANYONE) claims ``memobase.owner_user_id`` if it is not set yet — matching
the project's "zero-config, first real use wins" posture elsewhere in this
plugin (tools.py's session/collection binding) — any SUBSEQUENT caller who
is not that claimed owner is refused.

CAVEATS (best-effort, flagged for the integrator):
  * The cloud_key step's question ADVISES the user to delete the message
    containing their key themselves (checklist item — this module does NOT
    call any TG-side ``adapter.delete_message`` itself; an earlier draft's
    docstring promised best-effort bot-side deletion here, but no code path
    ever actually called it, so the promise was corrected to what is really
    true today: an honest, explicit ask to the human). Wiring real bot-side
    deletion would need ``message_id`` plumbed through
    ``pre_gateway_dispatch``'s event and is left as a follow-up.
  * RAM detection prefers ``psutil`` (present in this venv, confirmed) and
    falls back to a Windows ``ctypes`` call, then to "unknown" — never
    raises either way.
  * All onboarding CONTENT (question wording, key-format validation +
    wrong-key-type detection, .env upsert+masking, the live provider probe,
    RAM/Obsidian detection, the ffmpeg/pip dependency check) now lives in
    ``setup_core.py``, shared verbatim with the terminal ``hermes memobase setup``
    command (cli.py) — this module only owns the TG-specific bits: chat-
    scoped persisted state, the owner gate, and hook wiring/async replies.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from . import setup_core

logger = logging.getLogger("memobase.wizard")

_STATE_LOCK = threading.Lock()
_STATE_FILENAME = "wizard_state.json"

# Step order. `cloud_provider`/`cloud_key` are skipped entirely for a local
# embedder choice. There is no dedicated "obsidian" step — auto-detection
# has no question to ask, so its result is prepended as a one-line notice
# onto whichever reply transitions the wizard into "first_ingest" (from
# either the local-embedder branch or the end of the cloud-key branch).
# Owned by setup_core.py so the terminal `hermes memobase setup` command walks the
# exact same order.
_STEPS = setup_core.STEPS

CLOUD_PROVIDERS = setup_core.CLOUD_PROVIDERS
CLOUD_KEY_ENV = setup_core.CLOUD_KEY_ENV

# Thin re-exports of setup_core's shared onboarding primitives, kept as
# plain module-level names (not `setup_core.xxx(...)` call sites) so this
# module's own functions look them up via wizard.py's global namespace at
# call time — that is what lets tests `monkeypatch.setattr(wizard,
# "validate_provider_key", ...)` intercept the live probe without reaching
# into setup_core at all.
detect_ram_gb = setup_core.detect_ram_gb
detect_obsidian_message = setup_core.detect_obsidian_message
write_env_secret = setup_core.write_env_secret
validate_provider_key = setup_core.validate_provider_key
mask_secret = setup_core.mask_secret


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def _state_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "memobase" / _STATE_FILENAME


def _load_state() -> Dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        logger.warning("wizard: failed to read %s; starting fresh", path, exc_info=True)
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("wizard: failed to persist state to %s", path, exc_info=True)


def is_active(chat_id: str) -> bool:
    with _STATE_LOCK:
        state = _load_state()
    entry = state.get(str(chat_id))
    return bool(entry and entry.get("step") not in (None, "done"))


def _get_entry(state: Dict[str, Any], chat_id: str) -> Optional[Dict[str, Any]]:
    return state.get(str(chat_id))


def _set_entry(chat_id: str, entry: Dict[str, Any]) -> None:
    with _STATE_LOCK:
        state = _load_state()
        state[str(chat_id)] = entry
        _save_state(state)


def clear_wizard(chat_id: str) -> None:
    with _STATE_LOCK:
        state = _load_state()
        state.pop(str(chat_id), None)
        _save_state(state)


# ---------------------------------------------------------------------------
# Owner gate
# ---------------------------------------------------------------------------


def is_owner_allowed(user_id: Optional[str], memobase_cfg: Dict[str, Any]) -> bool:
    """First caller ever to reach the wizard claims ``memobase.owner_user_id`` if
    unset; everyone else must already match it. Mirrors
    ``security.is_privileged``'s "unset owner = not yet claimed" posture."""
    if not user_id:
        return False
    owner = (memobase_cfg or {}).get("owner_user_id") or ""
    if not owner:
        try:
            from . import config as kb_config

            kb_config.set_memobase_value("owner_user_id", str(user_id))
            logger.info("memobase wizard: claimed owner_user_id=%s", user_id)
        except Exception:
            logger.warning("wizard: failed to persist claimed owner_user_id", exc_info=True)
        return True
    return str(user_id) == str(owner)


# ---------------------------------------------------------------------------
# Step machine — question wording/validation/masking/deps-check all live in
# setup_core.py now; this module only threads them through the persisted
# per-chat step state.
# ---------------------------------------------------------------------------


def _question_for(step: str, entry: Dict[str, Any]) -> str:
    if step == "embedder":
        return setup_core.embedder_question(detect_ram_gb())
    if step == "embedder_confirm":
        return setup_core.local_confirm_question()
    if step == "cloud_provider":
        return setup_core.cloud_provider_question()
    if step == "cloud_key":
        provider = entry.get("data", {}).get("cloud_provider", "cloudflare")
        return setup_core.cloud_key_question(provider)
    if step == "first_ingest":
        return setup_core.first_ingest_question()
    if step == "control_question":
        return setup_core.control_question_question()
    return setup_core.done_message()


def start_wizard(chat_id: str, user_id: str) -> str:
    entry = {"step": "embedder", "data": {}, "user_id": user_id}
    _set_entry(chat_id, entry)
    deps_notice = setup_core.format_dependency_report(setup_core.check_dependencies())
    return f"{deps_notice}\n\n{_question_for('embedder', entry)}"


def handle_message(chat_id: str, user_id: str, text: str, memobase_cfg: Dict[str, Any]) -> Optional[str]:
    """Advance the wizard for *chat_id* by one step given the raw incoming
    *text*. Returns the reply to send (never None while the wizard is
    active — an unparseable answer just re-asks the same question)."""
    with _STATE_LOCK:
        state = _load_state()
        entry = _get_entry(state, chat_id)
    if entry is None:
        return None
    if str(entry.get("user_id")) != str(user_id):
        return None  # a different identity in the same chat_id -- ignore, don't hijack someone else's wizard

    step = entry.get("step")
    text = (text or "").strip()
    data = entry.setdefault("data", {})

    if step == "embedder":
        choice = text[:1]
        if choice == "2":
            entry["step"] = "cloud_provider"
            _set_entry(chat_id, entry)
            return _question_for("cloud_provider", entry)
        if choice == "1":
            entry["step"] = "embedder_confirm"
            _set_entry(chat_id, entry)
            return _question_for("embedder_confirm", entry)
        return "Не понял ответ — пришлите 1 (локально) или 2 (облако).\n\n" + _question_for("embedder", entry)

    if step == "embedder_confirm":
        if setup_core.is_back(text):
            entry["step"] = "embedder"
            _set_entry(chat_id, entry)
            return _question_for("embedder", entry)
        if not setup_core.is_affirmative(text):
            return "Напишите «да» — продолжить, или «назад» — вернуться к выбору.\n\n" + _question_for("embedder_confirm", entry)
        try:
            from . import config as kb_config

            kb_config.set_memobase_value("embedder.provider", "local")
            kb_config.set_memobase_value("embedder.model", setup_core.local_embedder_model())
            kb_config.set_memobase_value("embedder.dims", 1024)
        except Exception:
            logger.warning("wizard: failed to persist local embedder choice", exc_info=True)
        # A ~2.2 GB download can't run inline inside a synchronous chat handler,
        # so the Telegram path records the choice and points at the server-side
        # install step (the terminal `hermes memobase setup` downloads inline).
        local_note = (
            "Локальный режим записан (multilingual-e5-large). Модель ~2.2 ГБ ставится на сервере: "
            "выполните там `install.sh --local` (поставит fastembed и скачает модель). "
            "Без этого первый запрос попросит поставить fastembed.\n\n"
        )
        return _advance_to(chat_id, entry, "first_ingest", prefix=local_note + detect_obsidian_message())

    if step == "cloud_provider":
        provider = CLOUD_PROVIDERS.get(text[:1])
        if not provider:
            return "Не понял ответ — пришлите 1, 2 или 3.\n\n" + _question_for("cloud_provider", entry)
        data["cloud_provider"] = provider
        try:
            from . import config as kb_config

            kb_config.set_memobase_value("embedder.provider", provider)
        except Exception:
            logger.warning("wizard: failed to persist cloud provider choice", exc_info=True)
        entry["step"] = "cloud_key"
        _set_entry(chat_id, entry)
        return _question_for("cloud_key", entry)

    if step == "cloud_key":
        provider = data.get("cloud_provider", "cloudflare")
        env_var = CLOUD_KEY_ENV.get(provider, "API_KEY")
        ok_format, hint = setup_core.validate_key_format(provider, text)
        if not ok_format:
            return f"{hint}\n\n{_question_for('cloud_key', entry)}"
        try:
            write_env_secret(env_var, text)
        except Exception as exc:
            return f"Не удалось сохранить ключ: {exc}. Попробуйте прислать его ещё раз."
        ok, msg = validate_provider_key(provider)
        masked = mask_secret(text)
        prefix = f"Ключ ({masked}) сохранён в {setup_core.env_file_path()} — {msg}.\n\n{detect_obsidian_message()}"
        return _advance_to(chat_id, entry, "first_ingest", prefix=prefix)

    if step == "first_ingest":
        return _advance_to(chat_id, entry, "control_question", prefix="Принято.")

    if step == "control_question":
        return _advance_to(chat_id, entry, "done")

    return None


def _advance_to(chat_id: str, entry: Dict[str, Any], next_step: str, *, prefix: str = "") -> str:
    entry["step"] = next_step
    _set_entry(chat_id, entry)
    question = _question_for(next_step, entry)
    return f"{prefix}\n\n{question}".strip() if prefix else question


# ---------------------------------------------------------------------------
# pre_gateway_dispatch entry point
# ---------------------------------------------------------------------------

_SETUP_TRIGGER_RE = re.compile(r"^\s*/memobase\s+setup\b", re.IGNORECASE)


def _send_async(gateway: Any, platform: Any, chat_id: str, text: str) -> None:
    """Fire-and-forget reply, following the documented pattern for replying
    on a ``skip`` from a PLAIN (non-async) hook callback."""
    try:
        import asyncio

        adapter = gateway.adapters[platform]
        asyncio.create_task(adapter.send(chat_id, text))
    except Exception:
        logger.warning("wizard: failed to send reply via gateway adapter", exc_info=True)


def on_gateway_dispatch(event: Any = None, gateway: Any = None, session_store: Any = None, **_kw: Any) -> Optional[Dict[str, Any]]:
    """``pre_gateway_dispatch`` hook: intercepts ``/memobase setup`` and any
    follow-up message while a wizard is active for this chat, BEFORE the
    LLM ever sees it (Часть 2b's core requirement). Returns
    ``{"action": "skip", ...}`` when handled, ``None`` otherwise (let the
    message flow through normally) — never raises."""
    if event is None:
        return None
    try:
        source = getattr(event, "source", None)
        chat_id = getattr(source, "chat_id", None)
        user_id = getattr(source, "user_id", None)
        platform = getattr(source, "platform", None)
        text = getattr(event, "text", "") or ""
        if not chat_id:
            return None

        from . import config as kb_config

        memobase_cfg = kb_config.get_memobase_config_readonly()

        if _SETUP_TRIGGER_RE.match(text):
            if not is_owner_allowed(user_id, memobase_cfg):
                _send_async(gateway, platform, chat_id, "Мастер настройки доступен только владельцу базы знаний.")
                return {"action": "skip", "reason": "kb_setup_not_owner"}
            reply = start_wizard(chat_id, user_id)
            _send_async(gateway, platform, chat_id, reply)
            return {"action": "skip", "reason": "kb_setup_wizard"}

        if is_active(chat_id):
            reply = handle_message(chat_id, user_id, text, memobase_cfg)
            if reply is not None:
                _send_async(gateway, platform, chat_id, reply)
                return {"action": "skip", "reason": "kb_setup_wizard_step"}
        return None
    except Exception:  # noqa: BLE001 - hook contract: never raise, never block dispatch
        logger.debug("memobase wizard: pre_gateway_dispatch handling failed (non-fatal)", exc_info=True)
        return None


def register(ctx: Any) -> None:
    ctx.register_hook("pre_gateway_dispatch", on_gateway_dispatch)
