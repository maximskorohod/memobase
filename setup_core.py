"""UI-agnostic onboarding core shared by wizard.py (Telegram ``/memobase setup``)
and cli.py (terminal ``hermes memobase setup``).

Both entry points call into the SAME functions here for question text, key
format validation + wrong-key-type detection, the ``.env`` upsert+mask, the
live provider probe, RAM detection, Obsidian auto-detect, and the
ffmpeg/pip dependency check — so the two UIs cannot drift out of sync on
wording, validation rules, or masking behavior. Neither UI's own concern
(chat-state persistence, async replies, ``input()``/``print()``) lives here
— see wizard.py / cli.py for those.

Every public function here is meant to be called from a "must never crash
the onboarding flow" context (a hook callback, or a terminal a real person
is staring at) — none of them raise for their own sake; failures degrade to
an honest message instead.

Secrets discipline (project-wide hard rule): a full API key/token is NEVER
printed, logged, or embedded anywhere by this module — only
:func:`mask_secret`'s first-4-chars-plus-ellipsis form is ever surfaced back
to a human, and the raw value is written ONLY to ``HERMES_HOME/.env`` via
:func:`write_env_secret`'s upsert.
"""

from __future__ import annotations

import importlib
import logging
import os
import platform
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("memobase.setup_core")

# ---------------------------------------------------------------------------
# Step order (embedder -> [cloud_provider -> cloud_key] -> first_ingest ->
# control_question -> done). Obsidian auto-detect has no question of its own
# — its one-line result is prepended onto whichever reply transitions the
# flow into "first_ingest" (see wizard.py). Both UIs walk this same order;
# the Telegram wizard as a persisted per-chat state machine (one message at
# a time, must survive a restart), the terminal as a plain linear script
# (no persistence needed — the whole flow runs in one process, uninterrupted).
# ---------------------------------------------------------------------------

STEPS = ("embedder", "embedder_confirm", "cloud_provider", "cloud_key", "first_ingest", "control_question", "done")

CLOUD_PROVIDERS = {"1": "cloudflare", "2": "cohere", "3": "openai"}

# The single supported local embedder: fastembed (ONNX Runtime, no PyTorch),
# 1024-dim (drop-in for the Cloudflare bge-m3 schema), multilingual, MIT.
LOCAL_EMBED_MODEL = "intfloat/multilingual-e5-large"


# ---------------------------------------------------------------------------
# Key catalog — one field per cloud embedder provider (matches
# CLOUD_PROVIDERS above). Each entry carries everything needed to explain,
# validate, and mask that ONE key: the "ordered keys with metaphors/
# format-hints/validators" the onboarding round asked for.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyField:
    env_var: str
    label: str        # short RU name shown in questions/messages
    metaphor: str      # RU one-liner explaining what the key is FOR, via a metaphor
    format_hint: str   # RU description of the expected shape
    format_re: Any     # compiled re.Pattern[str]


CLOUD_KEY_FIELDS: Dict[str, KeyField] = {
    "cloudflare": KeyField(
        env_var="CLOUDFLARE_API_TOKEN",
        label="ключ Cloudflare (CLOUDFLARE_API_TOKEN)",
        metaphor=(
            "Это как читательский билет в одну конкретную библиотеку — Cloudflare "
            "Workers AI: билет открывает дверь только туда и ни на что больше."
        ),
        format_hint="непустая строка без пробелов (cfat_… или cfut_… — оба ок)",
        format_re=re.compile(r"^\S{8,200}$"),
    ),
    "cohere": KeyField(
        env_var="COHERE_API_KEY",
        label="ключ Cohere (COHERE_API_KEY)",
        metaphor=(
            "Это как абонемент в сервис, который переводит текст на язык чисел "
            "(эмбеддинги) — без абонемента Cohere на порог не пустит."
        ),
        format_hint="непустая строка без пробелов",
        format_re=re.compile(r"^\S{8,200}$"),
    ),
    "openai": KeyField(
        env_var="OPENAI_API_KEY",
        label="ключ OpenAI-совместимого сервиса (OPENAI_API_KEY)",
        metaphor="Это как персональный пропуск на проходной: пускает только вас и никого больше.",
        format_hint="непустая строка без пробелов (обычно sk-…, но зависит от сервера)",
        format_re=re.compile(r"^\S{8,200}$"),
    ),
}

# Backward-compatible single env-var-per-provider mapping (the shape
# wizard.py/cli.py used before this module existed) — derived from the
# catalog above so it is never hand-maintained twice.
CLOUD_KEY_ENV: Dict[str, str] = {p: f.env_var for p, f in CLOUD_KEY_FIELDS.items()}

# Cross-detection by provider prefix is intentionally disabled: providers change
# their prefixes (e.g. Gemini now issues AQ.… keys, not just AIzaSy…), so keying
# "you pasted the wrong key type" off a prefix produced false rejections of
# perfectly valid keys. A genuinely wrong key is still caught downstream by the
# live check. Kept as an empty list so classify_key_mismatch() and its callers
# keep working unchanged (it now simply never reports a mismatch).
_OTHER_KEY_SHAPES: List[Tuple[str, Any, str]] = []


def classify_key_mismatch(target_env_var: str, value: str) -> Optional[str]:
    """Return a human label (e.g. ``"OpenAI"``) if *value* looks like a KNOWN
    OTHER service's key/token instead of *target_env_var*'s own shape;
    ``None`` if no known mismatch is detected (does NOT mean the format is
    otherwise valid — see :func:`validate_key_format`)."""
    value = (value or "").strip()
    if not value:
        return None
    for env_var, pattern, label in _OTHER_KEY_SHAPES:
        if env_var == target_env_var:
            continue  # this IS the field's own expected shape, not a mismatch
        if pattern.match(value):
            return label
    return None


def validate_key_format(provider: str, value: str) -> Tuple[bool, str]:
    """Validate *value* as the key for *provider*'s :data:`CLOUD_KEY_FIELDS`
    entry.

    Returns ``(True, "")`` when acceptable, or ``(False, <RU re-ask
    message>)`` on either a cross-provider mismatch ("looks like an OpenAI
    key, not a Cloudflare token") or a plain format mismatch. Unknown
    providers are accepted as-is (no validator declared for them — never
    blocks on something this module doesn't know about).
    """
    field = CLOUD_KEY_FIELDS.get(provider)
    value = (value or "").strip()
    if field is None:
        return True, ""
    if not value:
        return False, "Пустое сообщение — пришлите, пожалуйста, сам ключ отдельным сообщением."
    mismatch = classify_key_mismatch(field.env_var, value)
    if mismatch:
        return False, (
            f"Это похоже на ключ от {mismatch}, а не {field.label}. "
            f"Пришлите, пожалуйста, именно {field.label} — {field.format_hint}."
        )
    if not field.format_re.match(value):
        return False, (
            f"Формат не похож на {field.label}: {field.format_hint}. "
            "Пришлите, пожалуйста, ещё раз, отдельным сообщением."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Masking + .env upsert — never print/log/embed a full key anywhere.
# ---------------------------------------------------------------------------


def mask_secret(value: str) -> str:
    """First 4 characters + the ellipsis character — the ONLY form a secret
    is ever allowed to appear in a log line, console message, or chat reply
    anywhere in this plugin."""
    value = value or ""
    return value[:4] + "…"


def _env_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / ".env"


def write_env_secret(env_var: str, value: str) -> None:
    """Upsert *env_var* in ``<HERMES_HOME>/.env``, creating the file with
    0600 permissions if it doesn't exist yet (best-effort on Windows, where
    POSIX mode bits are largely advisory — ``os.chmod`` is still called so
    this is correct on POSIX deployments, which is where this bot is
    expected to actually run per this project's VPS-profile framing)."""
    path = _env_path()
    lines = []
    if path.exists():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
    key_re = re.compile(rf"^{re.escape(env_var)}\s*=")
    replaced = False
    for i, line in enumerate(lines):
        if key_re.match(line):
            lines[i] = f"{env_var}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{env_var}={value}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0600 — best-effort on Windows
        except OSError:
            pass
    except OSError:
        logger.error("setup_core: failed to write secret to %s", path, exc_info=True)
        raise
    os.environ[env_var] = value  # make it visible to THIS process immediately, for the live-validate step


# ---------------------------------------------------------------------------
# Key visibility (read side) — a safe, network-free counterpart to
# write_env_secret. Lets `/memobase status` and the onboarding UIs SHOW
# "what is configured, where the .env lives, and is the running process
# actually seeing it" so a human never has to hand-open the file. A full
# secret is NEVER returned — only mask_secret()'s first-4-chars form.
# ---------------------------------------------------------------------------

# Same line grammar the wizard's upsert respects: optional indent, optional
# ``#``/``# `` comment prefix, KEY, ``=``, value. A commented-out or
# empty-valued line does NOT count as "configured".
_ENV_LINE_RE = re.compile(
    r"^(?P<indent>\s*)(?P<hash>#+\s*)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=(?P<rest>.*)$"
)


def env_file_path() -> Path:
    """Public accessor for ``<HERMES_HOME>/.env`` — the single file where this
    plugin's keys are written and read. Surfaced so status/onboarding can SHOW
    the path instead of making a human hunt for it."""
    return _env_path()


def _read_env_file_values() -> Dict[str, str]:
    """``{KEY: value}`` for every ACTIVE (uncommented, non-empty) line in
    ``<HERMES_HOME>/.env``. Reads the file once; never raises (a missing or
    unreadable file yields ``{}``). In-memory only — the read counterpart to
    :func:`write_env_secret`."""
    path = _env_path()
    result: Dict[str, str] = {}
    try:
        if not path.exists():
            return result
        text = path.read_text(encoding="utf-8")
    except OSError:
        return result
    for line in text.splitlines():
        m = _ENV_LINE_RE.match(line)
        if not m or m.group("hash"):
            continue
        val = m.group("rest").strip()
        if val:
            result[m.group("key")] = val
    return result


def key_status(env_vars: List[str]) -> Dict[str, Dict[str, Any]]:
    """Read-only status of each var in *env_vars*, for status/onboarding UIs.

    Returns ``{VAR: {"in_file": bool, "in_process": bool, "mask": str}}``:

    * ``in_file``    — an active, non-empty ``VAR=...`` exists in ``.env``.
    * ``in_process`` — ``os.environ[VAR]`` is set & non-empty — what the RUNNING
      hermes actually sees. Present in the file but NOT here means the gateway
      started before the key was added and needs a restart (bug B12).
    * ``mask``       — :func:`mask_secret` of the value (process value preferred,
      else file value), or ``""`` when set nowhere. NEVER the full value.

    Never raises; never surfaces a full secret.
    """
    file_vals = _read_env_file_values()
    out: Dict[str, Dict[str, Any]] = {}
    for var in env_vars:
        file_val = file_vals.get(var)
        proc_raw = os.environ.get(var)
        proc_val = proc_raw if (proc_raw and proc_raw.strip()) else None
        shown = proc_val or file_val
        out[var] = {
            "in_file": bool(file_val),
            "in_process": bool(proc_val),
            "mask": mask_secret(shown) if shown else "",
        }
    return out


def relevant_keys(memobase_cfg: Dict[str, Any], *, include_youtube: bool = True) -> List[Dict[str, Any]]:
    """Ordered catalog of the ``.env`` keys THIS install actually uses, derived
    from effective config, for the status "Ключи" block. Each item is
    ``{"env_var", "role", "required"}`` — ``role`` a short RU label, ``required``
    True when search/memory is broken without it (vs an optional extra).

    Embedder keys follow ``embedder.provider`` (local needs none); rerank adds
    COHERE when enabled; YouTube/STT keys are optional operator extras."""
    embedder = (memobase_cfg.get("embedder") or {}).get("provider") or "cloudflare"
    rerank = memobase_cfg.get("rerank") or {}
    items: List[Dict[str, Any]] = []

    if embedder == "cloudflare":
        items.append({"env_var": "CLOUDFLARE_ACCOUNT_ID", "role": "эмбеддер (Cloudflare)", "required": True})
        items.append({"env_var": "CLOUDFLARE_API_TOKEN", "role": "эмбеддер (Cloudflare)", "required": True})
    elif embedder == "cohere":
        items.append({"env_var": "COHERE_API_KEY", "role": "эмбеддер (Cohere)", "required": True})
    elif embedder in ("openai", "openai-compat"):
        items.append({"env_var": "OPENAI_API_KEY", "role": "эмбеддер (OpenAI-совм.)", "required": True})
    # local embedder: no keys needed.

    if rerank.get("enabled", True) and rerank.get("provider", "cohere") == "cohere":
        if not any(i["env_var"] == "COHERE_API_KEY" for i in items):
            items.append({"env_var": "COHERE_API_KEY", "role": "реранк (Cohere)", "required": False})

    if include_youtube:
        items.append({"env_var": "SCRAPECREATORS_API_KEY", "role": "YouTube-субтитры", "required": False})
        items.append({"env_var": "APIFY_TOKEN", "role": "YouTube-аудио (запас)", "required": False})
        items.append({"env_var": "GROQ_API_KEY", "role": "распознавание речи", "required": False})
        items.append({"env_var": "GEMINI_API_KEY", "role": "распознавание речи (запас)", "required": False})

    return items


def format_keys_block(memobase_cfg: Dict[str, Any], *, include_youtube: bool = True) -> str:
    """Render the read-only "Ключи" section for `/memobase status`: the ``.env``
    path plus one line per relevant key — ``✓ настроен`` / ``✗ нет`` / ``⚠`` when
    it is in the file but the running process has not picked it up yet. Never
    prints a full secret; never makes a network call."""
    catalog = relevant_keys(memobase_cfg, include_youtube=include_youtube)
    statuses = key_status([c["env_var"] for c in catalog])
    lines = [f"Ключи (.env: {env_file_path()}):"]
    stale = False
    for c in catalog:
        var = c["env_var"]
        st = statuses.get(var, {})
        role = f"[{c['role']}{'' if c['required'] else ', опц.'}]"
        if st.get("in_process"):
            lines.append(f"  {var} — ✓ настроен ({st['mask']}) {role}")
        elif st.get("in_file"):
            stale = True
            lines.append(f"  {var} — ⚠ есть в .env ({st['mask']}), но процесс не видит {role}")
        else:
            lines.append(f"  {var} — ✗ нет {role}")
    if stale:
        lines.append(f"  ⚠ Часть ключей записана, но не подхвачена процессом: {restart_hint()}")
    return "\n".join(lines)


def validate_provider_key(provider: str) -> Tuple[bool, str]:
    """Live-validate a just-entered provider key with the smallest possible
    real call. Never raises — returns ``(False, <message>)`` on any
    failure, exactly like the spec's "ключ принят, эмбеддинг-провайдер
    отвечает ✓" flow.

    NOTE (audit finding, kept as-is — see the onboarding round's notes):
    ``embed.py`` only implements the ``cloudflare`` and ``openai``/
    ``openai-compat`` embedder providers; a ``"cohere"`` probe here will
    always report failure (``unknown embedder provider``) even with a
    perfectly valid Cohere key, because Cohere is currently only wired up
    for RERANKING, not embedding. Fixing that is an ``embed.py`` change,
    out of scope for the onboarding flow itself — this function
    deliberately keeps its pre-existing behavior unchanged.
    """
    try:
        from . import embed as embed_mod

        cfg = {
            "embedder": {
                "provider": provider,
                "model": "@cf/baai/bge-m3" if provider == "cloudflare" else "text-embedding-3-small",
                "dims": 1024,
            }
        }
        embed_mod.embed_texts(["ping"], cfg)
        return True, "ключ принят, эмбеддинг-провайдер отвечает ✓"
    except Exception as exc:  # noqa: BLE001 - best-effort validation, never fatal to the onboarding flow
        return False, f"не удалось проверить ключ живым вызовом ({exc}) — сохранён, проверьте вручную"


# ---------------------------------------------------------------------------
# RAM detection (best-effort, never raises)
# ---------------------------------------------------------------------------


def detect_ram_gb() -> Optional[float]:
    try:
        import psutil  # optional; confirmed present in this project's venv

        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        pass
    try:
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        stat_ = _MEMORYSTATUSEX()
        stat_.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat_)):  # type: ignore[attr-defined]
            return round(stat_.ullTotalPhys / (1024 ** 3), 1)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Obsidian auto-detect notice (shared — the terminal setup wants the exact
# same "found vault(s)" wording the Telegram wizard uses)
# ---------------------------------------------------------------------------


def detect_obsidian_message() -> str:
    try:
        from . import obsidian as obsidian_mod

        vaults = obsidian_mod.detect_vaults()
    except Exception:
        vaults = []
    if not vaults:
        return "Obsidian не найден на этой машине — пропускаю этот шаг."
    names = ", ".join(v.get("name", "?") for v in vaults[:5])
    return (
        f"Нашёл vault(ы): {names}. Чтобы подключить, используйте позже: "
        "«/memobase ingest <путь к vault> obsidian»."
    )


# ---------------------------------------------------------------------------
# Dependency check — ffmpeg (STT/audio chunking) + optional pip packages
# (extract.py's document readers). Never raises; a missing dependency just
# degrades that one ingestion source gracefully (per extract.py/stt.py's own
# contracts) — this only makes that visible to a human BEFORE they hit it
# mid-ingest.
# ---------------------------------------------------------------------------


def detect_ffmpeg() -> Tuple[bool, Optional[str]]:
    """Return ``(found, path_or_None)``. Prefers ``stt.py``'s own discovery
    (single source of truth for its WinGet-glob fallback on Windows dev
    boxes); falls back to a bare ``shutil.which`` if importing ``stt.py``
    fails for any reason."""
    try:
        from . import stt as stt_mod

        path = stt_mod.find_ffmpeg()
        return bool(path), path
    except Exception:
        path = shutil.which("ffmpeg")
        return bool(path), path


# Optional pip packages this plugin degrades gracefully without (extract.py's
# "neither pdfplumber nor pypdf is installed" / "mammoth is not installed" /
# "trafilatura is not installed" skip reasons) — surfaced here so a human sees
# the gap BEFORE an ingest silently skips a file, not after. psutil is listed
# too even though it only affects the RAM hint's precision (never blocks
# anything) — cheap to mention, easy to fix.
PIP_DEPENDENCIES: List[Dict[str, str]] = [
    {
        "import_name": "pdfplumber",
        "pip_name": "pdfplumber",
        "why": "чтение PDF-файлов — как достать текст из отсканированной книги",
    },
    {
        "import_name": "mammoth",
        "pip_name": "mammoth",
        "why": "чтение Word-документов (.docx)",
    },
    {
        "import_name": "trafilatura",
        "pip_name": "trafilatura",
        "why": "чтение статей по ссылке (URL) — как выжимка сути веб-страницы",
    },
    {
        "import_name": "psutil",
        "pip_name": "psutil",
        "why": "точное определение объёма RAM (необязательно — без него просто не покажем ГБ)",
    },
]


def _pip_package_present(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        return True
    except Exception:
        return False


def check_dependencies() -> Dict[str, Any]:
    """Return a plain report dict — never raises. Shape:
    ``{"ffmpeg": {"ok": bool, "path": str|None},
       "pip": [{"import_name", "pip_name", "why", "ok"}, ...]}``."""
    ffmpeg_ok, ffmpeg_path = detect_ffmpeg()
    pip_report = [dict(dep, ok=_pip_package_present(dep["import_name"])) for dep in PIP_DEPENDENCIES]
    return {"ffmpeg": {"ok": ffmpeg_ok, "path": ffmpeg_path}, "pip": pip_report}


def _ffmpeg_install_hint() -> str:
    system = platform.system()
    if system == "Windows":
        return "поставьте так: winget install Gyan.FFmpeg (или choco install ffmpeg)"
    if system == "Darwin":
        return "поставьте так: brew install ffmpeg"
    return "поставьте так: sudo apt install ffmpeg (Debian/Ubuntu) или через пакетный менеджер вашей системы"


def format_dependency_report(report: Dict[str, Any]) -> str:
    """Render *report* (from :func:`check_dependencies`) as a short RU
    summary, simple language, no jargon pile-ups — used verbatim by BOTH the
    Telegram wizard (prepended to its first reply) and the terminal `hermes
    kb setup` command (printed before the embedder question)."""
    lines: List[str] = []
    ffmpeg = report.get("ffmpeg", {})
    missing_pip = [d for d in report.get("pip", []) if not d.get("ok")]

    if ffmpeg.get("ok"):
        lines.append("- ffmpeg: найден — сможет нарезать длинные аудио/видео для распознавания речи.")
    else:
        lines.append(
            "- ffmpeg: НЕ найден — без него не получится распознавать длинные аудио- и "
            f"видеозаписи. {_ffmpeg_install_hint()}."
        )

    if not missing_pip:
        lines.append("- Все нужные Python-библиотеки (pdfplumber, mammoth, trafilatura, psutil) на месте.")
    else:
        for dep in missing_pip:
            lines.append(
                f"- {dep['pip_name']}: не установлен — нужен для: {dep['why']}. "
                f"Поставьте: pip install {dep['pip_name']}"
            )

    return "Проверка окружения:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Question / content text (shared wording — same RU explanations in both UIs)
# ---------------------------------------------------------------------------


def embedder_question(ram_gb: Optional[float]) -> str:
    hint = f" (у вас примерно {ram_gb} ГБ RAM)" if ram_gb else ""
    # Local e5-large needs ~2 GB RAM to run; on a low-RAM box recommend cloud.
    rec = " — рекомендуется вариант 1" if ram_gb and ram_gb >= 8 else (" — здесь мало RAM, рекомендуется вариант 2" if ram_gb else "")
    return (
        "Настройка базы знаний MemoBase.\n"
        "Где считать эмбеддинги? Эмбеддинг — это перевод текста на язык чисел, "
        "чтобы компьютер сравнивал смысл фраз, а не просто буквы."
        f"{hint}{rec}\n"
        "1 — на этой машине, локально: модель multilingual-e5-large "
        "(~2.2 ГБ, скачается один раз; дальше без облака и без ключей для эмбеддингов)\n"
        "2 — через облако: Cloudflare Workers AI и др. (нужен бесплатный API-ключ; ничего не скачивать)"
    )


def local_confirm_question() -> str:
    """Warning + confirm shown after the user picks the local embedder."""
    return (
        "Локальный режим. Сейчас будет:\n"
        "• установлен fastembed (движок эмбеддингов на ONNX, без PyTorch) — небольшой пакет;\n"
        f"• скачана модель {LOCAL_EMBED_MODEL} — ~2.2 ГБ (один раз).\n"
        "Нужно ~2 ГБ свободной RAM. После этого эмбеддинги считаются на твоей машине, "
        "облачные ключи для них не нужны.\n\n"
        "Продолжить? «да» — начну ставить и качать; «назад» — вернуться к выбору."
    )


def is_affirmative(text: str) -> bool:
    return (text or "").strip().lower() in {"да", "д", "y", "yes", "ok", "ок", "продолжить", "go"}


def is_back(text: str) -> bool:
    return (text or "").strip().lower() in {"назад", "back", "нет", "n", "no", "отмена", "cancel"}


def cloud_provider_question() -> str:
    return "Какой облачный провайдер эмбеддингов?\n1 — Cloudflare Workers AI\n2 — Cohere\n3 — OpenAI-совместимый"


def cloud_key_question(provider: str) -> str:
    field = CLOUD_KEY_FIELDS.get(provider)
    if field is None:
        return f"Пришлите API-ключ для {provider} отдельным сообщением."
    return (
        f"{field.metaphor}\n"
        f"Пришлите {field.label} отдельным сообщением ({field.format_hint}). "
        "Он не попадёт в контекст модели и не будет сохранён в истории переписки — только в файл .env. "
        "После сохранения удалите это сообщение из чата вручную, чтобы ключ не остался в переписке."
    )


def first_ingest_question() -> str:
    return "Пришлите файл (документом) или путь к папке для первой загрузки в базу знаний."


def control_question_question() -> str:
    return "Загрузка принята. Задайте любой вопрос по загруженному — я отвечу с цитатой из базы."


def restart_hint() -> str:
    """RU one-liner explaining how to make a freshly-saved key actually
    reach a running hermes -- appended to both onboarding UIs' "done" text
    (:func:`done_message`, used by the Telegram wizard, and directly by
    cli.py's terminal ``hermes memobase setup``).

    The correct instruction depends on how the gateway process is
    supervised (bug B12, "systemd .env gap"): a systemd --user
    ``hermes-gateway.service`` builds its environment once at unit start
    and never re-reads ``.env`` on its own, so a plain ``hermes gateway
    restart`` only picks up a newly-saved key if install.sh's systemd
    drop-in (:func:`systemd_env.wire_gateway_env`) is in place -- detected
    live here, never assumed. Off systemd (plain CLI/foreground session),
    ``.env`` is simply inherited on the next process start, no
    systemd-specific caveat applies.

    Never raises: :func:`systemd_env.detect_systemd_gateway` already
    degrades to ``False`` on its own, but this is wrapped defensively too
    since it runs inside "must never crash the onboarding flow" call sites.
    """
    try:
        from . import systemd_env

        under_systemd = systemd_env.detect_systemd_gateway()
    except Exception:
        under_systemd = False

    if under_systemd:
        return (
            "Перезапустите gateway: `hermes gateway restart` — ключи из .env подхватятся "
            "(systemd EnvironmentFile настроен)."
        )
    return "Перезапустите hermes, чтобы подхватить ключи из .env."


def done_message() -> str:
    base = "Настройка завершена. Шпаргалка: спросить — просто вопросом; добавить — «добавь в базу ...»; статус — /memobase status."
    where = f"Ключи записаны в {env_file_path()}."
    return f"{base} {where} {restart_hint()}"


def local_embedder_model() -> str:
    """The local-embedder model id — a fastembed (ONNX, no PyTorch) multilingual
    model at 1024 dims, drop-in for the Cloudflare bge-m3 schema."""
    return LOCAL_EMBED_MODEL


def install_local_embedder(model: str = LOCAL_EMBED_MODEL) -> Tuple[bool, str]:
    """Install ``fastembed`` into the CURRENT interpreter's venv (the hermes
    venv, since the plugin runs inside it) and pre-download *model*, so the
    first real query is instant. Blocking — pip output streams to the terminal.
    Returns ``(ok, message)`` and never raises: on failure it points at the
    manual ``install.sh --local`` path instead."""
    import subprocess
    import sys

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "fastembed>=0.6"],
            check=True,
        )
    except Exception as exc:  # noqa: BLE001 - onboarding must never crash
        return False, (
            f"не удалось установить fastembed ({exc}). Поставьте вручную: "
            "plugins/memobase/install.sh --local (или install.ps1 -Local)."
        )
    try:
        from fastembed import TextEmbedding

        TextEmbedding(model_name=model)
    except Exception as exc:  # noqa: BLE001
        return False, (
            f"fastembed установлен, но модель не скачалась ({exc}). "
            "Повторите позже или запустите install.sh --local."
        )
    return True, "локальный движок готов: модель скачана, эмбеддинги считаются на этой машине."
