#!/usr/bin/env bash
# Installs MemoBase end-to-end: finds the hermes-agent venv, installs the
# Python dependencies into it, copies this plugin folder into
# <HERMES_HOME>/plugins/memobase/ (unless already running from there --
# updates it in place instead), enables it in config.yaml
# (plugins.enabled), and -- if the gateway runs as a systemd --user unit --
# wires a systemd drop-in (EnvironmentFile=<HERMES_HOME>/.env) so
# `hermes gateway restart` actually picks up keys saved to .env (bug B12;
# a no-op off systemd, see systemd_env.py). One command; nothing left to do
# by hand except optional API keys (see the printed summary at the end).
#
# Package list (DESIGN_v1.md "install script" section) -- all MIT/BSD/
# Apache, no torch, no pymupdf (AGPL), no docling:
#   sqlite-vec pdfplumber pypdf mammoth "trafilatura>=1.8" ftfy py3langid
#   PyStemmer requests
#
# Usage:
#   ./install.sh                       # auto-detect everything; install + copy + enable
#   ./install.sh /path/to/venv/bin/python
#   ./install.sh --local               # ALSO install the local embedder
#                                      #   (fastembed, ONNX, no PyTorch) and
#                                      #   pre-download multilingual-e5-large
#                                      #   (~2.2 GB) so no CLOUDFLARE_* keys
#                                      #   are needed for embeddings.
#   ./install.sh --no-config           # skip the automatic config.yaml patch
#                                      #   (prints the lines to add by hand)
#   HERMES_VENV_PYTHON=/path/to/python ./install.sh
#
# Safe to re-run: the copy and the config.yaml patch are both idempotent.

set -euo pipefail
export PYTHONIOENCODING=utf-8

PYTHON_OVERRIDE=""
INSTALL_LOCAL=0
NO_CONFIG=0
for arg in "$@"; do
    case "$arg" in
        --local) INSTALL_LOCAL=1 ;;
        --no-config) NO_CONFIG=1 ;;
        *) PYTHON_OVERRIDE="$arg" ;;
    esac
done

resolve_hermes_venv_python() {
    if [ -n "$PYTHON_OVERRIDE" ]; then
        if [ -x "$PYTHON_OVERRIDE" ]; then
            echo "$PYTHON_OVERRIDE"
            return 0
        fi
        echo "Указанный путь к python не существует или не исполняемый: $PYTHON_OVERRIDE" >&2
        return 1
    fi

    if [ -n "${HERMES_VENV_PYTHON:-}" ] && [ -x "${HERMES_VENV_PYTHON}" ]; then
        echo "$HERMES_VENV_PYTHON"
        return 0
    fi

    # `hermes` on PATH is normally a shim inside the venv's bin/ dir -- its
    # sibling python is exactly the interpreter every plugin runs under.
    if command -v hermes >/dev/null 2>&1; then
        local hermes_bin
        hermes_bin="$(command -v hermes)"
        local scripts_dir
        scripts_dir="$(dirname "$hermes_bin")"
        if [ -x "$scripts_dir/python" ]; then
            echo "$scripts_dir/python"
            return 0
        fi
        if [ -x "$scripts_dir/python3" ]; then
            echo "$scripts_dir/python3"
            return 0
        fi
    fi

    # Fall back to conventional layouts. Try both default HERMES_HOME locations
    # and both venv layouts -- POSIX (bin/python) and Windows (Scripts/python.exe) --
    # so running install.sh from Git Bash on Windows also resolves the venv.
    local h sub cand
    for h in "${HERMES_HOME:-}" "$HOME/.hermes" "$HOME/AppData/Local/hermes"; do
        [ -n "$h" ] || continue
        for sub in venv/bin/python venv/Scripts/python.exe .venv/bin/python .venv/Scripts/python.exe; do
            cand="$h/hermes-agent/$sub"
            if [ -x "$cand" ] || [ -f "$cand" ]; then
                echo "$cand"
                return 0
            fi
        done
    done

    echo "Не удалось найти python интерпретатор hermes-agent venv автоматически." >&2
    echo "Укажите его явно: ./install.sh /path/to/hermes-agent/venv/bin/python" >&2
    echo "(на Windows/Git Bash — .../hermes-agent/venv/Scripts/python.exe)" >&2
    echo "или задайте переменную окружения HERMES_VENV_PYTHON." >&2
    return 1
}

# Authoritative HERMES_HOME: ask the resolved venv python itself
# (hermes_constants.get_hermes_home()) so
# this never has to duplicate hermes' own HERMES_HOME env-var/default
# resolution. Falls back to the same convention resolve_hermes_venv_python()
# already uses above if the probe fails for any reason (e.g. a non-hermes
# python was passed explicitly).
resolve_hermes_home() {
    local py="$1"
    local home=""
    home="$("$py" -c 'from hermes_constants import get_hermes_home; print(get_hermes_home())' 2>/dev/null)" || true
    if [ -n "$home" ]; then
        printf '%s\n' "$home"
        return 0
    fi
    printf '%s\n' "${HERMES_HOME:-$HOME/.hermes}"
}

PYTHON="$(resolve_hermes_venv_python)"
echo "MemoBase: устанавливаю зависимости в $PYTHON"

"$PYTHON" -m pip install --upgrade \
    sqlite-vec \
    pdfplumber \
    pypdf \
    mammoth \
    "trafilatura>=1.8" \
    ftfy \
    py3langid \
    PyStemmer \
    requests

if [ "$INSTALL_LOCAL" = "1" ]; then
    echo ""
    echo "MemoBase: ставлю локальный эмбеддер (fastembed — ONNX Runtime, без PyTorch)..."
    # fastembed>=0.6: e5-large обучена под mean pooling; fastembed<=0.5.1 применял CLS —
    # другой вектор-спейс. Порог фиксирует протестированное поведение для новых баз.
    "$PYTHON" -m pip install --upgrade "fastembed>=0.6"
    echo "MemoBase: скачиваю модель intfloat/multilingual-e5-large (~2.2 ГБ, один раз)..."
    "$PYTHON" - <<'PYEOF'
from fastembed import TextEmbedding
TextEmbedding(model_name="intfloat/multilingual-e5-large")
print("  локальная модель готова к работе")
PYEOF
    echo "MemoBase: локальный режим установлен. Включите его в config.yaml (memobase.*):"
    echo "    memobase:"
    echo "      embedder: { provider: local, model: intfloat/multilingual-e5-large, dims: 1024 }"
fi

# ---------------------------------------------------------------------------
# Copy this plugin folder into <HERMES_HOME>/plugins/memobase/ -- unless the
# script is already running from there. Never deletes anything at the
# destination first (no `rm -rf`): a plain recursive overwrite-copy is
# idempotent and can't nuke unrelated files that happen to live alongside an
# existing install.
# ---------------------------------------------------------------------------
HERMES_HOME_RESOLVED="$(resolve_hermes_home "$PYTHON")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
DEST_DIR="$HERMES_HOME_RESOLVED/plugins/memobase"

echo ""
if [ -d "$DEST_DIR" ] && [ "$(cd "$DEST_DIR" && pwd -P)" = "$SCRIPT_DIR" ]; then
    echo "MemoBase: уже запущен из $DEST_DIR — копирование пропущено."
else
    echo "MemoBase: копирую плагин из $SCRIPT_DIR в $DEST_DIR ..."
    mkdir -p "$DEST_DIR"
    cp -R "$SCRIPT_DIR"/. "$DEST_DIR"/
    echo "MemoBase: скопировано (повторный запуск просто обновит файлы поверх)."
fi

# ---------------------------------------------------------------------------
# Enable memobase in config.yaml (plugins.enabled). Sanctioned path first
# (hermes_cli.config -- the same API the /setup wizard uses), a backed-up
# hand-edit fallback second, precise manual instructions if both fail.
# Never allowed to abort the install (the `|| true` below is belt-and-braces
# on top of the script's own internal try/except).
# ---------------------------------------------------------------------------
if [ "$NO_CONFIG" = "1" ]; then
    echo ""
    echo "MemoBase: --no-config указан — пропускаю автовключение в config.yaml."
    echo "Добавьте вручную:"
    echo "  plugins:"
    echo "    enabled:"
    echo "      - memobase"
    CONFIG_STATUS="skipped"
else
    echo ""
    echo "MemoBase: включаю плагин в config.yaml (plugins.enabled)..."
    STATUS_FILE="$(mktemp)"
    "$PYTHON" - "$STATUS_FILE" "$HERMES_HOME_RESOLVED" <<'PYEOF' || true
import os
import sys

status_path = sys.argv[1] if len(sys.argv) > 1 else None
hermes_home = sys.argv[2] if len(sys.argv) > 2 else ""

MANUAL_HINT = (
    "MemoBase: добавьте вручную в config.yaml:\n"
    "  plugins:\n"
    "    enabled:\n"
    "      - memobase"
)


def enable_via_hermes_cli():
    # Sanctioned path -- hermes_cli.config is the supported API for enabling
    # a plugin in config.yaml, not a hand-rolled YAML edit.
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    plugins_cfg = cfg.get("plugins")
    if not isinstance(plugins_cfg, dict):
        plugins_cfg = {}
    enabled = plugins_cfg.get("enabled")
    if not isinstance(enabled, list):
        enabled = []
    if "memobase" in enabled:
        return "already"
    plugins_cfg["enabled"] = enabled + ["memobase"]
    cfg["plugins"] = plugins_cfg
    save_config(cfg)
    return "enabled"


def enable_via_fallback():
    # hermes_cli.config itself is unavailable/failed -- hand-edit the YAML
    # directly, with a one-time backup, never clobbering an existing backup.
    import shutil
    import yaml

    config_path = None
    try:
        from hermes_cli.config import get_config_path
        config_path = get_config_path()
    except (Exception, SystemExit):
        config_path = None
    if not config_path:
        if not hermes_home:
            raise RuntimeError("HERMES_HOME неизвестен, путь к config.yaml не найден")
        config_path = os.path.join(hermes_home, "config.yaml")
    if not os.path.isfile(config_path):
        raise RuntimeError("config.yaml не найден: %s" % config_path)

    backup_path = config_path + ".bak"
    if not os.path.isfile(backup_path):
        shutil.copyfile(config_path, backup_path)

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise RuntimeError("config.yaml имеет неожиданный формат (не словарь)")

    plugins_cfg = raw.get("plugins")
    if not isinstance(plugins_cfg, dict):
        plugins_cfg = {}
    enabled = plugins_cfg.get("enabled")
    if not isinstance(enabled, list):
        enabled = []
    if "memobase" in enabled:
        return "already"
    plugins_cfg["enabled"] = enabled + ["memobase"]
    raw["plugins"] = plugins_cfg

    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, allow_unicode=True, sort_keys=False)
    print("MemoBase: config.yaml обновлён резервным способом (бэкап: %s)." % backup_path)
    print("MemoBase: резервный способ не сохраняет комментарии/форматирование — при желании сверьте файл.")
    return "enabled"


def main():
    try:
        status = enable_via_hermes_cli()
        if status == "already":
            print("MemoBase: 'memobase' уже в plugins.enabled — пропускаю.")
        else:
            print("MemoBase: добавлен в plugins.enabled (config.yaml, штатный способ).")
        return status
    except (Exception, SystemExit) as exc:
        print("MemoBase: штатный способ (hermes_cli.config) не сработал: %s" % (exc,))
        print("MemoBase: пробую резервный способ (прямая правка config.yaml)...")

    try:
        status = enable_via_fallback()
        if status == "already":
            print("MemoBase: 'memobase' уже в plugins.enabled (резервный способ) — пропускаю.")
        return status
    except (Exception, SystemExit) as exc:
        print("MemoBase: резервный способ тоже не сработал: %s" % (exc,))
        print(MANUAL_HINT)
        return "failed"


result = main()
if status_path:
    try:
        with open(status_path, "w", encoding="utf-8") as fh:
            fh.write(result)
    except Exception:
        pass
PYEOF
    CONFIG_STATUS="$(cat "$STATUS_FILE" 2>/dev/null || true)"
    rm -f "$STATUS_FILE"
    if [ -z "$CONFIG_STATUS" ]; then
        CONFIG_STATUS="failed"
    fi
fi

# ---------------------------------------------------------------------------
# systemd drop-in (bug B12, "systemd .env gap"): if the gateway runs as a
# systemd --user unit, wire an EnvironmentFile pointing at
# <HERMES_HOME>/.env so `hermes gateway restart` actually picks up keys
# saved by the setup wizard/CLI -- the shipped unit never reads .env on its
# own otherwise. Runs from the COPIED destination (not $SCRIPT_DIR) so the
# invoked systemd_env.py is always the version that just landed at
# $DEST_DIR. Safely a no-op off systemd (systemd_env.py's own guard); the
# `|| true` is belt-and-braces on top of that so this step can never abort
# the install.
# ---------------------------------------------------------------------------
"$PYTHON" "$DEST_DIR/systemd_env.py" "$HERMES_HOME_RESOLVED" || true

echo ""
echo "============================================================"
if [ "$CONFIG_STATUS" = "enabled" ] || [ "$CONFIG_STATUS" = "already" ]; then
    echo "MemoBase установлен и включён."
    echo "Работает сразу в keyword-режиме (без ключей)."
    echo "Семантика/ключи: hermes memobase setup"
    echo "Офлайн-вектор: перезапустите ./install.sh --local"
else
    echo "MemoBase установлен, но НЕ включён в config.yaml (см. сообщения выше)."
    echo "Включите вручную, затем: hermes memobase setup"
fi
echo "Перезапустите hermes и проверьте: /memobase status"
echo "============================================================"
