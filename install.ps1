#Requires -Version 5.1
<#
.SYNOPSIS
    Installs MemoBase end-to-end into the hermes-agent installation.

.DESCRIPTION
    One command does everything: finds the hermes-agent venv, installs the
    Python dependencies into it, copies this plugin folder into
    <HERMES_HOME>\plugins\memobase\ (unless already running from there --
    updates it in place instead), and enables it in config.yaml
    (plugins.enabled). Safe to re-run: the copy and the config.yaml patch
    are both idempotent.

    Package list (DESIGN_v1.md "install script" section) -- all MIT/BSD/
    Apache, no torch, no pymupdf (AGPL), no docling:
        sqlite-vec pdfplumber pypdf mammoth "trafilatura>=1.8" ftfy
        py3langid PyStemmer requests

    ``requests`` is already a hermes-core dependency in practice, but is
    listed explicitly so this script is self-sufficient even against a bare
    venv.

.NOTES
    Never installs into the REPO's own dev venv used for this project's test
    suite (see tests/README section in the operator checklist) -- always
    targets the resolved HERMES venv python.
#>

[CmdletBinding()]
param(
    # Explicit override -- pass the full path to the hermes venv's python.exe
    # if auto-detection below can't find it (e.g. non-standard install).
    [string]$PythonPath,

    # ALSO install the local embedder (fastembed, ONNX, no PyTorch) and
    # pre-download multilingual-e5-large (~2.2 GB), so no CLOUDFLARE_* keys
    # are needed for embeddings.
    [switch]$Local,

    # Skip the automatic config.yaml patch (plugins.enabled). Prints the
    # lines to add by hand instead.
    [switch]$NoConfig
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

function Resolve-HermesVenvPython {
    param([string]$Override)

    if ($Override) {
        if (Test-Path $Override) { return (Resolve-Path $Override).Path }
        throw "Указанный -PythonPath не существует: $Override"
    }

    if ($env:HERMES_VENV_PYTHON -and (Test-Path $env:HERMES_VENV_PYTHON)) {
        return (Resolve-Path $env:HERMES_VENV_PYTHON).Path
    }

    # `hermes` on PATH is normally a shim/exe inside the venv's Scripts/ dir --
    # its sibling python.exe is exactly the interpreter every plugin runs under.
    $hermesCmd = Get-Command hermes -ErrorAction SilentlyContinue
    if ($hermesCmd) {
        $scriptsDir = Split-Path -Parent $hermesCmd.Source
        $candidate = Join-Path $scriptsDir "python.exe"
        if (Test-Path $candidate) { return $candidate }
    }

    # Fall back to the conventional HERMES_HOME/hermes-agent/venv layout.
    $hermesHome = $env:HERMES_HOME
    if (-not $hermesHome) { $hermesHome = Join-Path $env:LOCALAPPDATA "hermes" }
    $candidate = Join-Path $hermesHome "hermes-agent\venv\Scripts\python.exe"
    if (Test-Path $candidate) { return $candidate }

    throw (
        "Не удалось найти python интерпретатор hermes-agent venv автоматически. " +
        "Укажите его явно: .\install.ps1 -PythonPath 'C:\path\to\hermes-agent\venv\Scripts\python.exe' " +
        "или задайте переменную окружения HERMES_VENV_PYTHON."
    )
}

# Authoritative HERMES_HOME: ask the resolved venv python itself
# (hermes_constants.get_hermes_home()) so
# this never has to duplicate hermes' own HERMES_HOME env-var/default
# resolution. Falls back to the same convention Resolve-HermesVenvPython
# already uses above if the probe fails for any reason (e.g. a non-hermes
# python was passed explicitly via -PythonPath).
function Resolve-HermesHome {
    param([string]$PythonExe)

    try {
        $probed = & $PythonExe -c "from hermes_constants import get_hermes_home; print(get_hermes_home())" 2>$null
        if ($LASTEXITCODE -eq 0 -and $probed) { return $probed.Trim() }
    } catch {}

    $hermesHome = $env:HERMES_HOME
    if (-not $hermesHome) { $hermesHome = Join-Path $env:LOCALAPPDATA "hermes" }
    return $hermesHome
}

$python = Resolve-HermesVenvPython -Override $PythonPath
Write-Host "MemoBase: устанавливаю зависимости в $python" -ForegroundColor Cyan

$packages = @(
    "sqlite-vec",
    "pdfplumber",
    "pypdf",
    "mammoth",
    "trafilatura>=1.8",
    "ftfy",
    "py3langid",
    "PyStemmer",
    "requests"
)

& $python -m pip install --upgrade @packages
if ($LASTEXITCODE -ne 0) {
    Write-Host "MemoBase: установка зависимостей не удалась (см. вывод pip выше)." -ForegroundColor Red
    exit $LASTEXITCODE
}

if ($Local) {
    Write-Host ""
    Write-Host "MemoBase: ставлю локальный эмбеддер (fastembed - ONNX Runtime, без PyTorch)..." -ForegroundColor Cyan
    & $python -m pip install --upgrade fastembed
    if ($LASTEXITCODE -ne 0) {
        Write-Host "MemoBase: установка fastembed не удалась (см. вывод pip выше)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "MemoBase: скачиваю модель intfloat/multilingual-e5-large (~2.2 ГБ, один раз)..." -ForegroundColor Cyan
    & $python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='intfloat/multilingual-e5-large'); print('  локальная модель готова к работе')"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "MemoBase: не удалось скачать локальную модель (см. вывод выше)." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "MemoBase: локальный режим установлен. Включите его в config.yaml (memobase.*):" -ForegroundColor Green
    Write-Host "    memobase:"
    Write-Host "      embedder: { provider: local, model: intfloat/multilingual-e5-large, dims: 1024 }"
}

# ---------------------------------------------------------------------------
# Copy this plugin folder into <HERMES_HOME>\plugins\memobase\ -- unless the
# script is already running from there. Copies file-by-file (never a bulk
# Copy-Item -Recurse over an existing destination folder, which can nest a
# duplicate copy instead of merging) and never deletes anything at the
# destination first, so it can't nuke unrelated files sitting alongside an
# existing install.
# ---------------------------------------------------------------------------
$hermesHome = Resolve-HermesHome -PythonExe $python
$scriptDir = $PSScriptRoot
$destDir = Join-Path (Join-Path $hermesHome "plugins") "memobase"

Write-Host ""
$scriptDirFull = (Resolve-Path $scriptDir).Path.TrimEnd('\')
$destDirFull = $null
if (Test-Path $destDir) {
    $destDirFull = (Resolve-Path $destDir).Path.TrimEnd('\')
}

if ($destDirFull -and ($destDirFull -ieq $scriptDirFull)) {
    Write-Host "MemoBase: уже запущен из $destDir — копирование пропущено." -ForegroundColor Yellow
} else {
    Write-Host "MemoBase: копирую плагин из $scriptDir в $destDir ..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    $sourceFiles = Get-ChildItem -Path $scriptDir -Recurse -File
    foreach ($file in $sourceFiles) {
        $relativePath = $file.FullName.Substring($scriptDir.Length).TrimStart('\')
        $targetPath = Join-Path $destDir $relativePath
        $targetParent = Split-Path -Parent $targetPath
        if (-not (Test-Path $targetParent)) {
            New-Item -ItemType Directory -Force -Path $targetParent | Out-Null
        }
        Copy-Item -Path $file.FullName -Destination $targetPath -Force
    }
    Write-Host "MemoBase: скопировано (повторный запуск просто обновит файлы поверх)." -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# Enable memobase in config.yaml (plugins.enabled). Sanctioned path first
# (hermes_cli.config -- the same API the /setup wizard uses), a backed-up
# hand-edit fallback second, precise manual instructions if both fail. The
# Python side never raises (everything funnels through try/except), and the
# try/catch here is belt-and-braces on top of that.
# ---------------------------------------------------------------------------
$configStatus = "failed"
if ($NoConfig) {
    Write-Host ""
    Write-Host "MemoBase: -NoConfig указан — пропускаю автовключение в config.yaml." -ForegroundColor Yellow
    Write-Host "Добавьте вручную:"
    Write-Host "  plugins:"
    Write-Host "    enabled:"
    Write-Host "      - memobase"
    $configStatus = "skipped"
} else {
    Write-Host ""
    Write-Host "MemoBase: включаю плагин в config.yaml (plugins.enabled)..." -ForegroundColor Cyan
    $statusFile = Join-Path $env:TEMP ("memobase_config_status_" + [guid]::NewGuid().ToString("N") + ".txt")
    $tmpPy = Join-Path $env:TEMP ("memobase_enable_config_" + [guid]::NewGuid().ToString("N") + ".py")
    $pyScript = @'
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
'@
    $pyScript | Set-Content -Path $tmpPy -Encoding utf8
    try {
        & $python $tmpPy $statusFile $hermesHome
    } catch {
        Write-Host "MemoBase: не удалось запустить патч конфига: $_" -ForegroundColor Red
    }
    if (Test-Path $statusFile) {
        $configStatus = (Get-Content -Path $statusFile -Raw).Trim()
        Remove-Item -Path $statusFile -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -Path $tmpPy -Force -ErrorAction SilentlyContinue
    if (-not $configStatus) { $configStatus = "failed" }
}

Write-Host ""
Write-Host "============================================================"
if ($configStatus -eq "enabled" -or $configStatus -eq "already") {
    Write-Host "MemoBase установлен и включён." -ForegroundColor Green
    Write-Host "Работает сразу в keyword-режиме (без ключей)."
    Write-Host "Семантика/ключи: hermes memobase setup"
    Write-Host "Офлайн-вектор: перезапустите .\install.ps1 -Local"
} else {
    Write-Host "MemoBase установлен, но НЕ включён в config.yaml (см. сообщения выше)." -ForegroundColor Yellow
    Write-Host "Включите вручную, затем: hermes memobase setup"
}
Write-Host "Перезапустите hermes и проверьте: /memobase status"
Write-Host "============================================================"
