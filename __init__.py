"""memobase plugin entry point.

Thin glue only, per this project's convention (see other plugins'
``__init__.py``, e.g. ``plugins/disk-cleanup/__init__.py``): wires the
tools, hook, slash command, and CLI command registered by this package's
other modules into the ``ctx`` the hermes plugin loader hands us. No logic
lives here.
"""

from __future__ import annotations

from . import backup as kb_backup
from . import cli as kb_cli
from . import commands as kb_commands
from . import tools as kb_tools
from . import wizard as kb_wizard


def register(ctx) -> None:
    kb_tools.register(ctx)      # memobase_ingest/memobase_query/memobase_ask/memobase_list/memobase_delete/memobase_status/memobase_selfcheck/memobase_map + MULTIUSER admin tools + subagent_start/pre_gateway_dispatch (identity) hooks
    kb_commands.register(ctx)   # /memobase slash command
    kb_cli.register(ctx)        # `hermes memobase ingest|list|status|reindex|backup-run|create-for|share|...`
    kb_wizard.register(ctx)     # /memobase setup onboarding wizard (pre_gateway_dispatch)
    kb_backup.register(ctx)     # no-op register(); run_doctor() is invoked via cli.py's `hermes memobase backup-run`
