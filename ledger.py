"""KB external-spend ledger.

Cloudflare (embed) and Cohere (rerank) calls happen entirely outside any
LLM-call cost tracking the host application does (they are plain HTTP calls
this plugin makes, not LLM API calls hermes tracks) — so this module is the
ONLY place their dollar cost is recorded and capped.

Public surface (used by embed.py, rerank.py, ingest.py):

    record_call(conn, provider, op, units=None, est_usd=None, collection_id=None) -> int
    estimate_cost_usd(provider, op, units) -> float
    check_monthly_ceiling(conn, provider, memobase_cfg) -> (within_ceiling, spent_so_far, ceiling)
    ensure_within_ceiling(conn, provider, memobase_cfg)  # raises LedgerError if already over

Pricing is a best-effort estimate — these
constants should be reviewed periodically against the providers' current
published pricing, not treated as billing-accurate.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import sqlite3
from typing import List, Optional, Tuple

from . import db

logger = logging.getLogger("memobase.ledger")


class LedgerError(RuntimeError):
    """Raised by :func:`ensure_within_ceiling` when a provider's monthly
    spend ceiling has already been reached — caller must not start the job."""


# (provider, op) -> $ per billing unit. Unit meaning is documented at each
# call site (embed.py passes "thousands of approx tokens" for cloudflare
# embed calls; rerank.py — built by another agent — is expected to pass
# "number of rerank search-units" for cohere rerank calls).
#
# Cloudflare Workers AI @cf/baai/bge-m3: priced per-token, on the order of
# ~$0.012 / 1M input tokens as of Cloudflare's public Workers AI pricing
# page — expressed here as $ per 1K tokens.
# Cohere rerank-v3.5: priced per "search unit" (roughly: 1 query x up to 100
# documents), historically ~$2.00 / 1000 search units.
_PRICING_USD: dict = {
    ("cloudflare", "embed"): 0.000012,   # $ per 1,000 approx-tokens
    ("local", "embed"): 0.0,             # on-box fastembed: free (listed so no "no pricing" warning per call)
    ("cohere", "rerank"): 0.002,         # $ per rerank search-unit
    # Apify actors (pay-per-result; same figures as youtube.estimate_channel_cost_usd)
    ("apify", "channel_list"): 0.0005,   # $ per listed video
    ("apify", "transcript"): 0.0005,     # $ per video transcript
    ("apify", "audio_download"): 0.002,  # $ per audio minute
    ("scrapecreators", "request"): 0.0,  # credit-based; recorded for visibility only
    # STT, $ per second of audio
    ("groq", "stt"): 0.04 / 3600.0,      # whisper-large-v3-turbo, $0.04/hour
    ("gemini", "stt"): 0.00001,          # gemini-flash-lite-latest audio input, ~32 tok/s
}

_DEFAULT_FALLBACK_PRICE = 0.0  # unknown (provider, op): estimate as free rather than guess wildly


def estimate_cost_usd(provider: str, op: str, units: float) -> float:
    """Best-effort dollar estimate for *units* of (*provider*, *op*).

    Returns 0.0 for an unrecognized (provider, op) pair rather than raising
    — callers should not let a pricing-table gap block an otherwise-valid
    operation, but SHOULD log/surface that the estimate is unknown (spend is
    still recorded with ``est_usd=0.0`` and remains visible in the ledger for
    manual review).
    """
    price = _PRICING_USD.get((provider, op))
    if price is None:
        logger.warning("ledger: no pricing entry for (%s, %s); estimating $0.00", provider, op)
        price = _DEFAULT_FALLBACK_PRICE
    return max(0.0, float(units)) * price


def record_call(
    conn: sqlite3.Connection,
    *,
    provider: str,
    op: str,
    units: Optional[float] = None,
    est_usd: Optional[float] = None,
    collection_id: Optional[int] = None,
    user_id: Optional[str] = None,
) -> int:
    """Append one row to the ``spend`` table. Thin, single-call-site wrapper
    around :func:`db.record_spend` so every module (embed.py/rerank.py/
    ingest.py) records spend the same way. If ``est_usd`` is not supplied,
    it is computed via :func:`estimate_cost_usd` (requires ``units``).
    ``user_id`` attributes the spend to a
    guest for their daily-budget check — pass it whenever the call was
    triggered by a guest's own ingest, leave it None for owner-initiated
    spend (never budget-limited).
    """
    if est_usd is None and units is not None:
        est_usd = estimate_cost_usd(provider, op, units)
    return db.record_spend(
        conn, provider=provider, op=op, units=units, est_usd=est_usd,
        collection_id=collection_id, user_id=user_id,
    )


# ---------------------------------------------------------------------------
# External calls made deep inside extractors (Apify / ScrapeCreators / STT)
# that have no DB connection of their own: they NOTE the call into a
# per-context bucket, and the orchestrator holding `conn` (ingest_source,
# ingest_channel, the raw YouTube tools) flushes it into `spend` with the
# right collection_id/user_id.
# ---------------------------------------------------------------------------

_pending_spend: "contextvars.ContextVar[Optional[List[tuple]]]" = contextvars.ContextVar(
    "memobase_pending_spend", default=None
)


@contextlib.contextmanager
def collect_external_spend():
    """Collect :func:`note_external_call` notes made inside this block."""
    bucket: List[tuple] = []
    token = _pending_spend.set(bucket)
    try:
        yield bucket
    finally:
        _pending_spend.reset(token)


def note_external_call(provider: str, op: str, units: float) -> None:
    """Note one paid external call; recorded when the enclosing
    :func:`collect_external_spend` block is flushed. Never raises."""
    bucket = _pending_spend.get()
    if bucket is None:
        logger.debug("ledger: external call %s/%s outside a spend collector (not recorded)", provider, op)
        return
    try:
        bucket.append((provider, op, max(0.0, float(units or 0.0))))
    except (TypeError, ValueError):
        bucket.append((provider, op, 0.0))


def flush_external_spend(conn: sqlite3.Connection, bucket: List[tuple], *,
                         collection_id: Optional[int] = None, user_id: Optional[str] = None) -> float:
    """Record every noted call into ``spend``; returns the summed $ estimate.
    Bookkeeping failures are logged, never raised."""
    total = 0.0
    while bucket:
        provider, op, units = bucket.pop(0)
        est = estimate_cost_usd(provider, op, units)
        total += est
        try:
            record_call(conn, provider=provider, op=op, units=units, est_usd=est,
                        collection_id=collection_id, user_id=user_id)
        except Exception:  # noqa: BLE001
            logger.error("ledger: failed to record %s/%s spend", provider, op, exc_info=True)
    return total


def check_monthly_ceiling(
    conn: sqlite3.Connection, provider: str, memobase_cfg: dict, *, estimated_usd: float = 0.0
) -> Tuple[bool, float, float]:
    """Return ``(within_ceiling, spent_so_far_usd, ceiling_usd)`` for
    *provider* over the trailing 30 days, per ``memobase.monthly_ceiling_usd.<provider>``.

    A provider with no configured ceiling (or a ceiling of 0/None, meaning
    "not tracked") is always reported ``within_ceiling=True`` — callers that
    want a hard "0 = forbidden" semantics should check ``ceiling_usd`` itself
    rather than relying on this function to enforce it.
    """
    ceiling = (memobase_cfg.get("monthly_ceiling_usd") or {}).get(provider)
    spent = db.monthly_spend(conn, provider)
    if ceiling is None:
        return True, spent, float("inf")
    ceiling = float(ceiling)
    estimated = max(0.0, float(estimated_usd or 0.0))
    # spent + estimated: a job that would push spend over the ceiling is
    # refused up front, not only the one after the ceiling was crossed.
    within = spent < ceiling and (spent + estimated) <= ceiling
    return within, spent, ceiling


def ensure_within_ceiling(conn: sqlite3.Connection, provider: str, memobase_cfg: dict, *,
                          estimated_usd: float = 0.0) -> None:
    """Raise :class:`LedgerError` if *provider*'s trailing-30-day spend has
    already reached/exceeded its configured monthly ceiling. Callers
    (ingest.py, before starting a costly embed/rerank job) should call this
    BEFORE spending more, not after — it does not itself prevent the
    in-flight call that triggered it from completing.
    """
    within, spent, ceiling = check_monthly_ceiling(conn, provider, memobase_cfg, estimated_usd=estimated_usd)
    if not within:
        raise LedgerError(
            f"{provider} monthly spend ceiling reached: ${spent:.4f} spent + ${estimated_usd:.4f} estimated "
            f"> ${ceiling:.4f} ceiling (memobase.monthly_ceiling_usd.{provider}); job refused"
        )
