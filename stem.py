"""Russian stemming for memobase's FTS5 BM25 leg (`chunks_fts.text_stem`).

STEMMER DECISION (made during this build, verified 2026-07-06): **PyStemmer**
(BSD-2-Clause; wraps the C Snowball library, which itself ships a Russian
stemming algorithm). Verified via ``pip download --only-binary=:all:`` that
``PyStemmer==3.1.0`` publishes prebuilt wheels for cp311 on BOTH platforms
this plugin targets:

  * ``pystemmer-3.1.0-cp311-cp311-win_amd64.whl``
  * ``pystemmer-3.1.0-cp311-cp311-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl``

No compiler needed on either the Windows dev box or a Linux VPS — a plain
``pip install PyStemmer`` (already the install.ps1/install.sh dependency
list's job) is enough. The fallback plan in DESIGN_v1.md — vendoring a
~200-line pure-Python snowball-ru — is therefore NOT needed for v1.

This module MUST import cleanly even before PyStemmer is installed (e.g.
right after these foundation modules are written, before install.ps1 has
run, or in a stripped-down test env) — the ``Stemmer`` import is deferred
into the first call to :func:`stem_ru`, and any failure degrades to a
lowercase-tokenized passthrough with a one-time warning, never an
``ImportError`` at module load time.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Optional, Union

logger = logging.getLogger("memobase.stem")

_lock = threading.Lock()
_stemmer: Optional[Union[object, bool]] = None  # None=not checked yet, False=unavailable
_unavailable_warned = False

# Letters only (Unicode-aware) — punctuation/digits are token boundaries and
# are dropped from the stemmed output. This side-channel (`text_stem`) is
# used ONLY for FTS matching; citations always quote the raw, unstemmed
# chunk text (see answer.py's quote-verification step) so dropping
# punctuation here can never corrupt a citation.
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _get_stemmer():
    global _stemmer, _unavailable_warned
    if _stemmer is not None:
        return _stemmer
    with _lock:
        if _stemmer is not None:
            return _stemmer
        try:
            import Stemmer  # PyStemmer

            _stemmer = Stemmer.Stemmer("russian")
        except ImportError:
            if not _unavailable_warned:
                logger.warning(
                    "PyStemmer not installed - stem_ru() will pass text through "
                    "lowercased/tokenized but UNSTEMMED (RU BM25 leg degraded to "
                    "exact-token match only). Run install.ps1/install.sh to fix."
                )
                _unavailable_warned = True
            _stemmer = False
    return _stemmer


def stem_ru(text: str) -> str:
    """Return *text* with each word replaced by its Russian Snowball stem,
    space-joined (e.g. ``"договора"`` and ``"договор"`` both stem to the
    same token, so FTS matches across inflected forms).

    Degrades to a lowercase, tokenized (but unstemmed) passthrough if
    PyStemmer is not installed or raises for any reason — this function
    never raises and never returns ``None``. Empty/falsy input returns
    ``""``.

    NOT for display or citations — this is purely an index/query-time
    transform for the FTS5 ``text_stem`` shadow column; the raw chunk text
    is what gets shown to the user and verified against quotes.
    """
    if not text:
        return ""

    stemmer = _get_stemmer()
    words = _WORD_RE.findall(text.lower())
    if not words:
        return ""

    if stemmer is False:
        return " ".join(words)

    try:
        stemmed = stemmer.stemWords(words)
    except Exception:
        logger.debug("PyStemmer.stemWords failed; falling back to raw tokens", exc_info=True)
        return " ".join(words)

    return " ".join(stemmed)
