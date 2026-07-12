"""``memobase_map`` — mind-map of a collection.

Builds a Mermaid ``graph`` from three signal sources, each best-effort and
independently disabled if its data isn't there (never raises):

  1. **Topics per document** — a cheap TF keyword extraction over each
     document's live chunk text (stopword-filtered, RU+EN aware via
     ``stem.stem_ru`` where it can help, otherwise plain casefolded
     tokens). No LLM call — this is meant to be free and instant, not a
     "summarize every document" pipeline.
  2. **Obsidian ``[[wikilinks]]``** — re-extracted from the STORED chunk
     text via ``obsidian.extract_wikilinks`` (works even if the source
     vault directory is gone/moved — the KB already has the text), matched
     against other documents' titles/filename stems in the same
     collection.
  3. **Co-occurrence** — two documents sharing 2+ top keywords get an edge
     labeled with one shared keyword.

Output is plain text (a fenced ``mermaid`` code block) — Telegram renders
fenced code blocks fine even without Mermaid support, and any Mermaid-aware
renderer (many chat clients, Obsidian itself) picks it up as a real diagram.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("memobase.map")

_MAX_DOCS = 60  # keep the graph readable; a collection with more docs gets a clear truncation notice
_MAX_CHARS_PER_DOC = 20_000  # cap how much chunk text feeds keyword extraction per document
_TOP_KEYWORDS_PER_DOC = 5

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ][a-zA-Zа-яА-ЯёЁ0-9_-]{2,}")
_STOPWORDS = {
    "и", "в", "не", "на", "с", "что", "как", "по", "это", "из", "к", "у", "за", "от", "для", "но", "а",
    "он", "она", "они", "мы", "вы", "то", "же", "бы", "или", "если", "так", "все", "его", "её", "их",
    "the", "and", "for", "are", "but", "not", "you", "with", "this", "that", "from", "have", "was", "were",
}


def _extract_keywords(text: str, top_n: int = _TOP_KEYWORDS_PER_DOC) -> List[str]:
    counts: Counter = Counter()
    for tok in _TOKEN_RE.findall((text or "").lower()):
        if tok in _STOPWORDS:
            continue
        counts[tok] += 1
    return [w for w, _ in counts.most_common(top_n)]


def _mermaid_id(n: int) -> str:
    return f"doc{n}"


def _escape_label(label: str) -> str:
    return (label or "?").replace('"', "'").replace("\n", " ")[:80]


def _build_documents_summary(conn, collection_id: int) -> List[Dict[str, Any]]:
    from . import obsidian as obsidian_mod

    docs = conn.execute(
        "SELECT id, source_uri, title, source_type FROM documents WHERE collection_id = ? ORDER BY id",
        (collection_id,),
    ).fetchall()
    summaries: List[Dict[str, Any]] = []
    for d in docs[:_MAX_DOCS]:
        rows = conn.execute(
            "SELECT text FROM chunks WHERE document_id = ? AND collection_id = ? AND tombstoned_at IS NULL ORDER BY seq",
            (d["id"], collection_id),
        ).fetchall()
        text = ""
        for r in rows:
            if len(text) >= _MAX_CHARS_PER_DOC:
                break
            text += (r["text"] or "") + "\n"
        title = d["title"] or (d["source_uri"] or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or f"doc{d['id']}"
        wikilinks: List[str] = []
        try:
            wikilinks = obsidian_mod.extract_wikilinks(text)
        except Exception:
            logger.debug("map: wikilink extraction failed for document %s", d["id"], exc_info=True)
        summaries.append({
            "id": d["id"], "title": title, "source_uri": d["source_uri"],
            "keywords": _extract_keywords(text), "wikilinks": wikilinks,
        })
    return summaries


def _match_wikilink_target(target: str, summaries: List[Dict[str, Any]], self_id: int) -> Optional[int]:
    target_norm = (target or "").strip().lower()
    if not target_norm:
        return None
    for s in summaries:
        if s["id"] == self_id:
            continue
        title_norm = (s["title"] or "").strip().lower()
        stem_norm = (s["source_uri"] or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0].strip().lower()
        if target_norm in (title_norm, stem_norm):
            return s["id"]
    return None


def build_mind_map(conn, collection_row: Dict[str, Any]) -> str:
    """Return a fenced Mermaid ``graph LR`` block for *collection_row*'s
    live documents. Never raises — an empty/errored collection returns a
    short explanatory text instead of a broken diagram."""
    collection_id = collection_row["id"]
    try:
        summaries = _build_documents_summary(conn, collection_id)
    except Exception:
        logger.warning("map: failed to build document summaries for collection %s", collection_id, exc_info=True)
        return f"Не удалось построить карту коллекции «{collection_row.get('name')}»."

    if not summaries:
        return f"В коллекции «{collection_row.get('name')}» пока нет документов для карты."

    lines = ["```mermaid", "graph LR"]
    id_by_doc = {s["id"]: _mermaid_id(i) for i, s in enumerate(summaries)}
    for s in summaries:
        node = id_by_doc[s["id"]]
        label = _escape_label(s["title"])
        if s["keywords"]:
            label += f"<br/><small>{', '.join(s['keywords'][:3])}</small>"
        lines.append(f'  {node}["{label}"]')

    seen_edges: set = set()

    # Obsidian wikilink edges (solid, directional-ish but rendered undirected
    # for readability — a mind-map, not a strict dependency graph).
    for s in summaries:
        for target in s["wikilinks"]:
            other_id = _match_wikilink_target(target, summaries, s["id"])
            if other_id is None:
                continue
            edge = tuple(sorted((s["id"], other_id)))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            lines.append(f'  {id_by_doc[s["id"]]} --- {id_by_doc[other_id]}')

    # Keyword co-occurrence edges (dashed, labeled with the shared term).
    for i, a in enumerate(summaries):
        for b in summaries[i + 1:]:
            shared = set(a["keywords"]) & set(b["keywords"])
            if len(shared) < 2:
                continue
            edge = tuple(sorted((a["id"], b["id"])))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            label = _escape_label(sorted(shared)[0])
            lines.append(f'  {id_by_doc[a["id"]]} -.->|"{label}"| {id_by_doc[b["id"]]}')

    lines.append("```")
    header = f"Карта коллекции «{collection_row.get('name')}» ({len(summaries)} документ(ов)"
    header += ", показаны первые {})".format(_MAX_DOCS) if len(summaries) >= _MAX_DOCS else ")"
    return header + ":\n\n" + "\n".join(lines)
