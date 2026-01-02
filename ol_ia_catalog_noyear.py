#!/usr/bin/env python3
"""
ol_ia_catalog_noyear.py
======================

Unified bibliographic search for historical books using:
- Open Library
- Internet Archive

This version REMOVES any year_range filtering.
All records matching author + title are considered, regardless of date.

Features:
1) Query Open Library and Internet Archive with a single (author, title).
2) Deduplicate results across sources.
3) Enrich Internet Archive hits with direct download links (PDF/EPUB/TEXT/DJVU).
4) Produce:
   - list of hits
   - pandas.DataFrame (optional)
   - a single "scheda unica" with grouped online links
5) Supports cluster_mode = "strict" | "loose" for work aggregation.

Dependencies:
  pip install requests pandas
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, List, Optional
import re
import time
import requests

try:
    import pandas as pd
except Exception:
    pd = None


@dataclass
class UnifiedHit:
    source: str
    author: str
    title: str
    year: Optional[int]
    language: Optional[str]
    identifier: Optional[str]
    url: str
    score: float
    extra: Dict[str, Any]


_ARTICLES = {
    "il", "lo", "la", "i", "gli", "le", "l", "un", "uno", "una",
    "the", "a", "an", "les", "le"
}


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[“”\"'’`]", "", s)
    s = re.sub(r"[^0-9a-zàèéìòóùç\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _token_set(s: str) -> set:
    return set(_norm(s).split())


def _jaccard(a: str, b: str) -> float:
    A, B = _token_set(a), _token_set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _choose_language(x):
    if isinstance(x, list) and x:
        return str(x[0])
    if isinstance(x, str):
        return x
    return None


def _main_title_loose(title: str) -> str:
    for sep in [":", ";", "—", "–", "-", ".", ","]:
        if sep in title:
            return title.split(sep, 1)[0].strip()
    return title.strip()


def _strip_articles(title: str) -> str:
    toks = _norm(title).split()
    if toks and toks[0] in _ARTICLES:
        toks = toks[1:]
    return " ".join(toks)


def search_openlibrary(author: str, title: str, limit: int = 15, language: Optional[str] = "ita"):
    r = requests.get(
        "https://openlibrary.org/search.json",
        params={"author": author, "title": title, "limit": limit, "language": language},
        timeout=20,
    )
    r.raise_for_status()
    docs = r.json().get("docs", [])

    hits = []
    for d in docs:
        htitle = d.get("title", "")
        hauthor = ", ".join(d.get("author_name", [])[:3]) if d.get("author_name") else author
        year = _safe_int(d.get("first_publish_year"))
        lang = _choose_language(d.get("language"))
        key = d.get("key")
        url = f"https://openlibrary.org{key}" if key else "https://openlibrary.org"

        score = 0.7 * _jaccard(title, htitle) + 0.3 * _jaccard(author, hauthor)

        hits.append(
            UnifiedHit(
                source="openlibrary",
                author=hauthor,
                title=htitle,
                year=year,
                language=lang,
                identifier=key,
                url=url,
                score=score,
                extra={
                    "publisher": (d.get("publisher") or [])[:3],
                    "isbn": (d.get("isbn") or [])[:3],
                    "ia_ids": (d.get("ia") or [])[:6],
                    "edition_count": d.get("edition_count"),
                },
            )
        )
    return sorted(hits, key=lambda h: -h.score)


def search_internet_archive(author: str, title: str, rows: int = 30):
    q = f'title:("{title}") AND creator:("{author}") AND mediatype:(texts)'

    r = requests.get(
        "https://archive.org/advancedsearch.php",
        params={
            "q": q,
            "fl[]": ["identifier", "title", "creator", "year", "language", "publisher"],
            "rows": rows,
            "output": "json",
        },
        timeout=25,
    )
    r.raise_for_status()
    docs = (r.json().get("response") or {}).get("docs", [])

    hits = []
    for d in docs:
        htitle = d.get("title", "")
        hauthor = d.get("creator", author)
        if isinstance(hauthor, list):
            hauthor = ", ".join(hauthor[:3])
        year = _safe_int(d.get("year"))
        lang = _choose_language(d.get("language"))
        ident = d.get("identifier")
        url = f"https://archive.org/details/{ident}" if ident else "https://archive.org"

        score = 0.7 * _jaccard(title, htitle) + 0.3 * _jaccard(author, hauthor)

        hits.append(
            UnifiedHit(
                source="internetarchive",
                author=hauthor,
                title=htitle,
                year=year,
                language=lang,
                identifier=ident,
                url=url,
                score=score,
                extra={"publisher": d.get("publisher"), "downloads": {}},
            )
        )
    return sorted(hits, key=lambda h: -h.score)


def fetch_ia_downloads(identifier: str):
    r = requests.get(f"https://archive.org/metadata/{identifier}", timeout=25)
    r.raise_for_status()
    files = r.json().get("files", [])

    base = f"https://archive.org/download/{identifier}"
    out = {}

    for f in files:
        name = f.get("name", "")
        lname = name.lower()
        if lname.endswith(".pdf"):
            out.setdefault("pdf", f"{base}/{name}")
        elif lname.endswith(".epub"):
            out.setdefault("epub", f"{base}/{name}")
        elif lname.endswith(".txt"):
            out.setdefault("text", f"{base}/{name}")
        elif lname.endswith(".djvu"):
            out.setdefault("djvu", f"{base}/{name}")
    return out


def unified_search(author: str, title: str, language: Optional[str] = "ita"):
    hits = []
    hits.extend(search_openlibrary(author, title, language=language))
    hits.extend(search_internet_archive(author, title))

    for h in hits:
        if h.source == "internetarchive" and h.identifier:
            try:
                h.extra["downloads"] = fetch_ia_downloads(h.identifier)
                time.sleep(0.25)
            except Exception:
                pass

    seen = {}
    for h in hits:
        key = (
            f"ia:{h.identifier}"
            if h.source == "internetarchive" and h.identifier
            else f"{_norm(h.author)}|{_norm(h.title)}"
        )
        if key not in seen or h.score > seen[key].score:
            seen[key] = h

    return sorted(seen.values(), key=lambda h: -h.score)


def to_dataframe(hits):
    if pd is None:
        raise RuntimeError("pandas not installed")
    rows = []
    for h in hits:
        dl = h.extra.get("downloads", {})
        rows.append({
            "source": h.source,
            "author": h.author,
            "title": h.title,
            "year": h.year,
            "language": h.language,
            "record_url": h.url,
            "pdf": dl.get("pdf"),
            "epub": dl.get("epub"),
            "text": dl.get("text"),
            "djvu": dl.get("djvu"),
            "score": h.score,
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def scheda_unica(hits, cluster_mode: str = "strict"):
    clusters = defaultdict(list)

    for h in hits:
        t = h.title
        if cluster_mode == "loose":
            t = _main_title_loose(t)
        key = f"{_norm(h.author)}||{_strip_articles(t)}"
        clusters[key].append(h)

    best_cluster = max(clusters.values(), key=lambda hs: max(h.score for h in hs))
    best = max(best_cluster, key=lambda h: h.score)

    lines = [
        f"{best.author} — {best.title} ({best.year or 's.d.'})",
        f"Cluster mode: {cluster_mode}",
        "",
        "Schede online:"
    ]

    for h in sorted(best_cluster, key=lambda h: -h.score):
        lines.append(f"- {h.source}: {h.url}")
        if h.source == "internetarchive":
            for k, v in h.extra.get("downloads", {}).items():
                lines.append(f"    {k.upper()}: {v}")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Import this module in Python or Jupyter. CLI intentionally minimal.")
