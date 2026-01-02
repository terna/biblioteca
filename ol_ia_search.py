#!/usr/bin/env python3
"""
Unified bibliographic search for historical books (late XIX / early XX) using:
- Open Library (search.json)
- Internet Archive (advancedsearch + per-item metadata)

Adds the three requested extensions:
1) Deduplication across sources (and within each source) using stable keys where possible.
2) Extract direct download links (PDF/EPUB/TEXT) from Internet Archive item metadata.
3) Return a pandas.DataFrame (optional) for Jupyter, plus a CLI-friendly printout.

Dependencies:
  pip install requests pandas

Notes:
- OL "language=ita" is helpful but not guaranteed (metadata quality varies).
- Internet Archive language/year fields vary; we keep them as soft filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re
import sys
import time
import requests

# Optional (only needed if you call to_dataframe())
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


# -----------------------------
# Data model
# -----------------------------
@dataclass
class UnifiedHit:
    source: str  # "openlibrary" | "internetarchive"
    author: str
    title: str
    year: Optional[int]
    language: Optional[str]
    identifier: Optional[str]  # OL work key or IA identifier
    url: str
    score: float
    extra: Dict[str, Any]


# -----------------------------
# Utilities
# -----------------------------
def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[“”\"'’`]", "", s)
    s = re.sub(r"[^0-9a-zàèéìòóùç\s\-]", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_set(s: str) -> set:
    return set(_norm(s).split())


def _jaccard(a: str, b: str) -> float:
    A, B = _token_set(a), _token_set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _extract_year_from_str(s: Any) -> Optional[int]:
    if isinstance(s, list):
        s = " ".join(str(x) for x in s)
    if isinstance(s, str):
        m = re.search(r"\b(1[6-9]\d{2}|20\d{2})\b", s)
        if m:
            return _safe_int(m.group(1))
    return None


def _choose_language(lang_field: Any) -> Optional[str]:
    if isinstance(lang_field, list) and lang_field:
        return str(lang_field[0])
    if isinstance(lang_field, str):
        return lang_field
    return None


# -----------------------------
# Open Library
# -----------------------------
def search_openlibrary(
    author: str,
    title: str,
    *,
    limit: int = 10,
    language: Optional[str] = "ita",
    timeout: int = 20,
) -> List[UnifiedHit]:
    params = {"author": author, "title": title, "limit": limit}
    if language:
        params["language"] = language

    r = requests.get("https://openlibrary.org/search.json", params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    hits: List[UnifiedHit] = []
    for d in data.get("docs", []):
        hit_title = d.get("title") or ""
        hit_author = ", ".join(d.get("author_name", [])[:3]) if d.get("author_name") else ""
        year = _safe_int(d.get("first_publish_year"))
        lang = _choose_language(d.get("language"))

        key = d.get("key")  # "/works/OLxxxxW"
        url = f"https://openlibrary.org{key}" if key else "https://openlibrary.org"

        score = 0.7 * _jaccard(title, hit_title) + 0.3 * _jaccard(author, hit_author)

        hits.append(
            UnifiedHit(
                source="openlibrary",
                author=hit_author or author,
                title=hit_title or title,
                year=year,
                language=lang,
                identifier=key,
                url=url,
                score=score,
                extra={
                    "edition_count": d.get("edition_count"),
                    "publisher": (d.get("publisher") or [])[:3],
                    "isbn": (d.get("isbn") or [])[:3],
                    # Sometimes OL already has IA identifiers; useful for dedup.
                    "ia_ids": (d.get("ia") or [])[:5],
                    "oclc_numbers": (d.get("oclc") or [])[:5],
                    "lccn": (d.get("lccn") or [])[:3],
                },
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


# -----------------------------
# Internet Archive
# -----------------------------
def _build_ia_query(
    author: str,
    title: str,
    *,
    language: Optional[str],
    year_range: Optional[Tuple[int, int]],
) -> str:
    q_parts = [f'title:("{title}")', f'creator:("{author}")', "mediatype:(texts)"]
    if language:
        q_parts.append(f'(language:("{language}") OR language:("Italian"))')
    if year_range:
        y0, y1 = year_range
        q_parts.append(f"year:[{y0} TO {y1}]")
    return " AND ".join(q_parts)


def search_internet_archive(
    author: str,
    title: str,
    *,
    rows: int = 20,
    language: Optional[str] = "ita",
    year_range: Optional[Tuple[int, int]] = (1870, 1930),
    timeout: int = 25,
) -> List[UnifiedHit]:
    q = _build_ia_query(author, title, language=language, year_range=year_range)

    params = {
        "q": q,
        "fl[]": ["identifier", "title", "creator", "year", "date", "language", "publisher"],
        "rows": rows,
        "page": 1,
        "output": "json",
    }

    r = requests.get("https://archive.org/advancedsearch.php", params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    docs = (data.get("response") or {}).get("docs") or []
    hits: List[UnifiedHit] = []

    for d in docs:
        hit_title = d.get("title") or ""
        hit_author = d.get("creator") or ""
        if isinstance(hit_author, list):
            hit_author = ", ".join(hit_author[:3])

        year = _safe_int(d.get("year"))
        if year is None:
            year = _extract_year_from_str(d.get("date"))

        lang = _choose_language(d.get("language"))
        identifier = d.get("identifier")
        url = f"https://archive.org/details/{identifier}" if identifier else "https://archive.org"

        score = 0.7 * _jaccard(title, hit_title) + 0.3 * _jaccard(author, hit_author)

        hits.append(
            UnifiedHit(
                source="internetarchive",
                author=hit_author or author,
                title=hit_title or title,
                year=year,
                language=lang,
                identifier=identifier,
                url=url,
                score=score,
                extra={
                    "publisher": d.get("publisher"),
                    "query": q,
                    # will be enriched later with download links
                    "downloads": {},
                },
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


def fetch_ia_downloads(
    ia_identifier: str,
    *,
    timeout: int = 25,
) -> Dict[str, Any]:
    """
    Fetch per-item metadata from Internet Archive and extract convenient download links.
    Returns a dict like:
      {
        "pdf": "https://archive.org/download/<id>/<file>.pdf",
        "epub": ...,
        "text": ...,
        "djvu": ...,
        "other": [...]
      }
    """
    url = f"https://archive.org/metadata/{ia_identifier}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    meta = r.json()

    files = meta.get("files") or []
    if not isinstance(files, list):
        return {}

    base = f"https://archive.org/download/{ia_identifier}"
    out: Dict[str, Any] = {"other": []}

    def add(kind: str, name: str) -> None:
        link = f"{base}/{name}"
        if kind not in out:
            out[kind] = link

    for f in files:
        name = f.get("name")
        if not name or not isinstance(name, str):
            continue
        lower = name.lower()

        # Prefer common best formats for reading/citation
        if lower.endswith(".pdf"):
            add("pdf", name)
        elif lower.endswith(".epub"):
            add("epub", name)
        elif lower.endswith(".txt") or lower.endswith("_text.txt"):
            add("text", name)
        elif lower.endswith(".djvu"):
            add("djvu", name)
        elif lower.endswith(".jp2") or lower.endswith(".zip"):
            # bulky; keep as "other"
            out["other"].append(f"{base}/{name}")

    # Remove empty "other" to keep output tidy
    if not out.get("other"):
        out.pop("other", None)
    return out


# -----------------------------
# Deduplication
# -----------------------------
def _dedup_key(hit: UnifiedHit) -> str:
    """
    Build a stable-ish dedup key:
    - If IA id exists: prefer it (it is unique on IA)
    - If OL record contains IA id: that helps bridge OL<->IA
    - Else fallback: normalized author+title+year bucket
    """
    if hit.source == "internetarchive" and hit.identifier:
        return f"ia:{hit.identifier}"

    # Open Library sometimes has IA identifiers inside extra
    ia_ids = hit.extra.get("ia_ids") if isinstance(hit.extra, dict) else None
    if ia_ids and isinstance(ia_ids, list) and len(ia_ids) > 0:
        return f"ia:{ia_ids[0]}"

    if hit.source == "openlibrary" and hit.identifier:
        # OL work key is stable; but doesn't merge with IA unless ia_ids present.
        return f"ol:{hit.identifier}"

    # Fallback fuzzy
    y = hit.year if hit.year is not None else 0
    y_bucket = (y // 5) * 5 if y else 0
    return f"f:{_norm(hit.author)}|{_norm(hit.title)}|{y_bucket}"


def deduplicate_hits(hits: List[UnifiedHit]) -> List[UnifiedHit]:
    """
    Deduplicate by key, keeping the best representative:
    - higher score wins
    - if tied, prefer Internet Archive (because it can provide full text links)
    """
    best: Dict[str, UnifiedHit] = {}
    for h in hits:
        k = _dedup_key(h)
        if k not in best:
            best[k] = h
            continue

        cur = best[k]
        if (h.score > cur.score) or (
            abs(h.score - cur.score) < 1e-9 and h.source == "internetarchive" and cur.source != "internetarchive"
        ):
            best[k] = h
        else:
            # merge a few extras if useful
            # e.g., keep OL ia_ids if the chosen one is IA
            if cur.source == "internetarchive" and h.source == "openlibrary":
                ia_ids = h.extra.get("ia_ids")
                if ia_ids:
                    cur.extra.setdefault("ol_ia_ids", ia_ids)
            if cur.source == "openlibrary" and h.source == "internetarchive":
                cur.extra.setdefault("ia_identifier_alt", h.identifier)

    # Return sorted by score
    out = list(best.values())
    source_rank = {"internetarchive": 0, "openlibrary": 1}
    out.sort(key=lambda h: (-(h.score), source_rank.get(h.source, 9), h.year or 9999))
    return out


# -----------------------------
# Unified search orchestration
# -----------------------------
def unified_search(
    author: str,
    title: str,
    *,
    ol_limit: int = 15,
    ia_rows: int = 30,
    language: Optional[str] = "ita",
    year_range: Optional[Tuple[int, int]] = (1870, 1930),
    min_score: float = 0.15,
    enrich_internetarchive_downloads: bool = True,
    ia_politeness_delay_s: float = 0.25,
) -> List[UnifiedHit]:
    ol_hits = search_openlibrary(author, title, limit=ol_limit, language=language)
    ia_hits = search_internet_archive(author, title, rows=ia_rows, language=language, year_range=year_range)

    hits = [h for h in (ol_hits + ia_hits) if h.score >= min_score]

    # Enrich IA hits with per-item download links
    if enrich_internetarchive_downloads:
        for h in hits:
            if h.source == "internetarchive" and h.identifier:
                try:
                    h.extra["downloads"] = fetch_ia_downloads(h.identifier)
                except Exception as e:
                    h.extra["downloads_error"] = str(e)
                time.sleep(max(0.0, ia_politeness_delay_s))

    # Dedup across sources
    hits = deduplicate_hits(hits)
    return hits


# -----------------------------
# Presentation: print and DataFrame
# -----------------------------
def print_hits(hits: List[UnifiedHit], *, max_items: int = 25) -> None:
    if not hits:
        print("No hits found.")
        return

    for i, h in enumerate(hits[:max_items], start=1):
        year = h.year if h.year is not None else "?"
        lang = h.language if h.language is not None else "?"
        print(f"{i:02d}. [{h.source}] score={h.score:.3f} year={year} lang={lang}")
        print(f"    {h.author} — {h.title}")
        print(f"    {h.url}")

        if h.source == "internetarchive":
            dl = (h.extra or {}).get("downloads") or {}
            if dl:
                # show best available
                for k in ("pdf", "epub", "text", "djvu"):
                    if k in dl:
                        print(f"    download_{k}: {dl[k]}")
                        break

        # Show compact extras
        if h.source == "openlibrary":
            extras_compact = {k: h.extra.get(k) for k in ("edition_count", "publisher", "isbn", "ia_ids") if k in h.extra}
            if extras_compact:
                print(f"    extra: {extras_compact}")
        print()


def to_dataframe(hits: List[UnifiedHit]):
    if pd is None:
        raise RuntimeError("pandas is not installed. Install with: pip install pandas")

    rows = []
    for h in hits:
        dl = (h.extra or {}).get("downloads") if isinstance(h.extra, dict) else None
        if not isinstance(dl, dict):
            dl = {}

        rows.append(
            {
                "source": h.source,
                "score": h.score,
                "author": h.author,
                "title": h.title,
                "year": h.year,
                "language": h.language,
                "identifier": h.identifier,
                "record_url": h.url,
                "pdf": dl.get("pdf"),
                "epub": dl.get("epub"),
                "text": dl.get("text"),
                "djvu": dl.get("djvu"),
            }
        )
    return pd.DataFrame(rows).sort_values(["score", "year"], ascending=[False, True]).reset_index(drop=True)


# -----------------------------
# CLI
# -----------------------------
def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print("Usage: python ol_ia_search.py 'AUTHOR' 'TITLE' [--lang ita|none] [--y0 1870 --y1 1930] [--all] [--df]")
        return 2

    author = argv[1]
    title = argv[2]

    show_all = "--all" in argv
    want_df = "--df" in argv

    lang: Optional[str] = "ita"
    y0, y1 = 1870, 1930

    if "--lang" in argv:
        i = argv.index("--lang")
        if i + 1 < len(argv):
            val = argv[i + 1].strip()
            lang = None if val.lower() == "none" else val

    if "--y0" in argv:
        i = argv.index("--y0")
        if i + 1 < len(argv):
            y0 = int(argv[i + 1])

    if "--y1" in argv:
        i = argv.index("--y1")
        if i + 1 < len(argv):
            y1 = int(argv[i + 1])

    hits = unified_search(
        author=author,
        title=title,
        language=lang,
        year_range=(y0, y1),
        ol_limit=40 if show_all else 15,
        ia_rows=80 if show_all else 30,
        min_score=0.10 if show_all else 0.15,
        enrich_internetarchive_downloads=True,
    )

    print_hits(hits, max_items=60 if show_all else 25)

    if want_df:
        df = to_dataframe(hits)
        print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
