#!/usr/bin/env python3
"""
ol_ia_catalog.py
================

A small Python client to query *both* Open Library and Internet Archive using a single
(author, title) record, then:

1) Deduplicate results across sources.
2) Enrich Internet Archive hits with direct download links (PDF/EPUB/TEXT/DJVU when present).
3) Produce either:
   - a list of hits,
   - a pandas.DataFrame (optional),
   - a single "scheda unica" (one consolidated work record) with grouped online links.

Dependencies:
  pip install requests pandas

Key functions:
  - unified_search(author, title, ...)
  - to_dataframe(hits)
  - scheda_unica(hits, cluster_mode="strict"|"loose", ...)

Notes:
- This is designed for historical books (late XIX / early XX) where metadata can be messy.
- Open Library language filtering can help but is not guaranteed.
- Internet Archive fields are not fully standardized; we use soft parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import re
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
    identifier: Optional[str]  # OL work key ("/works/...") or IA identifier
    url: str
    score: float
    extra: Dict[str, Any]


# -----------------------------
# Utilities
# -----------------------------
_ARTICLES = {"il", "lo", "la", "i", "gli", "le", "l", "un", "uno", "una", "the", "a", "an", "les", "le"}

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

def _main_title_loose(title: str) -> str:
    """
    Loose title normalization: remove common subtitles and trailing clauses.
    Example:
      "Saggio di teleologia: introduzione alla filosofia..." -> "saggio di teleologia"
    """
    t = (title or "").strip()
    # Split on common subtitle separators; keep the first chunk
    for sep in [":", ";", "—", "–", "-", ".", ","]:
        if sep in t:
            t = t.split(sep, 1)[0].strip()
            break
    return t

def _strip_leading_articles(t: str) -> str:
    toks = _norm(t).split()
    if toks and toks[0] in _ARTICLES:
        toks = toks[1:]
    return " ".join(toks)


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
                    # Sometimes OL already has IA identifiers; useful for bridging and dedup.
                    "ia_ids": (d.get("ia") or [])[:6],
                    "oclc_numbers": (d.get("oclc") or [])[:6],
                    "lccn": (d.get("lccn") or [])[:3],
                },
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


# -----------------------------
# Internet Archive (search + per-item metadata)
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
                    "downloads": {},  # enriched later
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
    Fetch per-item metadata from Internet Archive and extract convenient direct download links.
    Returns a dict like:
      {"pdf": "...", "epub": "...", "text": "...", "djvu": "...", "other": [...]}
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

        if lower.endswith(".pdf"):
            add("pdf", name)
        elif lower.endswith(".epub"):
            add("epub", name)
        elif lower.endswith(".txt") or lower.endswith("_text.txt"):
            add("text", name)
        elif lower.endswith(".djvu"):
            add("djvu", name)
        elif lower.endswith(".jp2") or lower.endswith(".zip"):
            out["other"].append(f"{base}/{name}")

    if not out.get("other"):
        out.pop("other", None)
    return out


# -----------------------------
# Deduplication
# -----------------------------
def _dedup_key(hit: UnifiedHit) -> str:
    """
    Stable-ish dedup key:
    - If IA id exists: unique on IA.
    - If OL contains IA ids: bridge to IA.
    - Else OL work key.
    - Else fallback to fuzzy author+title+year bucket.
    """
    if hit.source == "internetarchive" and hit.identifier:
        return f"ia:{hit.identifier}"

    ia_ids = hit.extra.get("ia_ids") if isinstance(hit.extra, dict) else None
    if ia_ids and isinstance(ia_ids, list) and len(ia_ids) > 0:
        return f"ia:{ia_ids[0]}"

    if hit.source == "openlibrary" and hit.identifier:
        return f"ol:{hit.identifier}"

    y = hit.year if hit.year is not None else 0
    y_bucket = (y // 5) * 5 if y else 0
    return f"f:{_norm(hit.author)}|{_norm(hit.title)}|{y_bucket}"

def deduplicate_hits(hits: List[UnifiedHit]) -> List[UnifiedHit]:
    """
    Deduplicate by key, keeping the best representative:
    - higher score wins
    - if tied, prefer Internet Archive (full text availability)
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
            # Merge a few extras when useful
            if cur.source == "internetarchive" and h.source == "openlibrary":
                ia_ids = h.extra.get("ia_ids")
                if ia_ids:
                    cur.extra.setdefault("ol_ia_ids", ia_ids)

    out = list(best.values())
    src_rank = {"internetarchive": 0, "openlibrary": 1}
    out.sort(key=lambda h: (-(h.score), src_rank.get(h.source, 9), h.year or 9999))
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
    """
    Run both searches, filter by min_score, optionally enrich IA hits with download links,
    then deduplicate.
    """
    ol_hits = search_openlibrary(author, title, limit=ol_limit, language=language)
    ia_hits = search_internet_archive(author, title, rows=ia_rows, language=language, year_range=year_range)

    hits = [h for h in (ol_hits + ia_hits) if h.score >= min_score]

    if enrich_internetarchive_downloads:
        for h in hits:
            if h.source == "internetarchive" and h.identifier:
                try:
                    h.extra["downloads"] = fetch_ia_downloads(h.identifier)
                except Exception as e:
                    h.extra["downloads_error"] = str(e)
                time.sleep(max(0.0, ia_politeness_delay_s))

    return deduplicate_hits(hits)


# -----------------------------
# DataFrame (Jupyter-friendly)
# -----------------------------
def to_dataframe(hits: List[UnifiedHit]):
    """
    Convert hits into a pandas DataFrame (requires pandas).
    Columns include record_url and (when present) direct IA download links.
    """
    if pd is None:
        raise RuntimeError("pandas is not installed. Install with: pip install pandas")

    rows = []
    for h in hits:
        dl = (h.extra or {}).get("downloads") if isinstance(h.extra, dict) else {}
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
# Scheda unica (single consolidated work record)
# -----------------------------
def _pick_best(hits: List[UnifiedHit]) -> UnifiedHit:
    src_rank = {"internetarchive": 0, "openlibrary": 1}
    return sorted(hits, key=lambda h: (-(h.score), src_rank.get(h.source, 9), h.year or 9999))[0]

def _canonical_work_key(h: UnifiedHit, *, cluster_mode: str) -> str:
    """
    Build a clustering key for "opera".

    cluster_mode:
      - "strict": uses full normalized author + full normalized title (minus leading articles)
      - "loose": uses normalized author + *main* title (subtitle stripped) (minus leading articles)

    Loose mode is useful for Italian philosophical works where subtitles vary across editions/records.
    """
    mode = (cluster_mode or "strict").strip().lower()
    if mode not in {"strict", "loose"}:
        raise ValueError("cluster_mode must be 'strict' or 'loose'")

    a = _norm(h.author)
    t_raw = h.title or ""

    if mode == "loose":
        t_raw = _main_title_loose(t_raw)

    t = _strip_leading_articles(t_raw)
    t = _norm(t)
    return f"{a}||{t}"

def scheda_unica(
    hits: List[UnifiedHit],
    *,
    cluster_mode: str = "strict",
    year_hint: Optional[Tuple[int, int]] = None,
    max_links_per_source: int = 6,
    max_len: int = 2600,
) -> str:
    """
    Build ONE 'scheda unica' (consolidated record) from a list of hits.
    If multiple clusters exist, chooses the cluster whose best hit has the highest score,
    and lists up to 3 alternative clusters at the end.

    Returns plain text suitable for copy/paste.
    """
    if not hits:
        return "Nessun risultato."

    # 1) Cluster
    clusters: Dict[str, List[UnifiedHit]] = defaultdict(list)
    for h in hits:
        clusters[_canonical_work_key(h, cluster_mode=cluster_mode)].append(h)

    # 2) Choose primary cluster by best score
    cluster_best: List[Tuple[str, UnifiedHit]] = [(k, _pick_best(v)) for k, v in clusters.items()]
    cluster_best.sort(key=lambda kv: -(kv[1].score))
    work_key = cluster_best[0][0]
    work_hits = clusters[work_key]
    best = _pick_best(work_hits)

    # 3) Aggregate
    authors = sorted({h.author.strip() for h in work_hits if h.author})
    author = authors[0] if authors else (best.author or "Autore n.d.")
    title = (best.title or "Titolo n.d.").strip()

    years = sorted({h.year for h in work_hits if h.year is not None})
    if years:
        year_str = str(years[0]) if len(years) == 1 else f"{years[0]}–{years[-1]} (edizioni/record)"
    else:
        year_str = "s.d."
    if year_hint:
        y0, y1 = year_hint
        year_str = f"{year_str} | filtro richiesto: {y0}–{y1}"

    languages = sorted({(h.language or "").strip() for h in work_hits if h.language})
    lang_str = ", ".join(languages) if languages else "n.d."

    # Collect compact bibliographic hints (best effort)
    publishers, isbns, edition_counts, ia_ids_from_ol = [], [], [], []
    oclc_numbers, lccn = [], []

    for h in work_hits:
        ex = h.extra or {}
        if h.source == "openlibrary":
            if ex.get("publisher"):
                publishers.extend([str(x) for x in ex.get("publisher") if x])
            if ex.get("isbn"):
                isbns.extend([str(x) for x in ex.get("isbn") if x])
            if ex.get("edition_count") is not None:
                edition_counts.append(ex.get("edition_count"))
            if ex.get("ia_ids"):
                ia_ids_from_ol.extend([str(x) for x in ex.get("ia_ids") if x])
            if ex.get("oclc_numbers"):
                oclc_numbers.extend([str(x) for x in ex.get("oclc_numbers") if x])
            if ex.get("lccn"):
                lccn.extend([str(x) for x in ex.get("lccn") if x])

    def uniq(xs: List[str], n: int) -> List[str]:
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
            if len(out) >= n:
                break
        return out

    publishers_u = uniq(publishers, 6)
    isbns_u = uniq(isbns, 6)
    ia_ids_from_ol_u = uniq(ia_ids_from_ol, 6)
    oclc_u = uniq(oclc_numbers, 6)
    lccn_u = uniq(lccn, 6)

    # Links grouped by source
    by_source: Dict[str, List[UnifiedHit]] = defaultdict(list)
    for h in work_hits:
        by_source[h.source].append(h)
    for s in by_source:
        by_source[s].sort(key=lambda h: -h.score)

    # Best digital text links
    best_pdf, best_epub, best_text, best_djvu = None, None, None, None
    if "internetarchive" in by_source:
        for h in by_source["internetarchive"]:
            dl = (h.extra or {}).get("downloads") or {}
            if isinstance(dl, dict):
                best_pdf = best_pdf or dl.get("pdf")
                best_epub = best_epub or dl.get("epub")
                best_text = best_text or dl.get("text")
                best_djvu = best_djvu or dl.get("djvu")
            if best_pdf and best_epub:
                break

    # 4) Compose text
    lines: List[str] = []
    lines.append(f"{author} — {title} ({year_str})")
    lines.append(f"Lingua/e (da record): {lang_str}")
    lines.append(f"Cluster mode: {cluster_mode}")

    if publishers_u:
        lines.append(f"Editore/i (indicativi): {publishers_u}")
    if isbns_u:
        lines.append(f"ISBN (se presenti): {isbns_u}")
    if edition_counts:
        try:
            lines.append(f"Edizioni Open Library (se disponibile): {max(int(x) for x in edition_counts if x is not None)}")
        except Exception:
            pass
    if oclc_u:
        lines.append(f"OCLC number (se presente): {oclc_u}")
    if lccn_u:
        lines.append(f"LCCN (se presente): {lccn_u}")

    lines.append("")
    lines.append("Schede online (raggruppate per fonte):")

    if "openlibrary" in by_source:
        lines.append("- Open Library:")
        c = 0
        for h in by_source["openlibrary"]:
            if h.url:
                lines.append(f"  • {h.url}  (score {h.score:.3f})")
                c += 1
            if c >= max_links_per_source:
                break

    if "internetarchive" in by_source:
        lines.append("- Internet Archive:")
        c = 0
        for h in by_source["internetarchive"]:
            if h.url:
                lines.append(f"  • {h.url}  (score {h.score:.3f})")
                c += 1
            if c >= max_links_per_source:
                break

        if ia_ids_from_ol_u:
            lines.append("  • (da Open Library, possibili item IA):")
            for iaid in ia_ids_from_ol_u[:max_links_per_source]:
                lines.append(f"    - https://archive.org/details/{iaid}")

    if best_pdf or best_epub or best_text or best_djvu:
        lines.append("")
        lines.append("Testo digitale (migliore disponibile):")
        if best_pdf:
            lines.append(f"- PDF: {best_pdf}")
        if best_epub:
            lines.append(f"- EPUB: {best_epub}")
        if best_text:
            lines.append(f"- TEXT: {best_text}")
        if best_djvu:
            lines.append(f"- DJVU: {best_djvu}")

    # Alternatives
    if len(clusters) > 1:
        lines.append("")
        lines.append("Altre opere simili trovate (cluster alternativi):")
        for k, b in cluster_best[1:4]:
            y = b.year if b.year is not None else "s.d."
            lines.append(f"- {b.author} — {b.title} ({y}) [score {b.score:.3f}]")

    out = "\n".join(lines).strip()
    if len(out) > max_len:
        out = out[: max_len - 3] + "..."
    return out


# -----------------------------
# CLI (optional)
# -----------------------------
def _print_hits(hits: List[UnifiedHit], *, max_items: int = 25) -> None:
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
            if isinstance(dl, dict):
                for k in ("pdf", "epub", "text", "djvu"):
                    if dl.get(k):
                        print(f"    download_{k}: {dl[k]}")
                        break
        print()

def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print("Usage: python ol_ia_catalog.py 'AUTHOR' 'TITLE' [--lang ita|none] [--y0 1870 --y1 1930] [--all] [--scheda] [--cluster strict|loose] [--df]")
        return 2

    author = argv[1]
    title = argv[2]
    show_all = "--all" in argv
    want_scheda = "--scheda" in argv
    want_df = "--df" in argv

    lang: Optional[str] = "ita"
    y0, y1 = 1870, 1930
    cluster_mode = "strict"

    if "--lang" in argv:
        i = argv.index("--lang")
        if i + 1 < len(argv):
            v = argv[i + 1].strip()
            lang = None if v.lower() == "none" else v

    if "--y0" in argv:
        i = argv.index("--y0")
        if i + 1 < len(argv):
            y0 = int(argv[i + 1])

    if "--y1" in argv:
        i = argv.index("--y1")
        if i + 1 < len(argv):
            y1 = int(argv[i + 1])

    if "--cluster" in argv:
        i = argv.index("--cluster")
        if i + 1 < len(argv):
            cluster_mode = argv[i + 1].strip().lower()

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

    if want_scheda:
        print(scheda_unica(hits, cluster_mode=cluster_mode, year_hint=(y0, y1)))
    else:
        _print_hits(hits, max_items=60 if show_all else 25)

    if want_df:
        df = to_dataframe(hits)
        print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
