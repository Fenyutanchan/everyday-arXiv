"""
JSON-based persistence for arXiv papers.

Storage model
~~~~~~~~~~~~~
Each paper is tracked by its **base arXiv ID** (version stripped).  Papers are
filed under date-keyed JSON files according to the following rules:

1. **First submission** — every paper appears in the file for its
   ``published`` date (the day it was originally submitted).
2. **Replacement** — if a paper has been revised (``updated`` ≠ ``published``),
   it also appears in the file for its latest ``updated`` date.
3. **Intermediate cleanup** — when a paper accumulates multiple revisions, only
   the *first submission* date and the *latest replacement* date are kept.
   Any records on dates in between are automatically removed.

Two tiers of data
~~~~~~~~~~~~~~~~~
**Raw** (``data/YYYY-MM-DD.json``)
    Fetched directly from arXiv.  Each paper dict has no ``relevance`` field.

**Filtered** (``relevant/YYYY-MM-DD.json``)
    Produced by the LLM filter step.  Organized by **filter run date**, not
    article date.  Each paper dict additionally contains a ``relevance`` dict
    with ``relevant``, ``score``, ``reason``, ``pass_number``, and ``evaluated`` fields.

Both tiers share the same multi-date filing logic and de-duplication behaviour.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Literal

from .fetcher import Paper
from .filter import FilterResult

logger = logging.getLogger(__name__)

Tier = Literal["raw", "filtered"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_id(arxiv_id: str) -> str:
    """Strip the version suffix from an arXiv ID.

    ``2604.15309v2`` → ``2604.15309``
    """
    return re.sub(r"v\d+$", "", arxiv_id)


def _date_from_iso(iso_str: str) -> str:
    """Extract ``YYYY-MM-DD`` from an ISO-8601 datetime string."""
    return iso_str[:10]


def _paper_to_raw_dict(paper: Paper) -> dict:
    """Serialize a :class:`Paper` into a JSON-friendly dict (no relevance)."""
    return {
        "arxiv_id": paper.arxiv_id,
        "base_id": _base_id(paper.arxiv_id),
        "title": paper.title,
        "authors": paper.authors,
        "abstract": paper.abstract,
        "categories": paper.categories,
        "published": paper.published,
        "updated": paper.updated,
        "pdf_url": paper.pdf_url,
        "html_url": paper.html_url,
        "entry_url": paper.entry_url,
    }


def _result_to_dict(result: FilterResult) -> dict:
    """Serialize a :class:`FilterResult` — raw dict + full relevance metadata."""
    d = _paper_to_raw_dict(result.paper)
    d["relevance"] = {
        "relevant": result.relevant,
        "score": result.score,
        "reason": result.reason,
        "pass_number": result.pass_number,
        "evaluated": result.evaluated,
    }
    return d


# ---------------------------------------------------------------------------
# Low-level file I/O
# ---------------------------------------------------------------------------

def _day_filename(date_str: str, tier: Tier) -> str:
    """Return the filename for a given date and tier."""
    if tier == "filtered":
        return f"{date_str}.filtered.json"
    return f"{date_str}.json"


def _load_day(output_dir: Path, date_str: str, tier: Tier) -> dict:
    """Load a day file.  Returns an empty skeleton if it does not exist."""
    path = output_dir / _day_filename(date_str, tier)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"date": date_str, "papers": []}


def _save_day(output_dir: Path, data: dict, tier: Tier) -> Path:
    """Write a day file and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = _day_filename(data["date"], tier)
    path = output_dir / filename
    data["total"] = len(data["papers"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def _find_paper_index(day_data: dict, base_id: str) -> int:
    """Return the list index of a paper in *day_data* by its base ID, or -1."""
    for i, p in enumerate(day_data.get("papers", [])):
        if p.get("base_id") == base_id or _base_id(p.get("arxiv_id", "")) == base_id:
            return i
    return -1


# ---------------------------------------------------------------------------
# Core operations on a single paper
# ---------------------------------------------------------------------------

def _upsert_paper(output_dir: Path, date_str: str, paper_dict: dict, tier: Tier) -> None:
    """Insert or update *paper_dict* in the file for *date_str* and *tier*."""
    day = _load_day(output_dir, date_str, tier)
    idx = _find_paper_index(day, paper_dict["base_id"])
    if idx >= 0:
        day["papers"][idx] = paper_dict
    else:
        day["papers"].append(paper_dict)
    _save_day(output_dir, day, tier)


def _remove_paper(output_dir: Path, date_str: str, base_id: str, tier: Tier) -> None:
    """Remove a paper from the file for *date_str* and *tier* (no-op if absent)."""
    path = output_dir / _day_filename(date_str, tier)
    if not path.exists():
        return
    day = _load_day(output_dir, date_str, tier)
    idx = _find_paper_index(day, base_id)
    if idx >= 0:
        day["papers"].pop(idx)
        if day["papers"]:
            _save_day(output_dir, day, tier)
        else:
            path.unlink()
            logger.info("Removed empty file %s", path)
        logger.debug("Removed %s from %s (%s)", base_id, date_str, tier)


def _cleanup_intermediate(
    output_dir: Path,
    base_id: str,
    first_date: str,
    latest_date: str,
    tier: Tier,
) -> None:
    """Remove *base_id* from any day files strictly between *first_date* and
    *latest_date* for the given *tier*."""
    if first_date >= latest_date:
        return

    suffix = ".filtered.json" if tier == "filtered" else ".json"
    for json_path in sorted(output_dir.glob(f"*{suffix}")):
        # stem for filtered is "2026-04-18.filtered", for raw is "2026-04-18"
        file_date = json_path.name.split(".")[0]
        if first_date < file_date < latest_date:
            _remove_paper(output_dir, file_date, base_id, tier)


def _save_many(
    items: list[tuple[dict, str, str]],
    output_dir: Path,
    tier: Tier,
) -> list[Path]:
    """Common save logic shared by :func:`save_papers` and :func:`save_filtered`.

    *items* is a list of ``(paper_dict, first_date, latest_date)`` tuples.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    modified_dates: set[str] = set()

    for paper_dict, first_date, latest_date in items:
        base_id = paper_dict["base_id"]

        _upsert_paper(output_dir, first_date, paper_dict, tier)
        modified_dates.add(first_date)

        if latest_date != first_date:
            _upsert_paper(output_dir, latest_date, paper_dict, tier)
            modified_dates.add(latest_date)
            _cleanup_intermediate(output_dir, base_id, first_date, latest_date, tier)

    suffix = ".filtered.json" if tier == "filtered" else ".json"
    paths = []
    for d in sorted(modified_dates):
        p = output_dir / _day_filename(d, tier)
        if p.exists():
            paths.append(p)

    logger.info(
        "Saved %d papers across %d date files (%s): %s",
        len(items),
        len(paths),
        tier,
        [p.name for p in paths],
    )
    return paths


# ---------------------------------------------------------------------------
# Public API — Raw tier
# ---------------------------------------------------------------------------

def save_papers(
    papers: list[Paper],
    output_dir: str | Path = "data",
) -> list[Path]:
    """Persist raw (unfiltered) papers according to the multi-date storage model.

    Parameters
    ----------
    papers:
        Papers fetched from arXiv.
    output_dir:
        Directory that holds the daily JSON files.

    Returns
    -------
    list[Path]
        Paths to all day files that were modified.
    """
    items = []
    for paper in papers:
        first_date = _date_from_iso(paper.published)
        latest_date = _date_from_iso(paper.updated)
        items.append((_paper_to_raw_dict(paper), first_date, latest_date))
    return _save_many(items, Path(output_dir), tier="raw")


# ---------------------------------------------------------------------------
# Public API — Filtered tier
# ---------------------------------------------------------------------------

def save_filtered(
    results: list[FilterResult],
    run_date: str,
    output_dir: str | Path = "relevant",
) -> list[Path]:
    """Persist LLM-filtered papers into a single file keyed by *run_date*.

    Unlike raw data (which is filed by article published/updated date),
    filtered output is organized by **when the filter was executed**.

    Parameters
    ----------
    results:
        Filtered paper results to persist.
    run_date:
        The date the filter was run (``YYYY-MM-DD``).
    output_dir:
        Directory that holds the filtered output files.

    Returns
    -------
    list[Path]
        Paths to all files that were modified.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paper_dicts = [_result_to_dict(r) for r in results]

    path = out / f"{run_date}.json"
    data = {"date": run_date, "total": len(paper_dicts), "papers": paper_dicts}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d filtered papers to %s", len(paper_dicts), path.name)
    return [path]


def save_trash(
    discarded: list[FilterResult],
    run_date: str,
    output_dir: str | Path = "trash",
) -> list[Path]:
    """Persist discarded papers (with rejection reasons) into a trash file.

    Merges with any existing trash for *run_date* by ``base_id`` — new entries
    are appended; existing entries with the same ``base_id`` are replaced.

    Parameters
    ----------
    discarded:
        Discarded paper results with LLM-generated reasons.
    run_date:
        The date the filter was run (``YYYY-MM-DD``).
    output_dir:
        Directory that holds the trash files.

    Returns
    -------
    list[Path]
        Paths to all files that were modified.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    new_dicts = [_result_to_dict(r) for r in discarded]

    path = out / f"{run_date}.json"
    existing: list[dict] = []
    if path.exists():
        with open(path) as f:
            existing = json.load(f).get("papers", [])

    existing_index = {_base_id(p.get("arxiv_id", "")): i for i, p in enumerate(existing)}
    for d in new_dicts:
        base = d.get("base_id", _base_id(d.get("arxiv_id", "")))
        if base in existing_index:
            existing[existing_index[base]] = d
        else:
            existing.append(d)
            existing_index[base] = len(existing) - 1

    data = {"date": run_date, "total": len(existing), "papers": existing}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d discarded papers to %s", len(existing), path.name)
    return [path]


# ---------------------------------------------------------------------------
# Public API — Reading
# ---------------------------------------------------------------------------

def load_raw(date_str: str, output_dir: str | Path = "data") -> list[dict]:
    """Load all raw paper dicts for a given date."""
    day = _load_day(Path(output_dir), date_str, "raw")
    return day.get("papers", [])


def latest_raw_date(output_dir: str | Path = "data") -> str | None:
    """Return the latest date string found in the raw data directory."""
    out = Path(output_dir)
    if not out.is_dir():
        return None
    dates = sorted(f.stem for f in out.glob("*.json"))
    return dates[-1] if dates else None


def load_filtered(date_str: str, output_dir: str | Path = "relevant") -> list[dict]:
    """Load filtered paper dicts for a given run date."""
    path = Path(output_dir) / f"{date_str}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f).get("papers", [])
    return []


def load_all_filtered(output_dir: str | Path = "relevant") -> list[dict]:
    """Load all filtered paper dicts from every run-date file.

    Returns papers sorted by file date (oldest first) so that later runs
    naturally override earlier ones when building a dedup index.
    """
    out = Path(output_dir)
    if not out.exists():
        return []
    all_papers: list[dict] = []
    for path in sorted(out.glob("*.json")):
        with open(path) as f:
            all_papers.extend(json.load(f).get("papers", []))
    return all_papers


def load_results(path: str | Path) -> dict:
    """Read an arbitrary JSON file and return the parsed dict."""
    with open(path) as f:
        return json.load(f)


# Backward compatibility — the old public name still works.
# Remove once all callers are migrated.
save_results = save_filtered
