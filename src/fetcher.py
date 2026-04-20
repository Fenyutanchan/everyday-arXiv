"""
ArXiv paper fetcher — retrieves recent papers from specified categories.

This module wraps the ``arxiv`` PyPI package and exposes a single high-level
function :func:`fetch_papers` that returns a list of :class:`Paper` dataclass
instances suitable for downstream filtering.

Typical usage::

    from src.fetcher import fetch_papers

    papers = fetch_papers(["hep-ph", "astro-ph.CO"], days=1, max_results=100)
    for p in papers:
        print(p.arxiv_id, p.title)
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

import arxiv


@dataclass
class Paper:
    """Normalized representation of a single arXiv preprint."""

    arxiv_id: str
    """Unique arXiv identifier (e.g. ``2301.01234v1``)."""

    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    """arXiv categories the paper belongs to (primary + cross-lists)."""

    published: str
    """ISO-8601 datetime string of the original submission."""

    updated: str
    """ISO-8601 datetime string of the most recent update."""

    pdf_url: str
    entry_url: str
    """Canonical arXiv abstract page URL."""

    html_url: str
    """URL to the arXiv HTML rendered version, or ``"N/A"`` if none exists."""


def _build_html_url(arxiv_id: str) -> str:
    """Return the arXiv HTML page URL for a given arXiv ID."""
    return f"https://arxiv.org/html/{arxiv_id}"


async def _check_html_available(arxiv_id: str) -> tuple[str, str]:
    """Check whether the HTML version of a paper is available.

    Returns a tuple of ``(arxiv_id, html_url_or_notice)``.
    """
    import httpx

    url = _build_html_url(arxiv_id)
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.head(url, timeout=10.0)
            if resp.status_code == 200:
                return arxiv_id, url
    except Exception:
        pass
    return arxiv_id, "N/A"


async def _resolve_html_urls(papers: list[Paper]) -> None:
    """Populate the ``html_url`` field of each paper in-place.

    Checks are performed concurrently to avoid blocking on slow responses.
    Papers without an HTML rendering get an unavailable notice.
    """
    import asyncio

    sem = asyncio.Semaphore(10)

    async def _limited(aid: str) -> tuple[str, str]:
        async with sem:
            return await _check_html_available(aid)

    results = await asyncio.gather(*[_limited(p.arxiv_id) for p in papers])
    lookup = dict(results)
    for p in papers:
        p.html_url = lookup.get(p.arxiv_id, "N/A")


def fetch_papers(
    categories: list[str],
    days: int = 1,
    page_size: int = 100,
) -> list[Paper]:
    """Fetch recent arXiv papers from the given categories.

    Papers are sorted by **last updated date** (descending), so the results
    include both new submissions and recently replaced / cross-listed papers.
    The client auto-paginates until it encounters a paper whose ``updated``
    field falls outside the requested *days* window — there is **no hard cap**
    on the total number of papers returned.

    Parameters
    ----------
    categories:
        A list of arXiv category codes (e.g. ``["hep-ph", "astro-ph.CO"]``).
        See the `full taxonomy <https://arxiv.org/category_taxonomy>`_ for
        available values.
    days:
        Only include papers updated within the last *days* days (UTC).
    page_size:
        Number of results per API request (passed to :class:`arxiv.Client`).
        Controls granularity of pagination; does **not** limit total results.

    Returns
    -------
    list[Paper]
        Papers sorted by last-updated date (newest first), each guaranteed to
        have been updated within the requested window.
    """
    query = " OR ".join(f"cat:{cat}" for cat in categories)

    since = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(days=days)
    ).strftime("%Y%m%d%H%M%S")

    client = arxiv.Client(page_size=page_size)
    search = arxiv.Search(
        query=query,
        max_results=10_000,
        sort_by=arxiv.SortCriterion.LastUpdatedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: list[Paper] = []
    for result in client.results(search):
        if result.updated.strftime("%Y%m%d%H%M%S") < since:
            break
        papers.append(
            Paper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title.replace("\n", " "),
                authors=[a.name for a in result.authors],
                abstract=result.summary.replace("\n", " "),
                categories=result.categories,
                published=result.published.isoformat(),
                updated=result.updated.isoformat(),
                pdf_url=result.pdf_url,
                entry_url=result.entry_id,
                html_url="",  # resolved asynchronously below
            )
        )

    return papers
