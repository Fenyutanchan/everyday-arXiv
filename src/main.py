"""
CLI entry point with two subcommands: ``fetch`` and ``filter``.

Usage::

    # Fetch papers from arXiv and save raw data
    uv run python -m src.main fetch --days 1

    # Run LLM filter on previously fetched raw data
    uv run python -m src.main filter --date 2026-04-19

Or via the console script::

    uv run everyday-arxiv fetch --days 1
    uv run everyday-arxiv filter --date 2026-04-19
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .fetcher import Paper, fetch_papers, _resolve_html_urls
from .filter import FilterOutcome, FilterResult, LLMConfig, filter_papers
from .storage import save_papers, save_filtered, save_trash, load_raw, load_filtered, load_all_filtered

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def load_config(path: str | Path) -> dict:
    """Load and parse the YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_llm_config(cfg: dict) -> LLMConfig:
    """Construct an :class:`LLMConfig` from the ``llm`` section of the config."""
    from .filter import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT_TEMPLATE
    llm = cfg.get("llm", {})

    base_url_env = llm.get("base_url_env", "LLM_BASE_URL")
    base_url = os.environ.get(base_url_env, llm.get("base_url", "https://api.openai.com/v1"))

    model_env = llm.get("model_env", "LLM_MODEL")
    model = os.environ.get(model_env, llm.get("model", "gpt-4o-mini"))

    profile_file = llm.get("research_profile_file", "")
    if profile_file and Path(profile_file).is_file():
        research_profile = Path(profile_file).read_text(encoding="utf-8").strip()
    else:
        research_profile = llm.get("research_profile", "")

    return LLMConfig(
        base_url=base_url,
        base_url_env=base_url_env,
        model=model,
        model_env=model_env,
        api_key_env=llm.get("api_key_env", "LLM_API_KEY"),
        max_concurrent=llm.get("max_concurrent", 5),
        max_retries=llm.get("max_retries", 3),
        max_backoff=llm.get("max_backoff", 32.0),
        max_tokens=llm.get("max_tokens", 2048),
        timeout_connect=llm.get("timeout_connect", 10.0),
        timeout_read=llm.get("timeout_read", 60.0),
        timeout_write=llm.get("timeout_write", 10.0),
        timeout_pool=llm.get("timeout_pool", 10.0),
        system_prompt=llm.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        user_prompt_template=llm.get("user_prompt_template", DEFAULT_USER_PROMPT_TEMPLATE),
        research_profile=research_profile,
        use_html=llm.get("use_html", True),
        use_pdf=llm.get("use_pdf", False),
        max_content_chars=llm.get("max_content_chars", 50000),
        borderline_min=llm.get("borderline_min", 4),
        borderline_max=llm.get("borderline_max", 7),
    )


# ---------------------------------------------------------------------------
# Subcommand: fetch
# ---------------------------------------------------------------------------

async def cmd_fetch(cfg: dict, days: int = 1) -> list[Path]:
    """Fetch papers from arXiv and save raw data.

    Returns
    -------
    list[Path]
        Paths to all raw day files that were written.
    """
    categories = cfg.get("categories", ["hep-ph"])
    fetch_opts = cfg.get("fetch", {})
    page_size = fetch_opts.get("page_size", 100)

    logger.info("Fetching papers from categories: %s (last %d days)", categories, days)
    papers = fetch_papers(categories, days=days, page_size=page_size)
    logger.info("Fetched %d papers", len(papers))

    if not papers:
        logger.info("No papers found, exiting")
        sys.exit(0)

    logger.info("Resolving HTML availability for %d papers", len(papers))
    await _resolve_html_urls(papers)

    raw_dir = cfg.get("output", {}).get("raw_dir", cfg.get("output", {}).get("dir", "data"))
    paths = save_papers(papers, output_dir=raw_dir)
    logger.info("Raw data saved (%d papers)", len(papers))
    return paths


# ---------------------------------------------------------------------------
# Subcommand: filter
# ---------------------------------------------------------------------------

async def cmd_filter(cfg: dict, date_str: str | None = None) -> list[Path]:
    """Run LLM filter on previously saved raw data and write filtered results.

    Papers that were already successfully evaluated (``evaluated=true``) in a
    previous run are **skipped** unless their ``updated`` field has changed.
    Papers marked ``evaluated=false`` are always re-evaluated.

    Output is organized by filter run date, not article date.
    """
    from datetime import datetime

    llm_config = build_llm_config(cfg)

    if not llm_config.research_profile:
        logger.error("No research_profile configured — cannot filter")
        sys.exit(1)

    api_key = os.environ.get(llm_config.api_key_env, "")
    if not api_key:
        logger.error("%s not set — cannot filter", llm_config.api_key_env)
        sys.exit(1)

    raw_dir = cfg.get("output", {}).get("raw_dir", cfg.get("output", {}).get("dir", "data"))
    filtered_dir = cfg.get("output", {}).get("filtered_dir", "output")
    trash_dir = cfg.get("output", {}).get("trash_dir", "trash")
    run_date = date_str or datetime.now().strftime("%Y-%m-%d")

    raw_papers = load_raw(date_str or run_date, output_dir=raw_dir)
    if not raw_papers:
        logger.info("No raw papers found for %s", date_str or run_date)
        sys.exit(0)

    # --- Dedup: check previous filtered output ---
    previously_filtered = load_all_filtered(output_dir=filtered_dir)
    prev_index: dict[str, dict] = {}
    for p in previously_filtered:
        rel = p.get("relevance", {})
        prev_index[p["arxiv_id"]] = {
            "evaluated": rel.get("evaluated", True),
            "updated": p.get("updated", ""),
            "result": p,
        }

    to_evaluate: list[dict] = []
    carried_over: list[FilterResult] = []

    for rp in raw_papers:
        aid = rp["arxiv_id"]
        prev = prev_index.get(aid)
        if prev and prev["evaluated"] and prev["updated"] == rp.get("updated", ""):
            # Already successfully evaluated and unchanged — carry over
            rel = prev["result"].get("relevance", {})
            carried_over.append(FilterResult(
                paper=_dicts_to_papers([rp])[0],
                relevant=rel.get("relevant", False),
                score=rel.get("score", 0),
                reason=rel.get("reason", ""),
                pass_number=rel.get("pass_number", 1),
                evaluated=True,
            ))
        else:
            to_evaluate.append(rp)

    new_discarded: list[FilterResult] = []

    if to_evaluate:
        papers = _dicts_to_papers(to_evaluate)
        logger.info(
            "Filtering %d papers via LLM (%s), %d carried over from previous runs",
            len(papers), llm_config.model, len(carried_over),
        )
        outcome = await filter_papers(papers, llm_config, api_key)
        new_relevant = [r for r in outcome.relevant if r.relevant]
        new_discarded = outcome.discarded
    else:
        logger.info("All papers already evaluated, nothing to do")
        new_relevant = []

    combined = sorted(
        new_relevant + carried_over,
        key=lambda r: r.score,
        reverse=True,
    )
    logger.info(
        "Filter done: %d relevant (%d new + %d carried over), %d discarded",
        len(combined), len(new_relevant), len(carried_over), len(new_discarded),
    )

    paths = save_filtered(combined, run_date=run_date, output_dir=filtered_dir)
    if new_discarded:
        save_trash(new_discarded, run_date=run_date, output_dir=trash_dir)
    return paths


# ---------------------------------------------------------------------------
# Subcommand: refilter
# ---------------------------------------------------------------------------

def _dicts_to_papers(raw_papers: list[dict]) -> list[Paper]:
    """Convert a list of raw paper dicts into :class:`Paper` objects."""
    return [
        Paper(
            arxiv_id=p["arxiv_id"],
            title=p["title"],
            authors=p["authors"],
            abstract=p["abstract"],
            categories=p["categories"],
            published=p["published"],
            updated=p["updated"],
            pdf_url=p["pdf_url"],
            html_url=p.get("html_url", "N/A"),
            entry_url=p["entry_url"],
        )
        for p in raw_papers
    ]


async def cmd_refilter(cfg: dict, date_str: str | None = None) -> list[Path]:
    """Re-evaluate papers that failed LLM evaluation in a previous filter run.

    Reads the filtered output for *date_str*, picks papers with
    ``evaluated=false``, re-runs them through the LLM, and overwrites the file.
    """
    from datetime import datetime

    llm_config = build_llm_config(cfg)

    if not llm_config.research_profile:
        logger.error("No research_profile configured — cannot refilter")
        sys.exit(1)

    api_key = os.environ.get(llm_config.api_key_env, "")
    if not api_key:
        logger.error("%s not set — cannot refilter", llm_config.api_key_env)
        sys.exit(1)

    filtered_dir = cfg.get("output", {}).get("filtered_dir", "output")
    trash_dir = cfg.get("output", {}).get("trash_dir", "trash")
    run_date = date_str or datetime.now().strftime("%Y-%m-%d")

    filtered = load_filtered(run_date, output_dir=filtered_dir)
    if not filtered:
        logger.info("No filtered papers found for %s", run_date)
        sys.exit(0)

    unevaluated = [p for p in filtered if not p.get("relevance", {}).get("evaluated", True)]
    already_done = [p for p in filtered if p.get("relevance", {}).get("evaluated", True)]

    if not unevaluated:
        logger.info("All %d papers already evaluated for %s — nothing to refilter", len(filtered), run_date)
        sys.exit(0)

    logger.info(
        "Refiltering %d unevaluated papers (%d already done)",
        len(unevaluated), len(already_done),
    )

    papers = _dicts_to_papers(unevaluated)
    outcome = await filter_papers(papers, llm_config, api_key)

    done_results = _filtered_dicts_to_results(already_done)
    combined = sorted(
        [r for r in outcome.relevant if r.relevant] + done_results,
        key=lambda r: r.score,
        reverse=True,
    )
    logger.info("Refilter done: %d total relevant papers", len(combined))

    paths = save_filtered(combined, run_date=run_date, output_dir=filtered_dir)
    if outcome.discarded:
        save_trash(outcome.discarded, run_date=run_date, output_dir=trash_dir)
    return paths


def _filtered_dicts_to_results(paper_dicts: list[dict]) -> list[FilterResult]:
    """Convert filtered output dicts back into :class:`FilterResult` objects."""
    results: list[FilterResult] = []
    for p_dict in paper_dicts:
        rel = p_dict.get("relevance", {})
        results.append(FilterResult(
            paper=_dicts_to_papers([p_dict])[0],
            relevant=rel.get("relevant", False),
            score=rel.get("score", 0),
            reason=rel.get("reason", ""),
            pass_number=rel.get("pass_number", 1),
            evaluated=rel.get("evaluated", True),
        ))
    return results


def cmd_cleanup(cfg: dict, keep_days: int = 7) -> list[Path]:
    """Remove trash files older than *keep_days* days.

    Returns
    -------
    list[Path]
        Paths of removed files.
    """
    from datetime import datetime, timedelta

    trash_dir = Path(cfg.get("output", {}).get("trash_dir", "trash"))
    if not trash_dir.is_dir():
        logger.info("Trash directory %s does not exist — nothing to clean", trash_dir)
        return []

    cutoff = (datetime.now() - timedelta(days=keep_days)).strftime("%Y-%m-%d")
    removed: list[Path] = []
    for f in sorted(trash_dir.glob("*.json")):
        file_date = f.stem
        if file_date < cutoff:
            f.unlink()
            removed.append(f)
            logger.info("Removed old trash file: %s", f.name)

    if removed:
        logger.info("Cleanup: removed %d trash files older than %s", len(removed), cutoff)
    else:
        logger.info("Cleanup: no trash files older than %s", cutoff)
    return removed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Daily arXiv paper fetcher and filter")
    parser.add_argument("--config", "-c", default=str(DEFAULT_CONFIG_PATH), help="Config file path")

    sub = parser.add_subparsers(dest="command", required=True)

    # --- fetch ---
    p_fetch = sub.add_parser("fetch", help="Fetch papers from arXiv and save raw data")
    p_fetch.add_argument("--days", "-d", type=int, default=1, help="How many days back to fetch")

    # --- filter ---
    p_filter = sub.add_parser("filter", help="Run LLM filter on saved raw data")
    p_filter.add_argument("--date", type=str, default=None, help="Date to filter (YYYY-MM-DD, default: today)")

    # --- refilter ---
    p_refilter = sub.add_parser("refilter", help="Re-evaluate papers that failed LLM evaluation")
    p_refilter.add_argument("--date", type=str, default=None, help="Date to refilter (YYYY-MM-DD, default: today)")

    # --- cleanup ---
    p_cleanup = sub.add_parser("cleanup", help="Remove old trash files")
    p_cleanup.add_argument("--keep", type=int, default=7, help="Keep trash files from the last N days (default: 7)")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "fetch":
        asyncio.run(cmd_fetch(cfg, days=args.days))
    elif args.command == "filter":
        asyncio.run(cmd_filter(cfg, date_str=args.date))
    elif args.command == "refilter":
        asyncio.run(cmd_refilter(cfg, date_str=args.date))
    elif args.command == "cleanup":
        cmd_cleanup(cfg, keep_days=args.keep)


if __name__ == "__main__":
    main()
