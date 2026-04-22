"""
LLM-based paper relevance filter with two-pass agent loop.

Filtering pipeline:

**Pass 1 — Quick screen** — every paper is evaluated using **title + abstract**
only.  Fast, cheap, covers the whole batch.

**Pass 2 — Deep review** — papers whose pass-1 score falls within the
*borderline* range (``borderline_min`` ≤ score ≤ ``borderline_max``) are
re-evaluated with their **full text** (HTML or PDF).  Papers scoring above the
range are accepted as-is; those below are discarded.

**Retry on parse failure** — if the LLM returns malformed JSON, the request is
retried (up to ``max_retries`` times) with a correction hint appended to the
prompt.

The core entry point is :func:`filter_papers`, which returns a list of
:class:`FilterResult` instances sorted by relevance score (highest first).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum

import httpx

from .fetcher import Paper

logger = logging.getLogger(__name__)


class _CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# ---------------------------------------------------------------------------
# Adaptive concurrency semaphore with circuit breaker
# ---------------------------------------------------------------------------

class AdaptiveSemaphore:
    """Concurrency limiter with TCP-style ramp-up and circuit breaker.

    Ramp-up strategy (inspired by TCP congestion control):

    * **Probe phase**: on success, double the limit (``×2``).
    * **Stable phase** (after reaching max): on success, grow linearly (``+1``).
    * On ``report_failure()`` the limit decreases by 1 (down to *min_limit*).

    **Failure ceiling**: once the same concurrency level fails *N* times,
    effective max is capped at ``level - 1``.

    **Circuit breaker**: after *cb_threshold* consecutive failures, the circuit
    opens and all ``acquire()`` calls block for a cooldown period.  After the
    cooldown, one probe request is allowed (HALF_OPEN).  If it succeeds, the
    circuit closes; if it fails, the cooldown doubles.
    """

    def __init__(
        self,
        initial: int = 1,
        min_limit: int = 1,
        max_limit: int = 5,
        fail_ceiling_hits: int = 3,
        cb_threshold: int = 3,
        cb_base_cooldown: float = 5.0,
        cb_max_cooldown: float = 300.0,
    ) -> None:
        self._min = min_limit
        self._max = max_limit
        self._limit = max(min(initial, max_limit), min_limit)
        self._count = self._limit
        self._cond: asyncio.Condition | None = None
        self._fail_ceiling_hits = fail_ceiling_hits
        self._fail_levels: dict[int, int] = {}
        self._effective_max = max_limit
        self._reached_max = False

        self._cb_threshold = cb_threshold
        self._cb_base_cooldown = cb_base_cooldown
        self._cb_max_cooldown = cb_max_cooldown
        self._cb_state = _CircuitState.CLOSED
        self._cb_consecutive_failures = 0
        self._cb_open_count = 0
        self._cb_open_until: float = 0.0

    def _get_cond(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    @property
    def circuit_state(self) -> _CircuitState:
        return self._cb_state

    async def acquire(self) -> None:
        cond = self._get_cond()
        async with cond:
            while True:
                if self._cb_state == _CircuitState.OPEN:
                    now = time.monotonic()
                    if now >= self._cb_open_until:
                        self._cb_state = _CircuitState.HALF_OPEN
                        logger.info("AdaptiveSemaphore: circuit HALF_OPEN — probing")
                        break
                    remaining = self._cb_open_until - now
                    logger.info(
                        "AdaptiveSemaphore: circuit OPEN — waiting %.0fs",
                        remaining,
                    )
                    try:
                        await asyncio.wait_for(cond.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        pass
                    continue
                if self._count > 0:
                    break
                await cond.wait()
            self._count -= 1

    async def wait_if_circuit_open(self) -> None:
        """Block until the circuit breaker is no longer OPEN.

        Unlike ``acquire()``, this does not consume a semaphore slot.  Use it
        in retry loops where the caller already holds a slot but should respect
        the circuit breaker cooldown.
        """
        cond = self._get_cond()
        async with cond:
            while self._cb_state == _CircuitState.OPEN:
                now = time.monotonic()
                if now >= self._cb_open_until:
                    self._cb_state = _CircuitState.HALF_OPEN
                    logger.info("AdaptiveSemaphore: circuit HALF_OPEN — probing")
                    return
                remaining = self._cb_open_until - now
                logger.info(
                    "AdaptiveSemaphore: retry blocked by circuit — waiting %.0fs",
                    remaining,
                )
                try:
                    await asyncio.wait_for(cond.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    pass

    async def release(self) -> None:
        cond = self._get_cond()
        async with cond:
            if self._count < self._limit:
                self._count += 1
                cond.notify()

    def _open_circuit(self) -> None:
        cooldown = min(
            self._cb_base_cooldown * (2 ** self._cb_open_count),
            self._cb_max_cooldown,
        )
        self._cb_open_until = time.monotonic() + cooldown
        self._cb_open_count += 1
        self._cb_state = _CircuitState.OPEN
        self._cb_consecutive_failures = 0
        logger.warning(
            "AdaptiveSemaphore: circuit OPEN — cooldown %.0fs (trip #%d)",
            cooldown, self._cb_open_count,
        )

    async def report_success(self) -> None:
        cond = self._get_cond()
        async with cond:
            self._cb_consecutive_failures = 0
            if self._cb_state == _CircuitState.HALF_OPEN:
                self._cb_state = _CircuitState.CLOSED
                self._cb_open_count = 0
                logger.info("AdaptiveSemaphore: circuit CLOSED — API recovered")
            if self._limit >= self._effective_max:
                return
            if self._reached_max:
                self._limit += 1
            else:
                self._limit = min(self._limit * 2, self._effective_max)
            self._limit = min(self._limit, self._effective_max)
            if self._limit >= self._max:
                self._reached_max = True
            self._count += 1
            logger.debug("AdaptiveSemaphore: success → limit=%d", self._limit)
            cond.notify()

    async def report_failure(self) -> None:
        cond = self._get_cond()
        async with cond:
            if self._cb_state == _CircuitState.HALF_OPEN:
                self._open_circuit()
                return
            self._cb_consecutive_failures += 1
            if self._cb_consecutive_failures >= self._cb_threshold:
                self._open_circuit()
                return

            old = self._limit
            level = self._limit
            hits = self._fail_levels.get(level, 0) + 1
            self._fail_levels[level] = hits
            new_max = max(level - 1, self._min)
            if hits >= self._fail_ceiling_hits and new_max < self._effective_max:
                self._effective_max = new_max
                logger.info(
                    "AdaptiveSemaphore: failure ceiling confirmed at %d "
                    "(%d hits) → effective_max=%d",
                    level, hits, self._effective_max,
                )
            self._limit = max(self._min, self._limit - 1)
            excess = self._count - self._limit
            if excess > 0:
                self._count = self._limit
            if old != self._limit:
                logger.info(
                    "AdaptiveSemaphore: failure → limit %d → %d", old, self._limit,
                )

    @property
    def limit(self) -> int:
        return self._limit

# ---------------------------------------------------------------------------
# Default prompt templates
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """\
You are a research paper relevance evaluator. Given a paper's metadata and \
(optionally) its full text, you must assess its relevance to the user's \
research interests and return a structured JSON response.
"""

DEFAULT_USER_PROMPT_TEMPLATE = """\
## Research Interest Profile
{research_profile}

## Paper
**Title:** {title}
**Abstract:** {abstract}
{full_text_section}
## Task
Evaluate this paper's relevance to the above research interests.
Respond with ONLY a JSON object (no markdown fences, no extra text) with these fields:
- "relevant": boolean - whether this paper is relevant
- "score": integer 1-10 - relevance score (10 = extremely relevant)
- "reason": string - one-sentence explanation of why it is or isn't relevant
"""

DEFAULT_FULL_TEXT_SECTION = """
**Full Text ({source}):**
```
{full_text}
```
"""

RETRY_HINT = """

## Important
Your previous response was NOT valid JSON.  Please respond with ONLY a raw \
JSON object — no commentary, no markdown code fences.
"""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """Outcome of filtering a single paper through the LLM."""

    paper: Paper
    relevant: bool
    score: int
    """Relevance score from 1 (barely related) to 10 (extremely relevant)."""

    reason: str
    """One-sentence explanation from the LLM."""

    pass_number: int = 1
    """Which evaluation pass produced this result (1 or 2)."""

    evaluated: bool = True
    """``False`` if the LLM could not evaluate this paper (HTTP failure or
    JSON parse error).  Unevaluated papers are included in the output so they
    are not silently lost."""


@dataclass
class LLMConfig:
    """Configuration for the LLM filtering backend.

    All fields have sensible defaults targeting OpenAI's ``gpt-4o-mini``.  To
    use a different provider (e.g. Azure OpenAI, Ollama, Together AI), simply
    change ``base_url`` and ``model`` in ``config.yaml``.
    """

    base_url: str = "https://api.openai.com/v1"
    """Base URL of the OpenAI-compatible API."""

    base_url_env: str = "LLM_BASE_URL"
    """Name of the environment variable that overrides *base_url*."""

    model: str = "gpt-4o-mini"
    """Model identifier passed in the ``model`` field of the chat completion request."""

    model_env: str = "LLM_MODEL"
    """Name of the environment variable that overrides *model*."""

    api_key_env: str = "LLM_API_KEY"
    """Name of the environment variable that holds the API key."""

    max_concurrent: int = 5
    """Maximum number of in-flight HTTP requests."""

    initial_concurrent: int = 1
    """Starting concurrency level.  The adaptive semaphore ramps up from
    this value toward *max_concurrent* on success."""

    system_prompt: str = field(default_factory=lambda: DEFAULT_SYSTEM_PROMPT)
    user_prompt_template: str = field(default_factory=lambda: DEFAULT_USER_PROMPT_TEMPLATE)
    research_profile: str = ""
    """Free-text description of the user's research interests.  This is the
    primary knob for tuning filter quality — treat it as a prompt-engineering
    task and iterate."""

    use_html: bool = True
    """Fetch and include the arXiv HTML full text when available."""

    use_pdf: bool = False
    """Download and extract PDF full text.  Significantly increases latency."""

    max_content_chars: int = 50000
    """Truncate fetched full-text content to this many characters."""

    borderline_min: int = 4
    """Minimum score (inclusive) to be considered borderline."""

    borderline_max: int = 7
    """Maximum score (inclusive) to be considered borderline."""

    max_retries: int = 10
    """Unified retry count for all retry scenarios: HTTP 429/5xx, timeouts,
    empty content, and JSON parse failures."""

    max_backoff: float = 32.0
    """Maximum backoff delay in seconds for exponential retries.
    Actual delay is ``min(2 ** attempt, max_backoff)``."""

    max_tokens: int = 2048
    """Maximum tokens in the LLM response.  Reasoning models (e.g. GLM-5.1,
    DeepSeek-R1) consume tokens for ``reasoning_content`` from the same budget,
    so this should be large enough to cover both reasoning and the actual JSON
    output (e.g. 2048).  For non-reasoning models, 512 is usually sufficient."""

    timeout_connect: float = 10.0
    """HTTP connect timeout in seconds."""

    timeout_read: float = 60.0
    """HTTP read timeout in seconds — covers the full LLM inference time."""

    timeout_write: float = 10.0
    """HTTP write timeout in seconds."""

    timeout_pool: float = 10.0
    """HTTP connection-pool wait timeout in seconds."""

    cb_threshold: int = 3
    """Consecutive failures before the circuit breaker opens."""

    cb_base_cooldown: float = 5.0
    """Initial cooldown in seconds when the circuit opens.  Doubles on each
    subsequent trip, up to *cb_max_cooldown*."""

    cb_max_cooldown: float = 60.0
    """Maximum cooldown in seconds for the circuit breaker."""


# ---------------------------------------------------------------------------
# Content fetching helpers
# ---------------------------------------------------------------------------

def _strip_html(html: str) -> str:
    """Naively extract visible text from HTML — good enough for arXiv pages."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def fetch_html_text(
    client: httpx.AsyncClient,
    url: str,
    max_chars: int = 50000,
) -> str | None:
    """Fetch an arXiv HTML page and return its visible text, or ``None`` on failure."""
    try:
        resp = await client.get(url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        text = _strip_html(resp.text)
        if len(text) > max_chars:
            logger.warning(
                "HTML content for %s truncated: %d → %d chars",
                url, len(text), max_chars,
            )
        return text[:max_chars]
    except Exception as e:
        logger.debug("Failed to fetch HTML from %s: %s", url, e)
        return None


async def fetch_pdf_text(
    client: httpx.AsyncClient,
    url: str,
    max_chars: int = 50000,
) -> str | None:
    """Download a PDF and extract its text, or ``None`` on failure."""
    try:
        from pypdf import PdfReader

        resp = await client.get(url, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()

        reader = PdfReader(io.BytesIO(resp.content))
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        text = "\n".join(pages)
        if len(text) > max_chars:
            logger.warning(
                "PDF content for %s truncated: %d → %d chars",
                url, len(text), max_chars,
            )
        return text[:max_chars]
    except Exception as e:
        logger.debug("Failed to extract PDF from %s: %s", url, e)
        return None


async def prefetch_content(
    papers: list[Paper],
    config: LLMConfig,
) -> dict[str, dict[str, str | None]]:
    """Fetch HTML / PDF full text for all papers concurrently.

    Returns a dict mapping ``arxiv_id`` → ``{"html": ..., "pdf": ...}``.
    """
    semaphore = asyncio.Semaphore(config.max_concurrent)
    results: dict[str, dict[str, str | None]] = {}

    async def _fetch(paper: Paper) -> None:
        entry: dict[str, str | None] = {"html": None, "pdf": None}

        async with semaphore:
            async with httpx.AsyncClient() as client:
                if config.use_html and paper.html_url and paper.html_url != "N/A":
                    entry["html"] = await fetch_html_text(
                        client, paper.html_url, config.max_content_chars,
                    )
                if config.use_pdf:
                    entry["pdf"] = await fetch_pdf_text(
                        client, paper.pdf_url, config.max_content_chars,
                    )

        results[paper.arxiv_id] = entry

    await asyncio.gather(*[_fetch(p) for p in papers])
    return results


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_user_prompt(
    template: str,
    paper: Paper,
    profile: str,
    content: dict[str, str | None] | None = None,
    max_content_chars: int = 50000,
    retry_hint: str = "",
) -> str:
    """Fill the user-prompt template with paper metadata and optional full text."""
    full_text_section = ""

    if content:
        full_text = content.get("pdf") or content.get("html")
        if full_text:
            source = "PDF" if content.get("pdf") else "HTML"
            if len(full_text) > max_content_chars:
                logger.warning(
                    "Prompt content for %s truncated: %d → %d chars",
                    paper.arxiv_id, len(full_text), max_content_chars,
                )
            full_text_section = DEFAULT_FULL_TEXT_SECTION.format(
                source=source,
                full_text=full_text[:max_content_chars],
            )

    return template.format(
        research_profile=profile,
        title=paper.title,
        abstract=paper.abstract,
        full_text_section=full_text_section,
    ) + retry_hint


# ---------------------------------------------------------------------------
# LLM call (with retry)
# ---------------------------------------------------------------------------

def _parse_llm_response(raw: str, paper: Paper) -> FilterResult | None:
    """Try to parse the LLM response into a FilterResult.

    Returns ``None`` if the response is not valid JSON.
    """
    cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    return FilterResult(
        paper=paper,
        relevant=bool(parsed.get("relevant", False)),
        score=int(parsed.get("score", 0)),
        reason=str(parsed.get("reason", "")),
    )


def _extract_response_text(choice: dict) -> str | None:
    """Extract the text content from a chat completion choice.

    For reasoning models (GLM-5.1, DeepSeek-R1) the actual answer may be in
    ``reasoning_content`` when ``content`` is empty because ``max_tokens`` was
    exhausted by the reasoning trace.  This helper falls back to extracting a
    JSON-like snippet from ``reasoning_content`` when ``content`` is empty.
    """
    msg = choice.get("message", {})
    content = msg.get("content")
    if content and content.strip():
        return content

    reasoning = msg.get("reasoning_content", "")
    if reasoning:
        json_match = re.search(r'\{[^{}]*"relevant"[^{}]*\}', reasoning, re.DOTALL)
        if json_match:
            logger.debug("Extracted JSON from reasoning_content (content was empty)")
            return json_match.group(0)

    return content


async def _backoff(attempt: int, max_backoff: float) -> None:
    await asyncio.sleep(min(2.0 ** attempt, max_backoff))


def _make_request_kwargs(config: LLMConfig, api_key: str, messages: list[dict]) -> dict:
    return {
        "url": f"{config.base_url}/chat/completions",
        "headers": {"Authorization": f"Bearer {api_key}"},
        "json": {
            "model": config.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": config.max_tokens,
        },
    }


async def _send_llm(
    client: httpx.AsyncClient,
    config: LLMConfig,
    api_key: str,
    messages: list[dict],
    paper: Paper,
    semaphore: AdaptiveSemaphore | None,
    attempt: int,
    pass_number: int,
) -> tuple[httpx.Response | None, FilterResult | None]:
    """Attempt a single LLM request.  Returns ``(response, None)`` on success,
    ``(None, FilterResult)`` on terminal failure (retries exhausted), or
    ``(None, None)`` to signal the caller should retry.
    """
    try:
        resp = await client.post(**_make_request_kwargs(config, api_key, messages))
    except httpx.TimeoutException:
        if semaphore:
            await semaphore.report_failure()
        if attempt < config.max_retries:
            logger.warning(
                "Timeout for %s, backing off (attempt %d/%d)",
                paper.arxiv_id, attempt + 1, 1 + config.max_retries,
            )
            await _backoff(attempt, config.max_backoff)
            return None, None
        logger.warning(
            "⚠ %s: timeout after %d attempts — included as unevaluated",
            paper.arxiv_id, 1 + config.max_retries,
        )
        return None, FilterResult(
            paper=paper, relevant=True, score=0,
            reason=f"UNEVALUATED: timeout after {1 + config.max_retries} attempts",
            pass_number=pass_number, evaluated=False,
        )

    if resp.status_code in (429, 500, 502, 503, 504):
        if semaphore:
            await semaphore.report_failure()
        if attempt < config.max_retries:
            logger.warning(
                "HTTP %d for %s, backing off (attempt %d/%d)",
                resp.status_code, paper.arxiv_id, attempt + 1, 1 + config.max_retries,
            )
            await _backoff(attempt, config.max_backoff)
            return None, None
        logger.warning(
            "⚠ %s: HTTP %d after %d attempts — included as unevaluated",
            paper.arxiv_id, resp.status_code, 1 + config.max_retries,
        )
        return None, FilterResult(
            paper=paper, relevant=True, score=0,
            reason=f"UNEVALUATED: HTTP {resp.status_code} after retries",
            pass_number=pass_number, evaluated=False,
        )

    return resp, None


async def _call_llm(
    client: httpx.AsyncClient,
    config: LLMConfig,
    paper: Paper,
    api_key: str,
    content: dict[str, str | None] | None = None,
    pass_number: int = 1,
    semaphore: AdaptiveSemaphore | None = None,
) -> FilterResult:
    """Send a single paper to the LLM with unified retry on all failures.

    All retry scenarios (HTTP 429/5xx, timeout, empty content, invalid JSON)
    share the same ``max_retries`` count and ``max_backoff`` ceiling.
    """
    user_content = _build_user_prompt(
        config.user_prompt_template, paper, config.research_profile, content,
        max_content_chars=config.max_content_chars,
    )
    messages: list[dict] = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": user_content},
    ]

    attempt = 0
    while attempt <= config.max_retries:
        if attempt > 0 and semaphore:
            await semaphore.wait_if_circuit_open()
        resp, terminal = await _send_llm(
            client, config, api_key, messages, paper, semaphore, attempt, pass_number,
        )
        if terminal is not None:
            return terminal
        if resp is None:
            attempt += 1
            continue

        resp.raise_for_status()

        if semaphore:
            await semaphore.report_success()

        raw = _extract_response_text(resp.json()["choices"][0])

        if raw is None or not raw.strip():
            if attempt < config.max_retries:
                logger.warning(
                    "Empty content for %s, backing off (attempt %d/%d)",
                    paper.arxiv_id, attempt + 1, 1 + config.max_retries,
                )
                await _backoff(attempt, config.max_backoff)
                attempt += 1
                continue
            logger.warning(
                "⚠ %s: empty content after %d attempts — included as unevaluated",
                paper.arxiv_id, 1 + config.max_retries,
            )
            return FilterResult(
                paper=paper, relevant=True, score=0,
                reason="UNEVALUATED: empty content after retries",
                pass_number=pass_number, evaluated=False,
            )

        raw = raw.strip()
        result = _parse_llm_response(raw, paper)

        if result is not None:
            result.pass_number = pass_number
            return result

        if attempt < config.max_retries:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": RETRY_HINT})
            logger.warning(
                "Invalid JSON for %s, retrying (attempt %d/%d): %s",
                paper.arxiv_id, attempt + 1, 1 + config.max_retries, raw[:200],
            )
            attempt += 1
            continue

        logger.warning(
            "⚠ %s: JSON parse error after %d attempts — included as unevaluated",
            paper.arxiv_id, 1 + config.max_retries,
        )
        return FilterResult(
            paper=paper, relevant=True, score=0,
            reason="UNEVALUATED: LLM parse error after retries",
            pass_number=pass_number, evaluated=False,
        )

    return FilterResult(
        paper=paper, relevant=True, score=0,
        reason="UNEVALUATED: failed after retries",
        pass_number=pass_number, evaluated=False,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def _evaluate_batch(
    papers: list[Paper],
    config: LLMConfig,
    api_key: str,
    content_map: dict[str, dict[str, str | None]],
    pass_number: int = 1,
    use_content: bool = False,
) -> list[FilterResult]:
    """Evaluate a batch of papers concurrently via the LLM."""
    semaphore = AdaptiveSemaphore(
        initial=config.initial_concurrent,
        min_limit=1,
        max_limit=config.max_concurrent,
        cb_threshold=config.cb_threshold,
        cb_base_cooldown=config.cb_base_cooldown,
        cb_max_cooldown=config.cb_max_cooldown,
    )

    async def _limited(client: httpx.AsyncClient, paper: Paper) -> FilterResult:
        await semaphore.acquire()
        try:
            content = content_map.get(paper.arxiv_id) if use_content else None
            return await _call_llm(
                client, config, paper, api_key,
                content=content, pass_number=pass_number,
                semaphore=semaphore,
            )
        except Exception as e:
            logger.warning(
                "⚠ %s: unexpected error (%s) — included as unevaluated: %s",
                paper.arxiv_id, type(e).__name__, e,
            )
            return FilterResult(
                paper=paper, relevant=True, score=0,
                reason=f"UNEVALUATED: {type(e).__name__}: {e}",
                pass_number=pass_number, evaluated=False,
            )
        finally:
            await semaphore.release()

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(
            config.timeout_connect,
            read=config.timeout_read,
            write=config.timeout_write,
            pool=config.timeout_pool,
        ),
    ) as client:
        tasks = [_limited(client, p) for p in papers]
        return list(await asyncio.gather(*tasks))


@dataclass
class FilterOutcome:
    """Result of the full two-pass filtering pipeline."""

    relevant: list[FilterResult]
    discarded: list[FilterResult]


def _is_borderline(result: FilterResult, config: LLMConfig) -> bool:
    """Check whether a result falls in the borderline score range."""
    return config.borderline_min <= result.score <= config.borderline_max


async def filter_papers(
    papers: list[Paper],
    config: LLMConfig,
    api_key: str,
) -> FilterOutcome:
    """Score and filter *papers* using a two-pass LLM agent loop.

    **Pass 1** — every paper is evaluated with title + abstract only.

    **Pass 2** — borderline papers (``borderline_min`` ≤ score ≤ ``borderline_max``)
    are re-evaluated with full text (HTML/PDF) if content fetching is enabled.

    Papers scoring above the borderline range are accepted as-is; those below
    are discarded.  Within the borderline, only papers that score above the
    range after pass 2 are kept.

    Each LLM call is retried (up to ``max_retries``) on JSON parse failure.

    Parameters
    ----------
    papers:
        The papers to evaluate.
    config:
        LLM endpoint, content-fetch, and agent-loop configuration.
    api_key:
        Bearer token for the API.

    Returns
    -------
    FilterOutcome
        ``relevant``: accepted + unevaluated results, sorted by score.
        ``discarded``: rejected results with reasons, sorted by score.
    """
    if not papers:
        return FilterOutcome(relevant=[], discarded=[])

    # --- Pass 1: abstract-only quick screen ---
    logger.info("Pass 1: evaluating %d papers (abstract only)", len(papers))
    empty_content: dict[str, dict[str, str | None]] = {}
    pass1_results = await _evaluate_batch(
        papers, config, api_key,
        content_map=empty_content, pass_number=1, use_content=False,
    )

    accepted: list[FilterResult] = []
    borderline: list[FilterResult] = []
    unevaluated: list[FilterResult] = []
    discarded: list[FilterResult] = []

    for r in pass1_results:
        if not r.evaluated:
            unevaluated.append(r)
        elif r.score > config.borderline_max:
            accepted.append(r)
        elif _is_borderline(r, config):
            borderline.append(r)
        else:
            discarded.append(r)

    logger.info(
        "Pass 1 done: %d accepted, %d borderline, %d discarded, %d unevaluated",
        len(accepted), len(borderline), len(discarded), len(unevaluated),
    )

    if not borderline:
        if unevaluated:
            logger.warning(
                "⚠ %d papers could NOT be evaluated (included anyway): %s",
                len(unevaluated),
                ", ".join(r.paper.arxiv_id for r in unevaluated),
            )
        return FilterOutcome(
            relevant=sorted(accepted + unevaluated, key=lambda r: r.score, reverse=True),
            discarded=sorted(discarded, key=lambda r: r.score, reverse=True),
        )

    # --- Pass 2: deep review with full text ---
    should_prefetch = config.use_html or config.use_pdf
    content_map: dict[str, dict[str, str | None]] = {}

    if should_prefetch:
        borderline_papers = [r.paper for r in borderline]
        logger.info(
            "Prefetching full text for %d borderline papers (html=%s, pdf=%s)",
            len(borderline_papers), config.use_html, config.use_pdf,
        )
        content_map = await prefetch_content(borderline_papers, config)

    logger.info("Pass 2: re-evaluating %d borderline papers with full text", len(borderline))
    borderline_papers_list = [r.paper for r in borderline]
    pass2_results = await _evaluate_batch(
        borderline_papers_list, config, api_key,
        content_map=content_map, pass_number=2, use_content=True,
    )

    for r in pass2_results:
        if not r.evaluated:
            unevaluated.append(r)
        elif r.relevant:
            accepted.append(r)
        else:
            discarded.append(r)

    if unevaluated:
        logger.warning(
            "⚠ %d papers could NOT be evaluated (included anyway): %s",
            len(unevaluated),
            ", ".join(r.paper.arxiv_id for r in unevaluated),
        )

    logger.info(
        "Filtering complete: %d relevant, %d discarded, %d unevaluated out of %d total",
        len(accepted), len(discarded), len(unevaluated), len(papers),
    )

    return FilterOutcome(
        relevant=sorted(accepted + unevaluated, key=lambda r: r.score, reverse=True),
        discarded=sorted(discarded, key=lambda r: r.score, reverse=True),
    )
