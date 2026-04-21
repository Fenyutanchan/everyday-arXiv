import json
import time

import httpx
import pytest

from src.fetcher import Paper
from src.filter import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT_TEMPLATE,
    AdaptiveSemaphore,
    FilterOutcome,
    FilterResult,
    LLMConfig,
    _CircuitState,
    _build_user_prompt,
    _call_llm,
    _extract_response_text,
    _is_borderline,
    _parse_llm_response,
    _strip_html,
    fetch_html_text,
    fetch_pdf_text,
    filter_papers,
)


def _make_paper(**overrides) -> Paper:
    defaults = dict(
        arxiv_id="2604.15309v1",
        title="Test Paper Title",
        authors=["Alice", "Bob"],
        abstract="This is a test abstract about LLM alignment.",
        categories=["hep-ph"],
        published="2026-04-16T10:00:00+00:00",
        updated="2026-04-16T10:00:00+00:00",
        pdf_url="https://arxiv.org/pdf/2604.15309v1",
        entry_url="https://arxiv.org/abs/2604.15309v1",
        html_url="https://arxiv.org/html/2604.15309v1",
    )
    defaults.update(overrides)
    return Paper(**defaults)


def _make_result(**overrides) -> FilterResult:
    paper_overrides = {k: v for k, v in overrides.items() if k in Paper.__dataclass_fields__}
    result_overrides = {k: v for k, v in overrides.items()
                        if k in FilterResult.__dataclass_fields__ and k != "paper"}
    defaults = dict(relevant=True, score=7, reason="test", pass_number=1)
    defaults.update(result_overrides)
    return FilterResult(paper=_make_paper(**paper_overrides), **defaults)


class TestLLMConfigDefaults:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.use_html is True
        assert cfg.use_pdf is False
        assert cfg.max_content_chars == 50000
        assert cfg.borderline_min == 4
        assert cfg.borderline_max == 7
        assert cfg.max_retries == 3
        assert cfg.max_backoff == 32.0

    def test_custom(self):
        cfg = LLMConfig(use_pdf=True, borderline_min=3, borderline_max=6, max_retries=5)
        assert cfg.use_pdf is True
        assert cfg.borderline_min == 3
        assert cfg.max_retries == 5


class TestBuildUserPrompt:
    def test_template_filling(self):
        paper = _make_paper()
        prompt = _build_user_prompt(DEFAULT_USER_PROMPT_TEMPLATE, paper, "I like RLHF.")
        assert "I like RLHF." in prompt
        assert "Test Paper Title" in prompt

    def test_includes_html_content(self):
        paper = _make_paper()
        content = {"html": "Full HTML text here.", "pdf": None}
        prompt = _build_user_prompt(DEFAULT_USER_PROMPT_TEMPLATE, paper, "profile", content)
        assert "Full HTML text here." in prompt
        assert "HTML" in prompt

    def test_prefers_pdf_over_html(self):
        paper = _make_paper()
        content = {"html": "html text", "pdf": "pdf text"}
        prompt = _build_user_prompt(DEFAULT_USER_PROMPT_TEMPLATE, paper, "profile", content)
        assert "pdf text" in prompt
        assert "PDF" in prompt

    def test_no_content_section_when_empty(self):
        paper = _make_paper()
        prompt = _build_user_prompt(DEFAULT_USER_PROMPT_TEMPLATE, paper, "profile", None)
        assert "Full Text" not in prompt

    def test_retry_hint_appended(self):
        paper = _make_paper()
        prompt = _build_user_prompt(
            DEFAULT_USER_PROMPT_TEMPLATE, paper, "profile", None,
            retry_hint="\n\nFix your JSON!",
        )
        assert "Fix your JSON!" in prompt


class TestStripHtml:
    def test_removes_tags(self):
        assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_removes_scripts_and_styles(self):
        html = "<script>var x=1;</script><style>.a{}</style><p>visible</p>"
        assert _strip_html(html) == "visible"

    def test_collapses_whitespace(self):
        assert _strip_html("  a  \n  b  ") == "a b"


class TestFetchHtmlText:
    async def test_success(self, httpx_mock):
        httpx_mock.add_response(
            url="https://arxiv.org/html/2604.15309v1",
            html="<html><body><p>Paper content here.</p></body></html>",
        )
        async with httpx.AsyncClient() as client:
            text = await fetch_html_text(client, "https://arxiv.org/html/2604.15309v1")
        assert "Paper content here." in text

    async def test_failure_returns_none(self, httpx_mock):
        httpx_mock.add_response(url="https://arxiv.org/html/2604.15309v1", status_code=404)
        async with httpx.AsyncClient() as client:
            text = await fetch_html_text(client, "https://arxiv.org/html/2604.15309v1")
        assert text is None

    async def test_truncation(self, httpx_mock):
        long_text = "x" * 100000
        httpx_mock.add_response(url="https://arxiv.org/html/2604.15309v1", text=long_text)
        async with httpx.AsyncClient() as client:
            text = await fetch_html_text(client, "https://arxiv.org/html/2604.15309v1", max_chars=100)
        assert len(text) == 100


class TestFetchPdfText:
    async def test_failure_returns_none(self, httpx_mock):
        httpx_mock.add_response(url="https://arxiv.org/pdf/2604.15309v1", status_code=404)
        async with httpx.AsyncClient() as client:
            text = await fetch_pdf_text(client, "https://arxiv.org/pdf/2604.15309v1")
        assert text is None


class TestParseLlmResponse:
    def test_valid_json(self):
        paper = _make_paper()
        raw = '{"relevant": true, "score": 8, "reason": "ok"}'
        result = _parse_llm_response(raw, paper)
        assert result is not None
        assert result.score == 8
        assert result.relevant is True

    def test_markdown_fenced(self):
        paper = _make_paper()
        raw = '```json\n{"relevant": false, "score": 2, "reason": "no"}\n```'
        result = _parse_llm_response(raw, paper)
        assert result is not None
        assert result.score == 2

    def test_invalid_json_returns_none(self):
        paper = _make_paper()
        result = _parse_llm_response("not json", paper)
        assert result is None


class TestExtractResponseText:
    def test_content_present(self):
        choice = {"message": {"content": '{"relevant": true, "score": 8, "reason": "ok"}'}}
        assert _extract_response_text(choice) == '{"relevant": true, "score": 8, "reason": "ok"}'

    def test_content_empty_with_reasoning_fallback(self):
        choice = {"message": {
            "content": "",
            "reasoning_content": 'Some reasoning then {"relevant": true, "score": 7, "reason": "test"} more text',
        }}
        result = _extract_response_text(choice)
        assert result == '{"relevant": true, "score": 7, "reason": "test"}'

    def test_content_none_with_reasoning_fallback(self):
        choice = {"message": {
            "content": None,
            "reasoning_content": 'I think {"relevant": false, "score": 3, "reason": "nope"} is the answer',
        }}
        result = _extract_response_text(choice)
        assert result == '{"relevant": false, "score": 3, "reason": "nope"}'

    def test_both_empty_returns_none(self):
        choice = {"message": {"content": "", "reasoning_content": "no json here"}}
        assert _extract_response_text(choice) == ""

    def test_no_reasoning_field(self):
        choice = {"message": {"content": None}}
        assert _extract_response_text(choice) is None


class TestIsBorderline:
    def test_inside(self):
        cfg = LLMConfig(borderline_min=4, borderline_max=7)
        assert _is_borderline(_make_result(score=4), cfg)
        assert _is_borderline(_make_result(score=7), cfg)
        assert _is_borderline(_make_result(score=5), cfg)

    def test_outside(self):
        cfg = LLMConfig(borderline_min=4, borderline_max=7)
        assert not _is_borderline(_make_result(score=3), cfg)
        assert not _is_borderline(_make_result(score=8), cfg)


class TestCallLlm:
    async def test_success(self, httpx_mock):
        body = {"choices": [{"message": {"content": '{"relevant": true, "score": 8, "reason": "ok"}'}}]}
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=body)

        config = LLMConfig(research_profile="I research physics.")
        paper = _make_paper()
        async with httpx.AsyncClient() as client:
            result = await _call_llm(client, config, paper, "test-key")

        assert result.relevant is True
        assert result.score == 8
        assert result.pass_number == 1

    async def test_retry_on_bad_json(self, httpx_mock):
        bad_body = {"choices": [{"message": {"content": "not json"}}]}
        good_body = {"choices": [{"message": {"content": '{"relevant": true, "score": 6, "reason": "ok"}'}}]}
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=bad_body)
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=good_body)

        config = LLMConfig(research_profile="physics", max_retries=1)
        paper = _make_paper()
        async with httpx.AsyncClient() as client:
            result = await _call_llm(client, config, paper, "test-key")

        assert result.score == 6
        assert result.relevant is True

    async def test_all_retries_exhausted(self, httpx_mock):
        bad = {"choices": [{"message": {"content": "not json"}}]}
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=bad)
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=bad)

        config = LLMConfig(max_retries=1)
        paper = _make_paper()
        async with httpx.AsyncClient() as client:
            result = await _call_llm(client, config, paper, "test-key")

        assert result.relevant is True
        assert result.evaluated is False
        assert "retries" in result.reason.lower()

    async def test_http_429_retried(self, httpx_mock):
        good_body = {"choices": [{"message": {"content": '{"relevant": true, "score": 9, "reason": "ok"}'}}]}
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", status_code=429)
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=good_body)

        config = LLMConfig(max_retries=2)
        paper = _make_paper()
        async with httpx.AsyncClient() as client:
            result = await _call_llm(client, config, paper, "test-key")

        assert result.score == 9
        assert result.relevant is True

    async def test_http_429_all_retries_exhausted(self, httpx_mock):
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", status_code=429)
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", status_code=429)

        config = LLMConfig(max_retries=1)
        paper = _make_paper()
        async with httpx.AsyncClient() as client:
            result = await _call_llm(client, config, paper, "test-key")

        assert result.relevant is True
        assert result.evaluated is False
        assert "429" in result.reason

    async def test_empty_content_retried_and_succeeds(self, httpx_mock):
        empty_body = {"choices": [{"message": {"content": ""}}]}
        good_body = {"choices": [{"message": {"content": '{"relevant": true, "score": 7, "reason": "ok"}'}}]}
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=empty_body)
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=good_body)

        config = LLMConfig()
        paper = _make_paper()
        async with httpx.AsyncClient() as client:
            result = await _call_llm(client, config, paper, "test-key")

        assert result.score == 7
        assert result.relevant is True
        assert result.evaluated is True

    async def test_empty_content_all_retries_exhausted(self, httpx_mock):
        empty_body = {"choices": [{"message": {"content": ""}}]}
        for _ in range(1 + LLMConfig().max_retries):
            httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=empty_body)

        config = LLMConfig()
        paper = _make_paper()
        async with httpx.AsyncClient() as client:
            result = await _call_llm(client, config, paper, "test-key")

        assert result.evaluated is False
        assert "empty content" in result.reason.lower()

    async def test_none_content_retried(self, httpx_mock):
        none_body = {"choices": [{"message": {"content": None}}]}
        good_body = {"choices": [{"message": {"content": '{"relevant": true, "score": 8, "reason": "ok"}'}}]}
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=none_body)
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=good_body)

        config = LLMConfig()
        paper = _make_paper()
        async with httpx.AsyncClient() as client:
            result = await _call_llm(client, config, paper, "test-key")

        assert result.score == 8
        assert result.evaluated is True

    async def test_timeout_retried_and_succeeds(self, httpx_mock):
        good_body = {"choices": [{"message": {"content": '{"relevant": true, "score": 9, "reason": "ok"}'}}]}
        httpx_mock.add_exception(httpx.ReadTimeout("timed out"), url="https://api.openai.com/v1/chat/completions")
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=good_body)

        config = LLMConfig(max_retries=2)
        paper = _make_paper()
        async with httpx.AsyncClient() as client:
            result = await _call_llm(client, config, paper, "test-key")

        assert result.score == 9
        assert result.evaluated is True

    async def test_timeout_all_retries_exhausted(self, httpx_mock):
        httpx_mock.add_exception(httpx.ReadTimeout("timed out"), url="https://api.openai.com/v1/chat/completions")
        httpx_mock.add_exception(httpx.ReadTimeout("timed out"), url="https://api.openai.com/v1/chat/completions")

        config = LLMConfig(max_retries=1)
        paper = _make_paper()
        async with httpx.AsyncClient() as client:
            result = await _call_llm(client, config, paper, "test-key")

        assert result.evaluated is False
        assert "timeout" in result.reason.lower()

    async def test_retry_waits_for_circuit_breaker(self, httpx_mock):
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", status_code=429)
        good_body = {"choices": [{"message": {"content": '{"relevant": true, "score": 8, "reason": "ok"}'}}]}
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=good_body)

        config = LLMConfig(max_retries=2)
        paper = _make_paper()
        sem = AdaptiveSemaphore(initial=1, min_limit=1, max_limit=3, cb_threshold=1)
        for _ in range(1):
            await sem.report_failure()
        assert sem.circuit_state == _CircuitState.OPEN
        sem._cb_open_until = time.monotonic() + 0.05

        async with httpx.AsyncClient() as client:
            result = await _call_llm(client, config, paper, "test-key", semaphore=sem)

        assert result.score == 8
        assert result.evaluated is True


class TestAdaptiveSemaphore:
    async def test_acquire_release(self):
        sem = AdaptiveSemaphore(initial=2, min_limit=1, max_limit=5)
        await sem.acquire()
        assert sem.limit == 2
        await sem.release()

    async def test_success_doubles_in_probe_phase(self):
        sem = AdaptiveSemaphore(initial=2, min_limit=1, max_limit=10)
        await sem.report_success()
        assert sem.limit == 4

    async def test_success_capped_at_effective_max(self):
        sem = AdaptiveSemaphore(initial=5, min_limit=1, max_limit=5)
        await sem.report_success()
        assert sem.limit == 5

    async def test_failure_decrements_by_one(self):
        sem = AdaptiveSemaphore(initial=4, min_limit=1, max_limit=10, cb_threshold=100)
        await sem.report_failure()
        assert sem.limit == 3

    async def test_failure_floored_at_min(self):
        sem = AdaptiveSemaphore(initial=1, min_limit=1, max_limit=10, cb_threshold=100)
        await sem.report_failure()
        assert sem.limit == 1

    async def test_failure_ceiling_stops_ramp_up(self):
        sem = AdaptiveSemaphore(
            initial=1, min_limit=1, max_limit=10,
            fail_ceiling_hits=3, cb_threshold=100,
        )
        await sem.report_success()
        assert sem.limit == 2
        await sem.report_failure()
        assert sem.limit == 1
        await sem.report_success()
        assert sem.limit == 2
        await sem.report_failure()
        assert sem.limit == 1
        await sem.report_success()
        assert sem.limit == 2
        await sem.report_failure()
        assert sem.limit == 1
        assert sem._effective_max == 1

    async def test_ceiling_at_min_locks_effective_max(self):
        sem = AdaptiveSemaphore(
            initial=1, min_limit=1, max_limit=10,
            fail_ceiling_hits=3, cb_threshold=100,
        )
        for _ in range(3):
            await sem.report_failure()
        assert sem.limit == 1
        assert sem._effective_max == 1
        await sem.report_success()
        assert sem.limit == 1

    async def test_linear_growth_after_reaching_max(self):
        sem = AdaptiveSemaphore(initial=3, min_limit=1, max_limit=4, cb_threshold=100)
        await sem.report_success()
        assert sem.limit == 4
        await sem.report_failure()
        assert sem.limit == 3
        await sem.report_success()
        assert sem.limit == 4


class TestCircuitBreaker:
    async def test_opens_after_consecutive_failures(self):
        sem = AdaptiveSemaphore(initial=1, min_limit=1, max_limit=5, cb_threshold=3)
        for _ in range(3):
            await sem.report_failure()
        assert sem.circuit_state == _CircuitState.OPEN

    async def test_does_not_open_below_threshold(self):
        sem = AdaptiveSemaphore(initial=1, min_limit=1, max_limit=5, cb_threshold=5)
        for _ in range(4):
            await sem.report_failure()
        assert sem.circuit_state == _CircuitState.CLOSED

    async def test_success_resets_consecutive_count(self):
        sem = AdaptiveSemaphore(initial=1, min_limit=1, max_limit=5, cb_threshold=3)
        await sem.report_failure()
        await sem.report_failure()
        await sem.report_success()
        await sem.report_failure()
        assert sem.circuit_state == _CircuitState.CLOSED

    async def test_half_open_probe_success_closes(self):
        sem = AdaptiveSemaphore(initial=1, min_limit=1, max_limit=5, cb_threshold=3)
        for _ in range(3):
            await sem.report_failure()
        assert sem.circuit_state == _CircuitState.OPEN
        sem._cb_open_until = 0
        await sem.acquire()
        assert sem.circuit_state == _CircuitState.HALF_OPEN
        await sem.report_success()
        assert sem.circuit_state == _CircuitState.CLOSED

    async def test_half_open_probe_failure_reopens(self):
        sem = AdaptiveSemaphore(initial=1, min_limit=1, max_limit=5, cb_threshold=3)
        for _ in range(3):
            await sem.report_failure()
        assert sem.circuit_state == _CircuitState.OPEN
        sem._cb_open_until = 0
        await sem.acquire()
        assert sem.circuit_state == _CircuitState.HALF_OPEN
        await sem.report_failure()
        assert sem.circuit_state == _CircuitState.OPEN
        assert sem._cb_open_count == 2

    async def test_wait_if_circuit_open_returns_immediately_when_closed(self):
        sem = AdaptiveSemaphore(initial=1, min_limit=1, max_limit=5, cb_threshold=3)
        assert sem.circuit_state == _CircuitState.CLOSED
        await sem.wait_if_circuit_open()

    async def test_wait_if_circuit_open_blocks_until_half_open(self):
        sem = AdaptiveSemaphore(initial=1, min_limit=1, max_limit=5, cb_threshold=3)
        for _ in range(3):
            await sem.report_failure()
        assert sem.circuit_state == _CircuitState.OPEN
        sem._cb_open_until = time.monotonic() + 0.05
        await sem.wait_if_circuit_open()
        assert sem.circuit_state == _CircuitState.HALF_OPEN


class TestFilterPapers:
    async def test_two_pass_accepts_high_score(self, httpx_mock):
        paper = _make_paper(arxiv_id="2604.00001v1")
        body = {"choices": [{"message": {"content": '{"relevant": true, "score": 9, "reason": "great"}'}}]}
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=body)

        config = LLMConfig(use_html=False, use_pdf=False)
        outcome = await filter_papers([paper], config, "test-key")

        assert len(outcome.relevant) == 1
        assert outcome.relevant[0].score == 9
        assert outcome.relevant[0].pass_number == 1
        assert len(outcome.discarded) == 0

    async def test_two_pass_discards_low_score(self, httpx_mock):
        paper = _make_paper(arxiv_id="2604.00001v1")
        body = {"choices": [{"message": {"content": '{"relevant": false, "score": 2, "reason": "nope"}'}}]}
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=body)

        config = LLMConfig(use_html=False, use_pdf=False)
        outcome = await filter_papers([paper], config, "test-key")

        assert len(outcome.relevant) == 0
        assert len(outcome.discarded) == 1
        assert outcome.discarded[0].score == 2
        assert outcome.discarded[0].reason == "nope"

    async def test_two_pass_borderline_gets_pass2(self, httpx_mock):
        from unittest.mock import AsyncMock, patch

        paper = _make_paper(arxiv_id="2604.00001v1")

        pass1_body = {"choices": [{"message": {"content": '{"relevant": true, "score": 5, "reason": "maybe"}'}}]}
        pass2_body = {"choices": [{"message": {"content": '{"relevant": true, "score": 7, "reason": "yes after full text"}'}}]}

        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=pass1_body)
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=pass2_body)

        fake_content = {"2604.00001v1": {"html": "Full text here.", "pdf": None}}
        config = LLMConfig(use_html=True, use_pdf=False)

        with patch("src.filter.prefetch_content", new_callable=AsyncMock, return_value=fake_content):
            outcome = await filter_papers([paper], config, "test-key")

        assert len(outcome.relevant) == 1
        assert outcome.relevant[0].score == 7
        assert outcome.relevant[0].pass_number == 2

    async def test_two_pass_borderline_rejected_in_pass2(self, httpx_mock):
        from unittest.mock import AsyncMock, patch

        paper = _make_paper(arxiv_id="2604.00001v1")

        pass1_body = {"choices": [{"message": {"content": '{"relevant": true, "score": 5, "reason": "maybe"}'}}]}
        pass2_body = {"choices": [{"message": {"content": '{"relevant": false, "score": 3, "reason": "not relevant"}'}}]}

        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=pass1_body)
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=pass2_body)

        fake_content = {"2604.00001v1": {"html": "Full text here.", "pdf": None}}
        config = LLMConfig(use_html=True, use_pdf=False)

        with patch("src.filter.prefetch_content", new_callable=AsyncMock, return_value=fake_content):
            outcome = await filter_papers([paper], config, "test-key")

        assert len(outcome.relevant) == 0
        assert len(outcome.discarded) == 1
        assert outcome.discarded[0].score == 3

    async def test_sorted_by_score(self, httpx_mock):
        from unittest.mock import AsyncMock, patch

        papers = [_make_paper(arxiv_id=f"2604.0000{i}v1") for i in range(3)]
        for paper, score in zip(papers, [9, 3, 7]):
            body = {"choices": [{"message": {"content": json.dumps({"relevant": score > 3, "score": score, "reason": "ok"})}}]}
            httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=body)
        pass2_body = {"choices": [{"message": {"content": '{"relevant": true, "score": 7, "reason": "ok"}'}}]}
        httpx_mock.add_response(url="https://api.openai.com/v1/chat/completions", json=pass2_body)

        fake_content = {f"2604.0000{i}v1": {"html": "text", "pdf": None} for i in range(3)}
        config = LLMConfig(use_html=True, use_pdf=False, borderline_min=4, borderline_max=7)

        with patch("src.filter.prefetch_content", new_callable=AsyncMock, return_value=fake_content):
            outcome = await filter_papers(papers, config, "test-key")

        assert [r.score for r in outcome.relevant] == [9, 7]
        assert len(outcome.discarded) == 1
        assert outcome.discarded[0].score == 3

    async def test_empty_input(self):
        config = LLMConfig()
        outcome = await filter_papers([], config, "test-key")
        assert outcome.relevant == []
        assert outcome.discarded == []
