import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.fetcher import Paper
from src.filter import FilterOutcome, FilterResult
from src.main import (
    build_llm_config,
    cmd_cleanup,
    cmd_fetch,
    cmd_filter,
    load_config,
    _filtered_dicts_to_results,
    cmd_refilter,
)

from .conftest import _make_paper, _make_result


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("categories:\n  - hep-ph\n")
        cfg = load_config(cfg_path)
        assert cfg["categories"] == ["hep-ph"]


class TestBuildLlmConfig:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        cfg = build_llm_config({})
        assert cfg.base_url == "https://api.openai.com/v1"
        assert cfg.model == "gpt-4o-mini"
        assert cfg.research_profile == ""

    def test_from_config(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        raw = {
            "llm": {
                "base_url": "http://localhost:11434/v1",
                "model": "llama3",
                "research_profile": "I like RLHF.",
            }
        }
        cfg = build_llm_config(raw)
        assert cfg.base_url == "http://localhost:11434/v1"
        assert cfg.model == "llama3"
        assert cfg.research_profile == "I like RLHF."

    def test_profile_from_file(self, tmp_path):
        profile_file = tmp_path / "profile.txt"
        profile_file.write_text("I love dark matter.")
        raw = {"llm": {"research_profile_file": str(profile_file)}}
        cfg = build_llm_config(raw)
        assert cfg.research_profile == "I love dark matter."

    def test_profile_file_missing_falls_back_to_inline(self, tmp_path):
        raw = {
            "llm": {
                "research_profile_file": str(tmp_path / "nonexistent.txt"),
                "research_profile": "Inline profile.",
            }
        }
        cfg = build_llm_config(raw)
        assert cfg.research_profile == "Inline profile."

    def test_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "http://ollama:11434/v1")
        cfg = build_llm_config({"llm": {"base_url": "https://api.openai.com/v1"}})
        assert cfg.base_url == "http://ollama:11434/v1"

    def test_max_retries_defaults_to_dataclass_default(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        cfg = build_llm_config({})
        assert cfg.max_retries == 10

    def test_max_retries_from_config(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        cfg = build_llm_config({"llm": {"max_retries": 5}})
        assert cfg.max_retries == 5


class TestCmdFetch:
    async def test_saves_raw_data(self, tmp_path):
        papers = [_make_paper(), _make_paper(arxiv_id="2604.22222v1")]

        with patch("src.main.fetch_papers", return_value=papers), \
             patch("src.main._resolve_html_urls", new_callable=AsyncMock):
            cfg = {"categories": ["hep-ph"], "fetch": {"max_results": 10}, "output": {"dir": str(tmp_path)}}
            paths = await cmd_fetch(cfg, days=1)

        assert len(paths) >= 1
        day = json.loads((tmp_path / "2026-04-16.json").read_text())
        assert day["total"] == 2
        assert "relevance" not in day["papers"][0]

    async def test_no_papers_exits(self, tmp_path):
        with patch("src.main.fetch_papers", return_value=[]):
            cfg = {"categories": ["hep-ph"], "output": {"dir": str(tmp_path)}}
            with pytest.raises(SystemExit):
                await cmd_fetch(cfg, days=1)

    async def test_fetch_failure_returns_empty(self, tmp_path):
        with patch("src.main.fetch_papers", side_effect=Exception("API error")):
            cfg = {"categories": ["hep-ph"], "output": {"dir": str(tmp_path)}}
            paths = await cmd_fetch(cfg, days=1)
        assert paths == []


class TestCmdFilter:
    async def test_reads_raw_and_saves_filtered(self, tmp_path):
        from src.storage import save_papers
        save_papers([_make_paper()], output_dir=tmp_path)

        mock_results = [
            FilterResult(
                paper=_make_paper(),
                relevant=True,
                score=8,
                reason="matches",
            )
        ]

        mock_outcome = FilterOutcome(relevant=mock_results, discarded=[])

        with patch("src.main.filter_papers", new_callable=AsyncMock, return_value=mock_outcome), \
             patch.dict("os.environ", {"LLM_API_KEY": "sk-test"}):
            cfg = {
                "output": {"raw_dir": str(tmp_path), "filtered_dir": str(tmp_path / "relevant")},
                "llm": {
                    "research_profile": "I like physics.",
                    "api_key_env": "LLM_API_KEY",
                },
            }
            paths = await cmd_filter(cfg, date_str="2026-04-16")

        assert len(paths) == 1
        assert paths[0].name == "2026-04-16.json"
        day = json.loads(paths[0].read_text())
        assert day["papers"][0]["relevance"]["score"] == 8

    async def test_no_profile_exits(self, tmp_path):
        from src.storage import save_papers
        save_papers([_make_paper()], output_dir=tmp_path)

        cfg = {"output": {"raw_dir": str(tmp_path)}, "llm": {}}
        with pytest.raises(SystemExit):
            await cmd_filter(cfg, date_str="2026-04-16")


class TestCmdCleanup:
    def test_removes_old_trash(self, tmp_path):
        from src.storage import save_trash

        old_result = FilterResult(
            paper=Paper("2604.00001v1", "T", ["A"], "abs", ["hep-ph"],
                         "2026-01-01", "2026-01-01", "pdf", "entry", "N/A"),
            relevant=False, score=1, reason="no",
        )
        new_result = FilterResult(
            paper=Paper("2604.00002v1", "T", ["A"], "abs", ["hep-ph"],
                         "2026-04-16", "2026-04-16", "pdf", "entry", "N/A"),
            relevant=False, score=2, reason="no",
        )

        save_trash([old_result], run_date="2026-01-01", output_dir=tmp_path / "trash")
        save_trash([new_result], run_date="2026-04-16", output_dir=tmp_path / "trash")

        cfg = {"output": {"trash_dir": str(tmp_path / "trash")}}
        removed = cmd_cleanup(cfg, keep_days=7)

        assert len(removed) == 1
        assert removed[0].stem == "2026-01-01"
        assert (tmp_path / "trash" / "2026-04-16.json").exists()

    def test_no_trash_dir(self, tmp_path):
        cfg = {"output": {"trash_dir": str(tmp_path / "nonexistent")}}
        removed = cmd_cleanup(cfg, keep_days=7)
        assert removed == []

    async def test_no_api_key_exits(self, tmp_path):
        from src.storage import save_papers
        save_papers([_make_paper()], output_dir=tmp_path)

        with patch.dict("os.environ", {}, clear=True):
            cfg = {
                "output": {"raw_dir": str(tmp_path)},
                "llm": {"research_profile": "I like physics."},
            }
            with pytest.raises(SystemExit):
                await cmd_filter(cfg, date_str="2026-04-16")

    async def test_no_raw_data_exits(self, tmp_path):
        with patch.dict("os.environ", {"LLM_API_KEY": "sk-test"}):
            cfg = {
                "output": {"raw_dir": str(tmp_path)},
                "llm": {"research_profile": "I like physics."},
            }
            with pytest.raises(SystemExit):
                await cmd_filter(cfg, date_str="2026-04-16")

    async def test_auto_selects_latest_date(self, tmp_path):
        from src.storage import save_papers
        save_papers([_make_paper()], output_dir=tmp_path)

        mock_outcome = FilterOutcome(
            relevant=[FilterResult(paper=_make_paper(), relevant=True, score=9, reason="yes")],
            discarded=[],
        )

        with patch("src.main.filter_papers", new_callable=AsyncMock, return_value=mock_outcome), \
             patch.dict("os.environ", {"LLM_API_KEY": "sk-test"}):
            cfg = {
                "output": {"raw_dir": str(tmp_path), "filtered_dir": str(tmp_path / "out")},
                "llm": {"research_profile": "physics", "api_key_env": "LLM_API_KEY"},
            }
            paths = await cmd_filter(cfg)

        assert len(paths) == 1
        assert paths[0].name == "2026-04-16.json"

    async def test_no_date_arg_and_no_raw_data_exits(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch.dict("os.environ", {"LLM_API_KEY": "sk-test"}):
            cfg = {
                "output": {"raw_dir": str(empty_dir)},
                "llm": {"research_profile": "physics"},
            }
            with pytest.raises(SystemExit):
                await cmd_filter(cfg)


class TestFilteredDictsToResults:
    def test_converts_dict_to_result(self):
        paper = _make_paper()
        paper_dict = {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "categories": paper.categories,
            "published": paper.published,
            "updated": paper.updated,
            "pdf_url": paper.pdf_url,
            "entry_url": paper.entry_url,
            "html_url": paper.html_url,
            "relevance": {"relevant": True, "score": 8, "reason": "good", "pass_number": 2, "evaluated": True},
        }
        results = _filtered_dicts_to_results([paper_dict])
        assert len(results) == 1
        r = results[0]
        assert r.relevant is True
        assert r.score == 8
        assert r.reason == "good"
        assert r.pass_number == 2
        assert r.evaluated is True
        assert r.paper.arxiv_id == paper.arxiv_id

    def test_defaults_when_relevance_missing(self):
        paper = _make_paper()
        paper_dict = {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "categories": paper.categories,
            "published": paper.published,
            "updated": paper.updated,
            "pdf_url": paper.pdf_url,
            "entry_url": paper.entry_url,
            "html_url": paper.html_url,
        }
        results = _filtered_dicts_to_results([paper_dict])
        assert len(results) == 1
        r = results[0]
        assert r.relevant is False
        assert r.score == 0
        assert r.evaluated is True

    def test_empty_list(self):
        assert _filtered_dicts_to_results([]) == []

    def test_multiple_papers(self):
        p1 = _make_paper()
        p2 = _make_paper(arxiv_id="2604.22222v1")
        dicts = [
            {"arxiv_id": p1.arxiv_id, "title": p1.title, "authors": p1.authors,
             "abstract": p1.abstract, "categories": p1.categories, "published": p1.published,
             "updated": p1.updated, "pdf_url": p1.pdf_url, "entry_url": p1.entry_url,
             "html_url": p1.html_url,
             "relevance": {"relevant": True, "score": 9, "reason": "great"}},
            {"arxiv_id": p2.arxiv_id, "title": p2.title, "authors": p2.authors,
             "abstract": p2.abstract, "categories": p2.categories, "published": p2.published,
             "updated": p2.updated, "pdf_url": p2.pdf_url, "entry_url": p2.entry_url,
             "html_url": p2.html_url,
             "relevance": {"relevant": False, "score": 2, "reason": "nope"}},
        ]
        results = _filtered_dicts_to_results(dicts)
        assert len(results) == 2
        assert results[0].score == 9
        assert results[1].score == 2


class TestCmdRefilter:
    def _make_filtered_dict(self, evaluated=True, **relevance_overrides):
        paper = _make_paper()
        rel = {"relevant": True, "score": 8, "reason": "ok", "pass_number": 1, "evaluated": evaluated}
        rel.update(relevance_overrides)
        return {
            "arxiv_id": paper.arxiv_id, "title": paper.title, "authors": paper.authors,
            "abstract": paper.abstract, "categories": paper.categories, "published": paper.published,
            "updated": paper.updated, "pdf_url": paper.pdf_url, "entry_url": paper.entry_url,
            "html_url": paper.html_url, "relevance": rel,
        }

    async def test_refilters_unevaluated_papers(self, tmp_path):
        from src.storage import save_filtered as save_filt
        unevaluated = self._make_filtered_dict(evaluated=False)
        already_done = self._make_filtered_dict(
            arxiv_id="2604.22222v1" if "arxiv_id" not in {} else "",
            evaluated=True, score=9,
        )
        paper = _make_paper()
        already_done["arxiv_id"] = "2604.22222v1"
        save_filt(
            [_make_result(paper=_make_paper(), relevant=True, score=9, reason="done")],
            run_date="2026-04-20", output_dir=tmp_path,
        )
        raw_file = tmp_path / "2026-04-20.json"
        data = json.loads(raw_file.read_text())
        data["papers"].append(self._make_filtered_dict(
            arxiv_id="2604.33333v1", evaluated=False,
        ))
        data["papers"][1]["arxiv_id"] = "2604.33333v1"
        raw_file.write_text(json.dumps(data))

        mock_outcome = FilterOutcome(
            relevant=[FilterResult(
                paper=_make_paper(arxiv_id="2604.33333v1"), relevant=True, score=6, reason="refiltered",
            )],
            discarded=[],
        )

        with patch("src.main.filter_papers", new_callable=AsyncMock, return_value=mock_outcome), \
             patch.dict("os.environ", {"LLM_API_KEY": "sk-test"}):
            cfg = {
                "output": {"filtered_dir": str(tmp_path), "trash_dir": str(tmp_path / "trash")},
                "llm": {"research_profile": "physics", "api_key_env": "LLM_API_KEY"},
            }
            paths = await cmd_refilter(cfg, date_str="2026-04-20")

        assert len(paths) == 1
        result_data = json.loads(paths[0].read_text())
        assert result_data["total"] == 2

    async def test_all_evaluated_exits(self, tmp_path):
        from src.storage import save_filtered as save_filt
        save_filt(
            [_make_result()],
            run_date="2026-04-20", output_dir=tmp_path,
        )
        with patch.dict("os.environ", {"LLM_API_KEY": "sk-test"}):
            cfg = {
                "output": {"filtered_dir": str(tmp_path)},
                "llm": {"research_profile": "physics"},
            }
            with pytest.raises(SystemExit):
                await cmd_refilter(cfg, date_str="2026-04-20")

    async def test_no_filtered_data_exits(self, tmp_path):
        empty = tmp_path / "relevant"
        empty.mkdir()
        with patch.dict("os.environ", {"LLM_API_KEY": "sk-test"}):
            cfg = {
                "output": {"filtered_dir": str(empty)},
                "llm": {"research_profile": "physics"},
            }
            with pytest.raises(SystemExit):
                await cmd_refilter(cfg, date_str="2026-04-20")

    async def test_auto_selects_latest_date(self, tmp_path):
        from src.storage import save_filtered as save_filt
        save_filt(
            [_make_result()],
            run_date="2026-04-19", output_dir=tmp_path,
        )
        d = self._make_filtered_dict(evaluated=False)
        d["arxiv_id"] = "2604.22222v1"
        save_filt(
            [_make_result(paper=_make_paper(arxiv_id="2604.22222v1"), evaluated=False)],
            run_date="2026-04-20", output_dir=tmp_path,
        )
        raw_file = tmp_path / "2026-04-20.json"
        data = json.loads(raw_file.read_text())
        data["papers"][0]["relevance"]["evaluated"] = False
        raw_file.write_text(json.dumps(data))

        mock_outcome = FilterOutcome(
            relevant=[FilterResult(
                paper=_make_paper(arxiv_id="2604.22222v1"), relevant=True, score=5, reason="ok",
            )],
            discarded=[],
        )
        with patch("src.main.filter_papers", new_callable=AsyncMock, return_value=mock_outcome), \
             patch.dict("os.environ", {"LLM_API_KEY": "sk-test"}):
            cfg = {
                "output": {"filtered_dir": str(tmp_path), "trash_dir": str(tmp_path / "trash")},
                "llm": {"research_profile": "physics", "api_key_env": "LLM_API_KEY"},
            }
            paths = await cmd_refilter(cfg)

        assert len(paths) == 1
        assert paths[0].name == "2026-04-20.json"

    async def test_no_api_key_exits(self, tmp_path):
        from src.storage import save_filtered as save_filt
        save_filt([_make_result()], run_date="2026-04-20", output_dir=tmp_path)
        with patch.dict("os.environ", {}, clear=True):
            cfg = {
                "output": {"filtered_dir": str(tmp_path)},
                "llm": {"research_profile": "physics", "api_key_env": "LLM_API_KEY"},
            }
            with pytest.raises(SystemExit):
                await cmd_refilter(cfg, date_str="2026-04-20")

    async def test_no_profile_exits(self, tmp_path):
        with patch.dict("os.environ", {"LLM_API_KEY": "sk-test"}):
            cfg = {"output": {"filtered_dir": str(tmp_path)}, "llm": {}}
            with pytest.raises(SystemExit):
                await cmd_refilter(cfg, date_str="2026-04-20")
