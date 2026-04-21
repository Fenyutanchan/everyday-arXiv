import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.fetcher import Paper
from src.filter import FilterOutcome, FilterResult
from src.main import build_llm_config, cmd_cleanup, cmd_fetch, cmd_filter, load_config


def _make_paper(**overrides) -> Paper:
    defaults = dict(
        arxiv_id="2604.15309v1",
        title="Test Paper",
        authors=["Alice"],
        abstract="Abstract.",
        categories=["hep-ph"],
        published="2026-04-16T10:00:00+00:00",
        updated="2026-04-16T10:00:00+00:00",
        pdf_url="https://arxiv.org/pdf/2604.15309v1",
        entry_url="https://arxiv.org/abs/2604.15309v1",
        html_url="https://arxiv.org/html/2604.15309v1",
    )
    defaults.update(overrides)
    return Paper(**defaults)


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
