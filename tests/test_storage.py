import json
from pathlib import Path

import pytest

from src.fetcher import Paper
from src.filter import FilterResult
from src.storage import (
    _base_id,
    _cleanup_intermediate,
    _date_from_iso,
    _day_filename,
    _find_paper_index,
    _load_day,
    _paper_to_raw_dict,
    _result_to_dict,
    _remove_paper,
    _save_day,
    _upsert_paper,
    load_all_filtered,
    load_filtered,
    load_raw,
    latest_raw_date,
    load_results,
    save_filtered,
    save_papers,
    save_trash,
)


def _make_paper(**overrides) -> Paper:
    defaults = dict(
        arxiv_id="2604.15309v1",
        title="Test Paper",
        authors=["Alice"],
        abstract="Abstract text.",
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
    paper_kw = {k: v for k, v in overrides.items() if k in Paper.__dataclass_fields__}
    result_kw = {k: v for k, v in overrides.items()
                 if k in FilterResult.__dataclass_fields__ and k != "paper"}
    defaults = dict(relevant=True, score=5, reason="test")
    defaults.update(result_kw)
    return FilterResult(
        paper=_make_paper(**paper_kw),
        **defaults,
    )


class TestBaseId:
    def test_strips_version(self):
        assert _base_id("2604.15309v2") == "2604.15309"

    def test_no_version(self):
        assert _base_id("2604.15309") == "2604.15309"

    def test_v10(self):
        assert _base_id("2604.15309v10") == "2604.15309"


class TestDateFromIso:
    def test_basic(self):
        assert _date_from_iso("2026-04-16T10:00:00+00:00") == "2026-04-16"

    def test_date_only(self):
        assert _date_from_iso("2026-04-16") == "2026-04-16"


class TestDayFilename:
    def test_raw(self):
        assert _day_filename("2026-04-16", "raw") == "2026-04-16.json"

    def test_filtered(self):
        assert _day_filename("2026-04-16", "filtered") == "2026-04-16.filtered.json"


class TestPaperToRawDict:
    def test_no_relevance_field(self):
        paper = _make_paper()
        d = _paper_to_raw_dict(paper)
        assert d["arxiv_id"] == "2604.15309v1"
        assert d["base_id"] == "2604.15309"
        assert d["html_url"] == "https://arxiv.org/html/2604.15309v1"
        assert "relevance" not in d


class TestResultToDict:
    def test_has_relevance(self):
        result = _make_result()
        d = _result_to_dict(result)
        assert d["base_id"] == "2604.15309"
        assert d["relevance"]["score"] == 5
        assert d["relevance"]["relevant"] is True


class TestLoadSaveDay:
    def test_load_nonexistent(self, tmp_path):
        data = _load_day(tmp_path, "2026-04-16", "raw")
        assert data == {"date": "2026-04-16", "papers": []}

    def test_roundtrip_raw(self, tmp_path):
        data = {"date": "2026-04-16", "papers": [{"base_id": "2604.15309"}]}
        path = _save_day(tmp_path, data, "raw")
        assert path.name == "2026-04-16.json"
        loaded = json.loads(path.read_text())
        assert loaded["total"] == 1

    def test_roundtrip_filtered(self, tmp_path):
        data = {"date": "2026-04-16", "papers": [{"base_id": "2604.15309"}]}
        path = _save_day(tmp_path, data, "filtered")
        assert path.name == "2026-04-16.filtered.json"
        loaded = json.loads(path.read_text())
        assert loaded["total"] == 1


class TestFindPaperIndex:
    def test_found(self):
        data = {"papers": [{"base_id": "2604.11111"}, {"base_id": "2604.22222"}]}
        assert _find_paper_index(data, "2604.22222") == 1

    def test_not_found(self):
        assert _find_paper_index({"papers": []}, "2604.00000") == -1

    def test_fallback_to_arxiv_id(self):
        data = {"papers": [{"arxiv_id": "2604.11111v1"}]}
        assert _find_paper_index(data, "2604.11111") == 0


class TestUpsertPaper:
    def test_insert(self, tmp_path):
        paper_dict = {"base_id": "2604.15309", "arxiv_id": "2604.15309v1", "title": "A"}
        _upsert_paper(tmp_path, "2026-04-16", paper_dict, "raw")

        day = _load_day(tmp_path, "2026-04-16", "raw")
        assert len(day["papers"]) == 1
        assert day["papers"][0]["title"] == "A"

    def test_update_existing(self, tmp_path):
        _upsert_paper(tmp_path, "2026-04-16", {"base_id": "2604.15309", "arxiv_id": "2604.15309v1", "title": "v1"}, "raw")
        _upsert_paper(tmp_path, "2026-04-16", {"base_id": "2604.15309", "arxiv_id": "2604.15309v2", "title": "v2"}, "raw")

        day = _load_day(tmp_path, "2026-04-16", "raw")
        assert len(day["papers"]) == 1
        assert day["papers"][0]["title"] == "v2"


class TestRemovePaper:
    def test_remove_existing(self, tmp_path):
        _upsert_paper(tmp_path, "2026-04-16", {"base_id": "2604.15309", "arxiv_id": "2604.15309v1"}, "raw")
        _remove_paper(tmp_path, "2026-04-16", "2604.15309", "raw")

        day = _load_day(tmp_path, "2026-04-16", "raw")
        assert len(day["papers"]) == 0

    def test_remove_only_paper_deletes_file(self, tmp_path):
        _upsert_paper(tmp_path, "2026-04-16", {"base_id": "2604.15309", "arxiv_id": "2604.15309v1"}, "raw")
        _remove_paper(tmp_path, "2026-04-16", "2604.15309", "raw")

        assert not (tmp_path / "2026-04-16.json").exists()

    def test_remove_absent_is_noop(self, tmp_path):
        _remove_paper(tmp_path, "2026-04-16", "2604.15309", "raw")
        assert not (tmp_path / "2026-04-16.json").exists()

    def test_remove_nonexistent_id_is_noop(self, tmp_path):
        _upsert_paper(tmp_path, "2026-04-16", {"base_id": "2604.15309", "arxiv_id": "2604.15309v1"}, "raw")
        _remove_paper(tmp_path, "2026-04-16", "2604.99999", "raw")

        day = _load_day(tmp_path, "2026-04-16", "raw")
        assert len(day["papers"]) == 1


class TestCleanupIntermediate:
    def test_removes_intermediate(self, tmp_path):
        base = {"base_id": "2604.00001", "arxiv_id": "2604.00001v1"}

        _upsert_paper(tmp_path, "2026-04-10", base, "raw")
        _upsert_paper(tmp_path, "2026-04-12", base, "raw")
        _upsert_paper(tmp_path, "2026-04-14", base, "raw")

        _cleanup_intermediate(tmp_path, "2604.00001", "2026-04-10", "2026-04-14", "raw")

        assert len(_load_day(tmp_path, "2026-04-10", "raw")["papers"]) == 1
        assert len(_load_day(tmp_path, "2026-04-12", "raw")["papers"]) == 0
        assert len(_load_day(tmp_path, "2026-04-14", "raw")["papers"]) == 1

    def test_same_date_is_noop(self, tmp_path):
        base = {"base_id": "2604.00001", "arxiv_id": "2604.00001v1"}
        _upsert_paper(tmp_path, "2026-04-10", base, "raw")

        _cleanup_intermediate(tmp_path, "2604.00001", "2026-04-10", "2026-04-10", "raw")
        assert len(_load_day(tmp_path, "2026-04-10", "raw")["papers"]) == 1


class TestSavePapers:
    def test_first_submission_only(self, tmp_path):
        paper = _make_paper()
        paths = save_papers([paper], output_dir=tmp_path)

        assert len(paths) == 1
        assert paths[0].name == "2026-04-16.json"
        day = load_results(paths[0])
        assert day["total"] == 1
        assert "relevance" not in day["papers"][0]

    def test_replacement_creates_two_files(self, tmp_path):
        paper = _make_paper(
            published="2026-04-14T10:00:00+00:00",
            updated="2026-04-18T10:00:00+00:00",
        )
        paths = save_papers([paper], output_dir=tmp_path)

        assert len(paths) == 2
        names = {p.name for p in paths}
        assert names == {"2026-04-14.json", "2026-04-18.json"}

    def test_multiple_papers(self, tmp_path):
        p1 = _make_paper(arxiv_id="2604.11111v1")
        p2 = _make_paper(
            arxiv_id="2604.22222v1",
            published="2026-04-15T10:00:00+00:00",
            updated="2026-04-15T10:00:00+00:00",
        )
        paths = save_papers([p1, p2], output_dir=tmp_path)
        assert len(paths) == 2


class TestSaveFiltered:
    def test_saves_to_run_date_file(self, tmp_path):
        result = _make_result()
        paths = save_filtered([result], run_date="2026-04-20", output_dir=tmp_path)

        assert len(paths) == 1
        assert paths[0].name == "2026-04-20.json"
        day = load_results(paths[0])
        assert day["papers"][0]["relevance"]["score"] == 5
        assert day["total"] == 1

    def test_overwrite_on_same_run_date(self, tmp_path):
        r1 = _make_result(score=3)
        r2 = _make_result(score=8)
        save_filtered([r1], run_date="2026-04-20", output_dir=tmp_path)
        save_filtered([r2], run_date="2026-04-20", output_dir=tmp_path)

        loaded = load_filtered("2026-04-20", output_dir=tmp_path)
        assert len(loaded) == 1
        assert loaded[0]["relevance"]["score"] == 8

    def test_different_run_dates(self, tmp_path):
        r1 = _make_result(score=3)
        r2 = _make_result(score=8)
        save_filtered([r1], run_date="2026-04-19", output_dir=tmp_path)
        save_filtered([r2], run_date="2026-04-20", output_dir=tmp_path)

        d19 = load_filtered("2026-04-19", output_dir=tmp_path)
        d20 = load_filtered("2026-04-20", output_dir=tmp_path)
        assert len(d19) == 1
        assert len(d20) == 1
        assert d19[0]["relevance"]["score"] == 3
        assert d20[0]["relevance"]["score"] == 8


class TestRawFilteredIsolation:
    def test_raw_and_filtered_do_not_interfere(self, tmp_path):
        raw_dir = tmp_path / "data"
        filtered_dir = tmp_path / "relevant"
        raw_dir.mkdir()
        filtered_dir.mkdir()

        save_papers([_make_paper()], output_dir=raw_dir)
        save_filtered([_make_result()], run_date="2026-04-20", output_dir=filtered_dir)

        assert len(list(raw_dir.glob("*.json"))) >= 1
        assert len(list(filtered_dir.glob("*.json"))) == 1
        raw_papers = load_raw("2026-04-16", output_dir=raw_dir)
        filtered_papers = load_filtered("2026-04-20", output_dir=filtered_dir)
        assert "relevance" not in raw_papers[0]
        assert "relevance" in filtered_papers[0]


class TestLoadRaw:
    def test_load_existing(self, tmp_path):
        save_papers([_make_paper()], output_dir=tmp_path)
        papers = load_raw("2026-04-16", output_dir=tmp_path)
        assert len(papers) == 1
        assert papers[0]["base_id"] == "2604.15309"

    def test_load_missing(self, tmp_path):
        papers = load_raw("2026-04-16", output_dir=tmp_path)
        assert papers == []


class TestLoadFiltered:
    def test_load_existing(self, tmp_path):
        save_filtered([_make_result()], run_date="2026-04-20", output_dir=tmp_path)
        papers = load_filtered("2026-04-20", output_dir=tmp_path)
        assert len(papers) == 1

    def test_load_missing(self, tmp_path):
        papers = load_filtered("2026-04-20", output_dir=tmp_path)
        assert papers == []


class TestLoadAllFiltered:
    def test_loads_all_run_dates(self, tmp_path):
        save_filtered([_make_result(score=3)], run_date="2026-04-19", output_dir=tmp_path)
        save_filtered([_make_result(score=8)], run_date="2026-04-20", output_dir=tmp_path)

        all_papers = load_all_filtered(output_dir=tmp_path)
        assert len(all_papers) == 2

    def test_empty_dir(self, tmp_path):
        all_papers = load_all_filtered(output_dir=tmp_path / "nonexistent")
        assert all_papers == []


class TestLoadResults:
    def test_load(self, tmp_path):
        data = {"date": "2026-04-16", "papers": [{"base_id": "x"}]}
        path = tmp_path / "2026-04-16.json"
        path.write_text(json.dumps(data))

        loaded = load_results(path)
        assert loaded["date"] == "2026-04-16"


class TestSaveTrash:
    def test_saves_to_trash_dir(self, tmp_path):
        discarded = [_make_result(score=2, arxiv_id="2604.00001v1")]
        paths = save_trash(discarded, run_date="2026-04-19", output_dir=tmp_path / "trash")

        assert len(paths) == 1
        data = json.loads(paths[0].read_text())
        assert data["total"] == 1
        assert data["papers"][0]["relevance"]["score"] == 2

    def test_creates_dir(self, tmp_path):
        trash_dir = tmp_path / "new_trash"
        save_trash([_make_result(score=1)], run_date="2026-04-19", output_dir=trash_dir)
        assert trash_dir.is_dir()

    def test_merges_with_existing_trash(self, tmp_path):
        trash_dir = tmp_path / "trash"
        save_trash(
            [_make_result(score=1, arxiv_id="2604.00001v1"), _make_result(score=2, arxiv_id="2604.00002v1")],
            run_date="2026-04-19",
            output_dir=trash_dir,
        )
        save_trash(
            [_make_result(score=3, arxiv_id="2604.00003v1")],
            run_date="2026-04-19",
            output_dir=trash_dir,
        )

        data = json.loads((trash_dir / "2026-04-19.json").read_text())
        assert data["total"] == 3
        ids = [p["base_id"] for p in data["papers"]]
        assert "2604.00001" in ids
        assert "2604.00002" in ids
        assert "2604.00003" in ids

    def test_replaces_existing_entry(self, tmp_path):
        trash_dir = tmp_path / "trash"
        save_trash(
            [_make_result(score=1, arxiv_id="2604.00001v1")],
            run_date="2026-04-19",
            output_dir=trash_dir,
        )
        save_trash(
            [_make_result(score=5, arxiv_id="2604.00001v1")],
            run_date="2026-04-19",
            output_dir=trash_dir,
        )

        data = json.loads((trash_dir / "2026-04-19.json").read_text())
        assert data["total"] == 1
        assert data["papers"][0]["relevance"]["score"] == 5


class TestLatestRawDate:
    def test_returns_latest_date(self, tmp_path):
        for d in ["2026-04-10", "2026-04-15", "2026-04-12"]:
            (tmp_path / f"{d}.json").write_text("{}")
        assert latest_raw_date(tmp_path) == "2026-04-15"

    def test_empty_dir(self, tmp_path):
        assert latest_raw_date(tmp_path) is None

    def test_nonexistent_dir(self, tmp_path):
        assert latest_raw_date(tmp_path / "nope") is None
