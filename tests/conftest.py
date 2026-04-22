from src.fetcher import Paper
from src.filter import FilterResult


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


def _make_result(**overrides) -> FilterResult:
    paper_kw = {k: v for k, v in overrides.items() if k in Paper.__dataclass_fields__}
    result_kw = {k: v for k, v in overrides.items()
                 if k in FilterResult.__dataclass_fields__ and k != "paper"}
    defaults = dict(relevant=True, score=7, reason="test", pass_number=1)
    defaults.update(result_kw)
    return FilterResult(paper=_make_paper(**paper_kw), **defaults)
