from src.fetcher import Paper, _build_html_url, _check_html_available, _resolve_html_urls


class TestBuildHtmlUrl:
    def test_basic(self):
        assert _build_html_url("2604.15309v1") == "https://arxiv.org/html/2604.15309v1"

    def test_no_version(self):
        assert _build_html_url("2604.15309") == "https://arxiv.org/html/2604.15309"


class TestCheckHtmlAvailable:
    async def test_available(self, httpx_mock):
        httpx_mock.add_response(
            url="https://arxiv.org/html/2604.15309v1",
            status_code=200,
        )
        arxiv_id, url = await _check_html_available("2604.15309v1")
        assert arxiv_id == "2604.15309v1"
        assert url == "https://arxiv.org/html/2604.15309v1"

    async def test_not_available(self, httpx_mock):
        httpx_mock.add_response(
            url="https://arxiv.org/html/2604.00000v1",
            status_code=404,
        )
        arxiv_id, url = await _check_html_available("2604.00000v1")
        assert url == "N/A"

    async def test_network_error(self, httpx_mock):
        import httpx
        httpx_mock.add_exception(httpx.ConnectTimeout("timeout"))
        arxiv_id, url = await _check_html_available("2604.00000v1")
        assert url == "N/A"


class TestResolveHtmlUrls:
    async def test_batch_resolution(self, httpx_mock):
        httpx_mock.add_response(url="https://arxiv.org/html/2604.11111v1", status_code=200)
        httpx_mock.add_response(url="https://arxiv.org/html/2604.22222v1", status_code=404)

        papers = [
            Paper(
                arxiv_id="2604.11111v1", title="A", authors=[], abstract="",
                categories=[], published="2026-04-16T10:00:00+00:00",
                updated="2026-04-16T10:00:00+00:00",
                pdf_url="https://arxiv.org/pdf/2604.11111v1",
                entry_url="https://arxiv.org/abs/2604.11111v1",
                html_url="",
            ),
            Paper(
                arxiv_id="2604.22222v1", title="B", authors=[], abstract="",
                categories=[], published="2026-04-16T10:00:00+00:00",
                updated="2026-04-16T10:00:00+00:00",
                pdf_url="https://arxiv.org/pdf/2604.22222v1",
                entry_url="https://arxiv.org/abs/2604.22222v1",
                html_url="",
            ),
        ]

        await _resolve_html_urls(papers)
        assert papers[0].html_url == "https://arxiv.org/html/2604.11111v1"
        assert papers[1].html_url == "N/A"
