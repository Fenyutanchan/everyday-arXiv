# Everyday arXiv

Daily arXiv paper fetcher with LLM-based relevance filtering.  **[Full documentation →](https://fytc.ac/guide/)**

> 每日 arXiv 论文获取与 LLM 智能筛选框架。**[完整文档 →](https://fytc.ac/guide/)**

```bash
uv sync

# Fetch papers (no API key needed)
uv run python -m src.main fetch --days 1

# Filter with LLM
export LLM_API_KEY="sk-..."
uv run python -m src.main filter --date 2026-04-19
```

## License

MIT
