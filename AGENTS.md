# Project Guidance / 项目指南

Everyday arXiv — a daily arXiv paper fetcher with LLM-based relevance filtering.
Full documentation: [`docs/index.md`](docs/index.md) (deployed at [/guide/](https://fytc.ac/guide/)).

> 每日 arXiv 论文获取与 LLM 智能筛选框架。完整文档：[`docs/index.md`](docs/index.md)（部署在 [/guide/](https://fytc.ac/guide/)）。

## Setup / 环境搭建

```bash
uv sync                  # Runtime / 运行时
uv sync --extra dev      # With tests / 含测试
```

## Workflow / 开发流程

```bash
uv run python -m src.main fetch --days 1          # Fetch / 拉取
export LLM_API_KEY="sk-..."
uv run python -m src.main filter --date 2026-04-19 # Filter / 筛选
uv run pytest -v                                   # Test / 测试
```

## Key Rules / 关键规则

- **Every code change MUST update corresponding tests.** / **每次代码变更都必须同步更新测试。**
- External dependencies (arXiv API, LLM API) must be mocked in tests. / 外部依赖必须 mock。
- Package manager: `uv` only. Do NOT add requirements.txt or setup.py. / 包管理器仅用 `uv`。
- Python ≥ 3.11, type hints everywhere, English docstrings. / Python ≥ 3.11，全面类型注解。
- **No comments in code unless explicitly requested.** / **除非明确要求，否则不添加注释。**

## AI Coding Assistant Policy / AI 编码助手政策

Adapted from [Linux Kernel AI Coding Assistants Guidelines](https://docs.kernel.org/process/coding-assistants.html).

> 参考 [Linux 内核 AI 编码助手指南](https://docs.kernel.org/process/coding-assistants.html) 改编。

### Human Responsibility / 人类责任

- **AI agents MUST NOT commit or push code autonomously.** Only humans may authorize commits. / **AI 不得自动提交或推送代码，必须由人类授权。**
- The human submitter is responsible for: / 代码提交者需负责：
  - Reviewing all AI-generated code before merging / 合并前审查所有 AI 生成的代码
  - Ensuring code quality, correctness, and security / 确保代码质量、正确性和安全性
  - Taking full responsibility for the contribution / 对提交内容负全部责任

### Attribution / 署名

When AI tools contribute significantly to this project, use the following format in commit messages:

> 当 AI 工具对本项目做出显著贡献时，在 commit message 中使用以下格式：

```
Assisted-by: AGENT_NAME:MODEL_VERSION [TOOL1] [TOOL2]
```

- `AGENT_NAME`: AI tool or framework name (e.g., Claude, GPT, Cursor) / AI 工具名称
- `MODEL_VERSION`: Specific model version (e.g., claude-3-opus, gpt-4o) / 具体模型版本
- `[TOOL1] [TOOL2]`: Optional analysis tools used (e.g., pytest, ruff, mypy) / 可选的分析工具

Basic tools (git, python, uv, editors) need not be listed. / 基础工具（git、python、uv、编辑器）无需列出。

Example:

```
Assisted-by: Claude:claude-3-opus pytest ruff
```

### What AI Should NOT Do / AI 不应做的事

- Do NOT add license headers or copyright claims. / 不要添加许可证头或版权声明。
- Do NOT make architectural decisions without human approval. / 未经人类批准，不要做架构决策。
- Do NOT silently modify existing behavior — always explain proposed changes. / 不要静默修改现有行为——始终解释提议的变更。
- Do NOT introduce new dependencies without checking with the user. / 不要未经用户同意引入新依赖。
