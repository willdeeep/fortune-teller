# 0003 — MCP Servers

## Purpose

Project-local `.opencode/opencode.json` registers MCP servers that give
AI coding agents (and opencode sessions) access to up-to-date library
documentation, filesystem operations, and git/GitHub tooling — all scoped
to this project.

## Servers to Register

### `context7`
- **Purpose**: Official library documentation lookups.
- **Useful for**: langchain, langchain-core, langchain-openai,
  langchain-huggingface, pydantic v2, duckdb, gradio, sentence-transformers,
  httpx, pytest, ruff, mypy, uv.
- **Transport**: stdio via `npx -y @upstash/context7-mcp`

### `filesystem`
- **Purpose**: Scoped file read/write/list operations.
- **Scope**: Repo root only — do not grant access to `~` or system dirs.
- **Transport**: stdio via `npx -y @modelcontextprotocol/server-filesystem`

### `github`
- **Purpose**: Issue creation, PR management, branch operations.
- **Scope**: `fortune-teller` repo only.
- **Transport**: stdio via `npx -y @modelcontextprotocol/server-github`
- **Requires**: `GITHUB_PERSONAL_ACCESS_TOKEN` env var.

### `git`
- **Purpose**: Local git operations (status, diff, log, add, commit).
- **Transport**: stdio via `uvx mcp-server-git`

## Permission Rules to Configure

Allow without confirmation:
- `uv run *`, `uv sync *`, `uv add *`
- `pytest *`, `ruff *`, `mypy *`
- `git status`, `git diff`, `git log *`, `git add *`, `git commit *`
- Context7 docs lookups
- Filesystem reads within repo root

Require confirmation:
- `gh repo create`, `git push`, `git push --force`
- Any file deletion
- Writing outside the repo root
- Any outbound HTTP not going to a docs site

## `.opencode/opencode.json` Target Shape

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "context7": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    },
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/willhuntleyclarke/repos/fortune-teller"
      ]
    },
    "github": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}"
      }
    },
    "git": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-server-git", "--repository", "/Users/willhuntleyclarke/repos/fortune-teller"]
    }
  }
}
```

## Notes

- The `customize-opencode` skill should be loaded when writing
  `.opencode/opencode.json` to ensure the JSON shape conforms to current
  opencode conventions.
- `GITHUB_PERSONAL_ACCESS_TOKEN` must be set in the shell environment
  (e.g. via `.envrc` / direnv, or in the terminal session); it is NOT
  committed to the repo.
- `context7` requires Node.js to be available (`node --version` should work).
- `git` MCP server requires `uvx` (ships with `uv`).
