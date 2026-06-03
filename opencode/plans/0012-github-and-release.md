# 0012 — GitHub & Release

---

## Repository Creation

Run after the initial local commit is ready:

```bash
gh repo create fortune-teller \
  --private \
  --source=. \
  --description "Local-first Tarot reading app — RAG over scraped definitions, local llama.cpp + sentence-transformers" \
  --remote=origin \
  --push
```

This command:
1. Creates the private `fortune-teller` repo under your GitHub user.
2. Sets the local remote to `origin`.
3. Pushes the current branch (`main`) to GitHub.

Requires the `gh` CLI and authentication (`gh auth status`).

---

## Initial Repository Setup (post-push)

```bash
# Set default branch (should already be main)
gh repo edit fortune-teller --default-branch main

# Add branch protection (requires GitHub account with that feature)
gh api repos/{owner}/fortune-teller/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["test (macos-14, 3.13)","test (ubuntu-latest, 3.13)"]}' \
  --field enforce_admins=false \
  --field required_pull_request_reviews=null \
  --field restrictions=null \
  --field allow_force_pushes=false \
  --field allow_deletions=false
```

---

## `.gitignore` (project-specific additions)

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.egg-info/
dist/
build/
.eggs/

# uv
.venv/
uv.lock         # commit this — remove from gitignore if you want lock tracked
                # (recommended: keep uv.lock tracked for reproducibility)

# macOS
.DS_Store
**/.DS_Store

# Test artefacts
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Data (never commit scraped or generated data)
data/

# Environment
.env
.envrc

# DuckDB
*.duckdb
*.duckdb.wal

# SQLite
*.db
*.sqlite
*.sqlite3

# Editor
.vscode/
.idea/
*.swp
*.swo
```

Note: `uv.lock` should be committed for reproducible installs. Remove the
`uv.lock` entry from `.gitignore` above.

---

## Branch Strategy

```
main                    ← protected; always CI-green
  └── feat/0004-domain-model
  └── feat/0005-deck-engine
  └── feat/0006-scraper-parser
  └── feat/0007-vector-store
  └── feat/0008-chains
  └── feat/0009-gradio-ui
  └── chore/tooling-setup
  └── docs/initial
```

Each plan item gets its own branch, PR against `main`, CI must pass.

---

## First Release: `v0.0.1-spike`

Once all acceptance criteria in `0013-spike-acceptance.md` are met:

```bash
git tag -a v0.0.1-spike -m "Spike: single deck, three-card spread, auto-deal"
git push origin v0.0.1-spike

gh release create v0.0.1-spike \
  --title "v0.0.1-spike — Book of Thoth, New Moon spread" \
  --notes "Initial spike. Auto-deal only. No auth. Requires local llama.cpp + sentence-transformers." \
  --prerelease
```

---

## `GITHUB_PERSONAL_ACCESS_TOKEN`

Required for the `github` MCP server. Create at:
`https://github.com/settings/tokens/new`

Scopes needed:
- `repo` (full)
- `workflow` (for CI inspection via MCP)

Store it in your shell environment (e.g. `~/.zshrc` or `.envrc`):
```bash
export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...
```

Do NOT commit this token anywhere in the repository.
