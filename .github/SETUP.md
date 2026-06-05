# GitHub Setup — Deferred Operations

This document captures the **non-reversible GitHub operations** from plan
0012 that you asked to defer. Run them manually when you're ready, in
the order shown.

> All commands assume you're in the repository root, authenticated to
> GitHub as `willdeeep` (`gh auth status`), and on the `main` branch.

---

## 1. Create the Private GitHub Repository

This creates the private `fortune-teller` repo and pushes the local
`main` branch.

```bash
gh repo create fortune-teller \
  --private \
  --source=. \
  --description "Local-first Tarot reading app — RAG over scraped definitions, local llama.cpp + sentence-transformers" \
  --remote=origin \
  --push
```

**Verify:** `gh repo view willdeeep/fortune-teller --json visibility` should
return `visibility: PRIVATE`.

---

## 2. Confirm Default Branch

```bash
gh repo edit fortune-teller --default-branch main
```

**Verify:** `gh repo view willdeeep/fortune-teller --json defaultBranchRef`
should show `name: "main"`.

---

## 3. Configure Branch Protection on `main`

This requires **two CI status checks** to be passing before `main` can
be updated. Replace the contexts with the exact names shown in your
first CI run.

```bash
gh api repos/willdeeep/fortune-teller/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["test (macos-14, 3.13)","test (ubuntu-latest, 3.13)"]}' \
  --field enforce_admins=false \
  --field required_pull_request_reviews=null \
  --field restrictions=null \
  --field allow_force_pushes=false \
  --field allow_deletions=false
```

> **Heads up:** the strict mode means the branch must be **up-to-date**
> with `main` before merging. The current branch protection does not
> require a code review (no `required_pull_request_reviews`), so single-
> owner projects can merge without an approver. Add review
> requirements later with:
> `gh api ... --field required_pull_request_reviews='{"dismiss_stale_reviews":true,"require_code_owner_reviews":true}'`

**Verify:** `gh api repos/willdeeep/fortune-teller/branches/main/protection | jq '.required_status_checks.contexts'`
should list both context names.

---

## 4. Set Up `GITHUB_PERSONAL_ACCESS_TOKEN` (for opencode MCP)

The `github` MCP server needs a PAT. Create one at
<https://github.com/settings/tokens/new> with these scopes:

- `repo` (full)
- `workflow` (for CI inspection via MCP)

Store it in your shell environment (e.g. `~/.zshrc` or `.envrc`):

```bash
export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...
```

> **Do not** commit this token to the repository. `.env` and `.envrc`
> are already in `.gitignore`.

---

## 5. First Release: `v0.0.1-spike`

Once all acceptance criteria in plan 0013 are met:

```bash
git tag -a v0.0.1-spike -m "Spike: single deck, three-card spread, auto-deal"
git push origin v0.0.1-spike

gh release create v0.0.1-spike \
  --title "v0.0.1-spike — Book of Thoth, New Moon spread" \
  --notes "Initial spike. Auto-deal only. No auth. Requires local llama.cpp + sentence-transformers." \
  --prerelease
```

---

## 6. Branch Strategy (reference)

```
main                    ← protected; always CI-green
  └── feat/<slug>
  └── fix/<slug>
  └── chore/<slug>
  └── docs/<slug>
```

Each plan item gets its own branch, PR against `main`, CI must pass.
Branch names follow Conventional Commits prefixes (`feat/`, `fix/`,
`chore/`, `docs/`, `refactor/`, `test/`).

---

## Already in Place (no action needed)

- ✅ Local Git repository with `main` branch
- ✅ `.gitignore` complete (Python, uv, macOS, test artefacts, `data/`,
     `.env`, DuckDB, SQLite, editor scratch files, `.opencode/`)
- ✅ `uv.lock` tracked for reproducible installs
- ✅ `.pre-commit-config.yaml` (ruff, ruff-format, mypy, fast pytest)
- ✅ `.github/workflows/ci.yml` (matrix: macos-14 + ubuntu-latest,
     Python 3.13, ruff + mypy + pytest + coverage upload)
- ✅ Conventional Commits commit convention in use
