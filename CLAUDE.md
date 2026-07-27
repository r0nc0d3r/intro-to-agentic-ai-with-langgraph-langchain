# Repo Guidelines

Reference notes for how this repository is set up, to reuse as a template
for future projects.

## Baseline files

- **README.md** — project description, prerequisites, and a "Getting
  Started" section with two setup paths: `uv` (recommended) and standard
  `venv` + `pip`.
- **.gitignore** — Python-focused (`__pycache__`, `.venv`, `.env`, Jupyter
  checkpoints, IDE/OS cruft).
- **LICENSE** — MIT, with an added "Additional Disclaimer" paragraph
  clarifying no guarantee of accuracy/suitability, beyond the standard
  MIT "AS IS" clause. Copy this pattern for other repos that want an
  explicit no-guarantee note without switching license types.

## Python tooling: uv

Prefer `uv` over raw `venv`/`pip` in instructions and examples:

| pip/venv                         | uv equivalent      |
|-----------------------------------|---------------------|
| `python -m venv .venv`            | `uv venv`           |
| `pip install -r requirements.txt` | `uv sync`           |
| `pip install <pkg>`               | `uv add <pkg>`      |
| `source .venv/bin/activate && python script.py` | `uv run python script.py` |
| `pipx run <tool>`                 | `uvx <tool>`         |

Document both paths in the README (uv as primary, venv/pip as fallback) —
not everyone has uv installed yet.

## Branch protection

`main` is protected: **direct pushes are rejected**, changes must go
through a pull request (`GH013: Repository rule violations... Changes
must be made through a pull request.`). Set via GitHub UI → Settings →
Branches → Add rule → branch name `main` → "Require a pull request
before merging" (+ optionally "Do not allow bypassing the above
settings", "Block force pushes", "Restrict deletions").

Note: none of the available GitHub MCP tools expose branch-protection /
repository-settings management, and the default Actions `GITHUB_TOKEN`
has no `administration` scope either — that setting can only be applied
by a human in the GitHub UI (or via API using a manually-provisioned PAT
with `repo` scope). Don't try to talk around this restriction.

## Contribution workflow

- **CONTRIBUTING.md** — standard GitHub-recognized filename (not
  CONTRIBUTE/CONTRIBUTION); GitHub links to it automatically from new
  issues/PRs. Covers the PR-only workflow and commit message style
  (imperative summary line, ~50–72 chars, blank line + "why" body when
  non-obvious).
- **.github/pull_request_template.md** — minimal Summary + Test plan
  sections, auto-populated into new PRs on GitHub.

## Plugins / skills

The `superpowers@superpowers-marketplace` plugin (marketplace source:
`obra/superpowers-marketplace`) is installed at user scope, providing
skills for brainstorming, TDD, systematic debugging, writing/executing
plans, code review flows, etc. Installed via:

```bash
claude plugin marketplace add obra/superpowers-marketplace
claude plugin install superpowers@superpowers-marketplace
```

This is a user-level install (not project-scoped) — it doesn't appear in
repo files and applies across sessions on this machine.
