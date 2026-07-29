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

## Lessons from building the 9-chapter course

These held up repeatedly across chapters 5-9 and are worth applying to
any future agentic-AI project, not just this one:

- **Fast-moving AI/agent package versions break APIs often.** Before
  writing code against a newly-added package, verify the actual import
  paths work first (a quick `uv run python -c "from pkg import Thing"`
  per symbol you need). `a2a-sdk` 1.x dropped `A2AStarletteApplication`
  and several types entirely in a breaking protobuf-schema rewrite —
  caught only by testing imports before committing to a version. When you
  find a breaking change, pin the exact working version (`==`, not `>=`)
  and note why in a commit/comment, so a routine `uv add`/`uv sync`
  doesn't silently reintroduce the break.
- **Mocked tests are necessary but not sufficient for anything touching
  an external protocol or SDK.** Nearly every chapter's most consequential
  bug was invisible to mocked pytest and only surfaced via a real,
  unmocked run: chapter 7's DeepEval judge silently dropping schema
  enforcement (`except TypeError: pass` swallowed the real signature
  mismatch), chapter 8's four A2A wire-protocol mismatches (mocking
  `httpx` just re-encodes your own wrong assumption about the remote
  API), chapter 9's coaching-message collision (only visible by actually
  running the Streamlit UI). Always pair mocked unit tests with one real
  end-to-end verification for integration-shaped code.
- **Local Ollama model tags aren't interchangeable, even same-family.**
  `gemma4:12b-mlx` failed structured-output/tool-calling in ways
  `gemma4:12b` didn't. Verify empirically per-model rather than assuming
  a variant behaves the same as its sibling.
- **When a tutorial's own page doesn't yield full chapter source**, check
  for a companion reference repo (often linked in the article) and pull
  real code directly: `gh api repos/OWNER/REPO/contents/PATH --jq
  '.content' | base64 -d`. Much more reliable than re-scraping prose for
  code that was never fully rendered on the page.
- **Gate slow/non-deterministic LLM-calling tests behind a custom pytest
  marker**, not by leaving them out of the suite entirely:
  `markers = ["eval: ..."]` plus `addopts = "-m 'not eval'"` in
  `[tool.pytest.ini_options]` keeps the default `pytest` run fast and
  network-free, while `pytest -m eval` opts back in on demand (CLI `-m`
  correctly overrides the addopts-level one, no conflict).
- **subagent-driven-development (implementer subagent → task reviewer
  subagent → fix loop → final whole-branch review on the most capable
  model) reliably caught real bugs** that straight implementation would
  have shipped. Worth defaulting to for future multi-task work in this
  repo, not just as a one-off choice.
