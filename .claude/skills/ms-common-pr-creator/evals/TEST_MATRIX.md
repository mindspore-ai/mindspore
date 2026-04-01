# GitCode PR Test Matrix

This matrix splits `ms-pr-creator` testing into layers so we can keep most checks deterministic while still preserving a small amount of end-to-end coverage.

## Layer 1: Description Trigger

- Goal: verify whether the `SKILL.md` description causes the skill to trigger for the right user queries.
- Source: `evals/trigger_evals.json`
- Environment: no local git repo, no GitCode credentials, no GitCode project needed.
- Automation: `skill-creator` description eval flow (`run_eval.py` / `run_loop.py`).
- Current coverage:
  - Explicit `/ms-pr-creator`
  - Natural-language GitCode PR creation requests
  - Near misses such as GitHub, GitLab, PR status lookup, merge, close, review, and push requests
- Gaps to add later:
  - Ambiguous prompts that mention "PR" but only imply GitCode from context
  - Near misses that mention "GitCode" but actually ask for issue, code review, or branch operations
  - Queries that omit "PR" but clearly request the same workflow

## Layer 2: Simulated PR Body Generation

- Goal: verify template preservation and PR body quality for Step 3 and Step 4 of the skill.
- Source: `evals/evals.json` plus `evals/metadata/*.json`
- Environment: no local git repo, no GitCode credentials, no network.
- Automation: subagent execution plus workspace grading in `ms-pr-creator-workspace/`.
- Current coverage:
  - `akg` bugfix, quickfix
  - `mindspore` feature, large feature, mixed-commit refactor
  - `docs` task using the `mindspore` template
  - workflow gating cases for missing `--dst`, unpushed commits, unconfigured templates, configured-template continuation, missing credentials, auth failure, and accidental token leakage
  - credential-recovery replies that use the skill README.md as the user-facing source of truth while preserving local-only secret handling
  - Template preservation, `/kind` formatting, Chinese content, body structure, test plan quality, and LLM-judge semantic assertions
  - unconfigured-template guidance that surfaces the resolved default template path, keeps internal implementation docs out of the user reply, and points custom-template saves to the persistent `~/.config/gitcode-skills/pr_templates/` directory
  - in-chat template-authoring flow that turns a user-provided draft into a reusable template file in the normal user template directory, saves a short confirmation reply for review, then moves the generated template back into the eval's `outputs/` root after grading
  - template-authoring from a sparse English heading-only draft where the agent must infer sensible `{{FILL: ...}}` guidance on its own
  - template-authoring for a shared template that must emit correct multi-project YAML frontmatter for three named projects
- Note:
  - workflow-gating cases save the next user-facing reply as `outputs/assistant_reply.md`
  - body-generation cases continue to save the generated PR body as `outputs/pr_body.md`
- Gaps to add later:
  - More docs-heavy cases with no code tests
  - Very small change where the correct output should stay terse
  - More ambiguous PR type selection cases
  - More variations of local-only credential recovery wording across different failure surfaces

## Layer 3: Local Hermetic Script Tests

- Goal: verify deterministic `create_pr.py` behavior without talking to the real GitCode API.
- Environment: local filesystem only; temporary directories, temporary body files, and mocked git/API dependencies.
- Automation: standard local Python test runner.
- Why this layer matters:
  - It covers the non-LLM logic that the current simulated evals do not touch.
  - It catches regressions in exit codes, template routing, and input validation quickly.
- Cases to cover now:
  - `sanitize_pr_body()` strips leaked leading frontmatter
  - token-boundary project matching avoids accidental substring matches
  - user template precedence over bundled template
  - configured template lookup for `docs -> mindspore`
  - missing `upstream` remote exits with code `1`
  - missing `--base` exits with code `2`
  - missing body file / empty body after sanitization exits with code `3`
- Cases to add later:
  - `load_config()` precedence between environment variables and config file
  - `get_pr_title()` fallback behavior
  - `ensure_upstream_branch_fetched()` behavior with mocked git responses
  - `create_gitcode_pr()` request payload via mocked `urllib.request.urlopen`

## Layer 4: Live GitCode Smoke Tests

- Goal: verify the real create-PR path against GitCode itself.
- Environment:
  - real git repo
  - real `upstream` remote
  - pushed source branch on a fork
  - valid GitCode credentials
  - network access
- Automation: optional and sparse; do not make this the default benchmark path.
- Recommended cases:
  - one happy-path PR creation against a dedicated sandbox repository
  - one auth or permission failure case
- Why only a few:
  - these tests are slow, flaky, and environment-dependent
  - they are best treated as smoke tests rather than routine evals

## Recommended Default Mix

- Run Layer 1 and Layer 2 routinely when improving the skill.
- Run Layer 3 routinely when changing `create_pr.py`.
- Run Layer 4 only before releases or when touching real API behavior.
