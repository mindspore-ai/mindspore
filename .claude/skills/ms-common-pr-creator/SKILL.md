---
name: ms-pr-creator
description: Create a GitCode pull request for the current branch in a MindSpore-family repository, automatically analyzing code changes and generating a Chinese PR description. Use when the user is working on a MindSpore repository and asks to create a GitCode PR/merge request, including requests like "提个 mindspore 的 GitCode PR", "提个 GitCode PR", "在 gitcode 上提 PR", "创建 PR 到 main 分支". Do not use for generic non-MindSpore GitCode projects, and do not use for GitHub or GitLab PRs.
---

# ms-pr-creator

Create GitCode pull requests with template-driven, LLM-generated descriptions.

## Security Boundaries

- Treat `access_token` as a secret that must stay on the user's machine.
- Never ask the user to send `access_token`, environment-variable values, config-file contents, or screenshots/copies of either.
- Explicitly remind the user not to paste tokens or credential files into chat because that can expose sensitive information to the agent/LLM provider.
- Never read credential values directly from environment variables or `~/.config/gitcode-skills/config.json` / `%USERPROFILE%\.config\gitcode-skills\config.json`.
- Never run commands that print or dump secret-bearing sources, such as `echo $GITCODE_ACCESS_TOKEN`, `Get-ChildItem Env:`, `env`, `printenv`, or `cat/Get-Content` on the shared credentials file.
- It is OK to run `scripts/create_pr.py` as part of the workflow. The script may consume locally configured credentials internally, but the agent must not inspect, echo, summarize, or validate the secret values themselves.
- If the user accidentally sends a token, stop using that value, avoid repeating it, and tell the user to revoke/rotate it in GitCode.
- When credential guidance is needed, give only the minimum necessary summary in chat, point the user to [README.md](README.md) for local credential setup steps, and use [CREDENTIALS_SETUP.md](references/CREDENTIALS_SETUP.md) as the internal safety workflow reference.

## Quick Start

```bash
# With target branch specified - create PR directly
/ms-pr-creator --dst main

# Without target branch - will ask user in Chinese
/ms-pr-creator
```

## Workflow

1. **Pre-flight check**:
   - Verify the current branch has been pushed to the remote fork:
     `git log origin/<current_branch>..HEAD --oneline`
   - If there are unpushed commits, remind the user to `git push` first.
     The GitCode API creates PRs by referencing the remote head branch — if unpushed, the API will fail with a "branch not found" error.

2. **Resolve target branch**:
   - If `--dst <branch>` provided → use directly
   - Otherwise → ask: "请问 PR 的目标分支是哪个？" and **wait for response**
   - Optionally show available branches: `git branch -r --list 'upstream/*'`

3. **Read the PR template**:
   - Run: `python <skill_dir>/scripts/create_pr.py --template-status`
   - The output includes the resolved template source and file path — read that file with the Read tool
   - If the project is already configured, continue directly with that template
   - If the project is still using the bundled default template:
     - tell the user that the current project does not yet have a dedicated PR template and will use the resolved default template file; prefer the full path from `--template-status`, not only the bare filename
     - offer two user-facing choices:
       1. continue with the current default template
       2. ask the user to provide a PR template in chat, then help organize and configure it as a reusable project-specific template
     - if the user chooses in-chat configuration, collect the PR template content from the user, then read [TEMPLATE_AUTHORING.md](references/TEMPLATE_AUTHORING.md) and follow that workflow to generate, save, and verify the template
     - keep manual maintenance as a backup path for advanced users
   - Use [IMPLEMENTATION.md](references/IMPLEMENTATION.md) for discovery/runtime mechanics and [TEMPLATE_AUTHORING.md](references/TEMPLATE_AUTHORING.md) for template-writing mechanics. Do not tell the user to go read internal repo docs such as `IMPLEMENTATION.md`.

4. **Analyze code and generate PR body** (only after steps 1-3 are resolved):

   This is the core step. Analyze the code changes, then produce the **complete PR body**
   by filling in the template's `{{FILL:...}}` markers while preserving all other content verbatim.

   a. **Analyze changes** — query `git log` and `git diff` against `upstream/<target_branch>`,
      read key changed files to understand the intent and scope of the PR.

   b. **Generate body** — take the template content and replace each `{{FILL: instruction}}`
      marker with the appropriate generated content following the instruction inside the marker.
      Everything outside `{{FILL:...}}` markers is fixed template structure — copy it exactly as-is,
      character for character. This includes HTML comments, section headers, checklists, `/kind` lines, etc.

      **Three critical rules:**

      1. **Skip template frontmatter** — the top `---` YAML block is template metadata, not PR body.
         Your output starts after that block. Never start the final PR body with `---`.

      2. **Preserve fixed template structure exactly** — everything outside `{{FILL:...}}` is fixed:
         headers, HTML comments, checklists, `/kind` lines, URLs, and other bot-readable text.

      3. **Consume agent-only comments, then remove them completely** — `<!-- AGENT_INSTRUCTION — ... -->`
         comments are instructions for generation only. Follow them, then delete the entire comment line
         from the final PR body. Before writing the body file, scan the completed PR body once and
         confirm that no `AGENT_INSTRUCTION` comment line remains.

   c. **Pass body to the script** — prefer sending the final PR body to the script via stdin with `--body-stdin`, so the workflow does not depend on OS-specific temp-file paths. Only fall back to `--body-file <absolute_path>` when stdin is genuinely inconvenient, and do not hardcode `/tmp/...` on Windows.

   See [LLM_GUIDELINES.md](references/LLM_GUIDELINES.md) for the detailed generation rules,
   agent-only comment format, PR type selection, summary structure, and examples.

   Why strict preservation matters: templates contain keywords and structural patterns that CI bots
   and review bots parse to determine PR preconditions (e.g., `/kind feature`, checklist items,
   specific section headers). Altering or omitting these can cause bot checks to fail, blocking CI or merge.

5. **Create PR**:
   ```bash
   python <skill_dir>/scripts/create_pr.py --base <branch> --body-stdin
   ```
   Pipe the complete PR body to stdin for the command above. If stdin is not practical in the current environment, fall back to:
   ```bash
   python <skill_dir>/scripts/create_pr.py --base <branch> --body-file <absolute_temp_file>
   ```

## Parameter Mapping

| User parameter | Script parameter | Description |
|---|---|---|
| `--dst <branch>` | `--base <branch>` | Target branch for the PR |
| (LLM-generated) | `--body-stdin` | Read the complete PR body from stdin; preferred path |
| (LLM-generated fallback) | `--body-file <path>` | Path to file containing the complete PR body when stdin is impractical |
| — | `--template-status` | Show template file path for the current project |

## Project Templates

Templates are auto-discovered at runtime. Built-in templates currently include **akg** and
**mindspore** (also used by **docs**). User-created templates can override bundled ones for the
same project.

If you need to create or revise a template file, read
[TEMPLATE_AUTHORING.md](references/TEMPLATE_AUTHORING.md).

If you need discovery/runtime details such as directory priority and template resolution, read
[IMPLEMENTATION.md](references/IMPLEMENTATION.md). Keep both references out of the
user-facing reply unless the user explicitly asks for that level of detail.

## Error Handling

| Exit Code | Meaning | Action |
|---|---|---|
| 1 | Git or API error | Check stderr for details |
| 2 | Missing target branch | Ask user: "请问 PR 的目标分支是哪个？" |
| 3 | Empty or missing body file | LLM must generate a non-empty body and write it to a file |
| 4 | Credentials not configured | Read [CREDENTIALS_SETUP.md](references/CREDENTIALS_SETUP.md), give a brief local-only setup reminder, point the user to [README.md](README.md) for credential setup steps, remind them not to send secrets, and wait for confirmation before retrying |

See [API_REFERENCE.md](references/API_REFERENCE.md) for API documentation and auth troubleshooting.
