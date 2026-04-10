# GitCode PR Implementation Details

Auto-detection logic and implementation details for the PR creation script.

This file explains runtime behavior such as template discovery order and project resolution.
For the mechanics of writing a new template file, read [TEMPLATE_AUTHORING.md](TEMPLATE_AUTHORING.md).

## Project Template System

### Template Directories

Templates are discovered from two directories, scanned in priority order:

| Priority | Directory | Purpose | Survives skill upgrade? |
|----------|-----------|---------|------------------------|
| 1 (high) | `~/.config/gitcode-skills/pr_templates/` | User / agent added templates | Yes |
| 2 (low) | `<skill_dir>/templates/` | Bundled defaults | No (overwritten on upgrade) |

If the same project name is declared in both directories, the **user directory wins**.

```
~/.config/gitcode-skills/pr_templates/  ← user templates (persistent)
├── my_project.md
└── ...

<skill_dir>/templates/                  ← bundled templates (shipped with skill)
├── akg.md
└── mindspore.md
```

Each template file has two parts:
1. **YAML frontmatter** — declares which projects this template applies to
2. **Template body** — the actual PR body structure with `{{FILL: instruction}}` markers

### Template Frontmatter

Each template declares its scope via YAML frontmatter at the top of the file:

```yaml
---
projects:
  - mindspore
  - docs
default: true
---
(template body follows)
```

| Field | Required | Description |
|-------|----------|-------------|
| `projects` | Yes | List of repository names this template applies to |
| `default` | No | Set to `true` to use this template when no match is found |

The frontmatter is metadata for the script only — the LLM must **not** include it in the PR body.

### Auto-Discovery

The script scans both template directories at runtime, reads each `.md` file's YAML frontmatter,
and builds the project-to-template mapping dynamically. No hardcoded mapping exists in the script.

**Adding a new template requires zero code changes** — just create a `.md` file with the
right frontmatter in either directory.

### Template Selection Logic

The `get_template_for_project()` function determines which template to use:

1. **Exact match**: If repo name matches a project listed in any template's frontmatter
2. **Partial match**: If repo name contains a project name (e.g., "mindspore-akg" matches "akg")
3. **Default**: Template marked `default: true` in its frontmatter

Within each step, user templates take precedence over bundled templates.

### `{{FILL:...}}` Marker Syntax

Templates contain two kinds of content:

| Content type | Description | LLM action |
|---|---|---|
| Fixed structure | Everything outside `{{FILL:...}}` markers | Copy verbatim — do not modify |
| Fillable zones | `{{FILL: instruction}}` markers | Replace with generated content |

Example from a template:

```markdown
**What type of PR is this?**
/kind {{FILL: 根据代码变更类型选择一个: feature / bug / task / refactor / clean_code}}

**What does this PR do / why do we need it**:
{{FILL: 用中文描述 PR 目的、主要方案和关键修改点（参见 LLM_GUIDELINES.md 的内容生成指南）}}
```

After LLM fills in the markers, the output would look like:

```markdown
**What type of PR is this?**
/kind feature

**What does this PR do / why do we need it**:
### PR目的
为 Conv2D 算子实现多级 tiling 策略...
```

The instruction text inside each marker tells the LLM what to generate for that specific zone.
Different templates can have entirely different sets of markers — the LLM adapts to whatever
the template asks for.

### Adding a New Project Template

Authoring rules are intentionally centralized in [TEMPLATE_AUTHORING.md](TEMPLATE_AUTHORING.md).

At the runtime level, the only important facts are:

1. project-specific templates should normally be written under `~/.config/gitcode-skills/pr_templates/`
2. user templates override bundled templates automatically when they declare the same project name
3. template discovery is dynamic; no code changes are required after the file is added

## Auto-Detection Logic

### Repository Name

```bash
git remote get-url upstream
```

Extracts the repository name from upstream remote URL using regex.

**Example**: `https://gitcode.com/mindspore/akg.git` → `akg`

### PR Title

```bash
git log upstream/<base>..HEAD --reverse --format=%s --max-count=1
```

Extracts the **earliest** commit's subject line, used as the PR title.

**Example**: `refactor(muse-to-dvm): unify attribute naming`

### Current Branch (Head)

```bash
git branch --show-current
```

Gets the current git branch name, then formats as `{user_id}:<branch>`.

**Example**: `feat-muse-to-dvm` → `{user_id}:feat-muse-to-dvm`

### Base Branch

Must be specified by user via `--base` parameter. No auto-detection.

## Handling Unconfigured Projects

When `get_template_for_project()` returns `is_configured=False`, runtime behavior is:

1. The script still returns a usable template path by falling back to the default template
2. The skill should surface that resolved default template path to the user
3. The skill may either continue with the default template or create a project-specific user template
4. If creating a project-specific template, use the persistent user template directory and follow [TEMPLATE_AUTHORING.md](TEMPLATE_AUTHORING.md)

## Edge Cases

### No Upstream Configured

**Detection**: `git remote get-url upstream` returns empty

**Solution**:
1. Display error message
2. Require user to configure upstream remote

### Empty Commit Message

**Detection**: `git log` returns empty

**Solution**: Script exits with error, prompts user to ensure commit exists

### Multiple Commits on Branch

**Behavior**: Uses **earliest** commit's subject line for PR title

### Authentication Failure

**Detection**: HTTP 401 or 403 response

**Solution**:
1. Display error message
2. Give a brief local-only re-check reminder and point the user to [../README.md](../README.md) for credential setup steps.
3. Use [CREDENTIALS_SETUP.md](CREDENTIALS_SETUP.md) only as the internal safety workflow reference.

### Unconfigured Project

**Detection**: `get_template_for_project()` returns `is_configured=False`

**Solution**:
1. Present the default-vs-custom template choice to the user
2. If custom, save the resulting template under `~/.config/gitcode-skills/pr_templates/`
3. Re-run template resolution after the file is added

## Script Structure

```python
# Main workflow
1. get_upstream_info()        # Detect repository owner and name
2. get_current_branch()       # Get feature branch
3. get_template_for_project() # Resolve template file path (--template-status only)
4. get_pr_title()             # Get title from earliest commit subject
5. read body from --body-file # Complete body provided by LLM
6. create_gitcode_pr()        # POST to API
7. Display result             # Show PR URL
```

## Dependencies

- Python 3.6+
- Standard library only (no external packages)
  - `subprocess` - Git commands
  - `urllib.request` - HTTP requests
  - `json` - Request/response parsing
