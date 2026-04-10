# GitCode PR Template Authoring Guide

Use this file when the skill needs to create or revise a PR template file, especially for the
"unconfigured project -> configure a template in chat" workflow.

This guide covers template authoring mechanics. For template discovery order, runtime path
resolution, and `--template-status`, read [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Goal

Turn one of these user-provided inputs into an executable template file:

- the project's current PR template
- a representative high-quality PR body from that project
- a list of required sections and formatting constraints

The output file must be usable by the skill on future runs without manual cleanup.

## In-Chat Configuration Workflow

Use this workflow when an unconfigured project is currently falling back to a bundled default
template and the user chooses "let the agent configure a project-specific template now".

1. Collect source material from the user:
   - the project's current PR template, or
   - a representative high-quality PR body from that project, or
   - a clear list of required sections and formatting constraints
2. Convert that source material into an executable template using the rules below.
3. Save the file under `~/.config/gitcode-skills/pr_templates/<project>.md`.
4. Tell the user the exact saved path so they can inspect or edit it later.
5. Re-run `python <skill_dir>/scripts/create_pr.py --template-status`.
6. Confirm that the project now resolves to the newly saved template.

Manual editing remains a backup path for advanced users, but the in-chat flow above should be
the default recommendation.

## Save Location

Prefer the persistent user template directory:

- `~/.config/gitcode-skills/pr_templates/`

On Windows this resolves under the user's home directory, for example:

- `%USERPROFILE%\.config\gitcode-skills\pr_templates\`

Tell the user the exact saved file path after writing the file.

## Template Anatomy

Each template file has two parts:

1. YAML frontmatter
2. Template body

Minimal structure:

```md
---
projects:
  - newcompiler
---
**What type of PR is this?**
/kind {{FILL: Select one label based on the dominant change type: bug / task / feature / refactor / clean_code}}

**What does this PR do / why do we need it**:
{{FILL: Summarize the purpose, motivation, main design, and key changes in Chinese}}
```

## YAML Frontmatter

The frontmatter is metadata for template discovery. It is not part of the final PR body.

Required field:

- `projects`: repo names that should resolve to this template

Optional field:

- `default: true`: marks the fallback template when no project matches

Example:

```yaml
---
projects:
  - mindspore
  - docs
default: true
---
```

Guidelines:

- Use the upstream repo name, not the owner name.
- Add aliases only when they are genuinely intended to share one template.
- Do not put prose instructions in frontmatter.

### About `default: true`

Treat `default: true` as an advanced field.

What it does:

- it marks a template as the fallback when no project-specific template matches
- because user templates are scanned before bundled templates, a user template with
  `default: true` can override the skill's bundled fallback template globally

Default recommendation:

- for normal project-specific templates, do **not** set `default: true`
- omit the `default` field entirely unless the user explicitly wants to change the global
  fallback template for unmatched projects

Only use `default: true` when the user clearly asks for behavior like:

- "make this the default template for repos that do not have a dedicated template yet"
- "replace the bundled fallback template"

## `{{FILL: ...}}` Markers

`{{FILL: ...}}` defines a fillable zone. The instruction inside the marker tells the agent what
to generate there later.

Default language rule:

- unless the user explicitly asks for another language, write `{{FILL: ...}}` guidance so the
  generated PR content will be in Chinese
- this default still applies even if the template's fixed headings happen to be in English
- if the user explicitly wants English or another language, state that language clearly in the
  relevant `{{FILL: ...}}` guidance

Use `{{FILL: ...}}` when:

- the content should be generated from the semantics of the current PR
- the agent can reasonably infer or summarize the content from commit history, diffs, file changes,
  or project context
- the template contains a fixed frame with one variable slot that the agent is expected to choose
  or fill for each PR

Do not use `{{FILL: ...}}` when:

- the text must remain verbatim for bots or reviewers
- the value must be provided or confirmed by a human rather than inferred by the agent
- the field refers to external facts that the agent should not guess, such as the exact linked
  issue number or issue URL the author intends to close

This is a responsibility boundary, not a formatting rule:

- a checklist can still contain `{{FILL: ...}}` if the project genuinely expects the agent to
  generate part of it
- a command-like line can also contain `{{FILL: ...}}` if the variable fragment is something
  the agent is supposed to determine from the PR semantics
- a heading or section label can remain fixed while the body below it uses `{{FILL: ...}}`

Good:

```md
/kind {{FILL: Select one label based on the dominant change type: bug / task / feature / refactor / clean_code}}
```

Also good:

```md
**Special notes for your reviewers**:
{{FILL: List reviewer focus points only when the current PR has notable risks, compatibility concerns, or special review areas; otherwise write `无`}}
```

Do not convert human-owned fields like this into LLM-filled markers:

```md
**Which issue(s) this PR fixes**:
Fixes #
```

Marker-writing guidelines:

- Be explicit about what the generated text must contain.
- State output language when it matters.
- Mention preservation rules only when the local context needs them.
- Prefer one marker per logical variable zone.
- Avoid giant markers that ask the model to rewrite the whole document.

## `<!-- AGENT_INSTRUCTION ... -->` Comments

Use these comments sparingly. They are for generation-time guidance only and must be removed
from the final PR body after generation.

Use them when a nearby section needs an extra local rule about generation responsibility or
preservation, and that rule is easy to violate unless stated in place.

Good uses include cases where the comment tells the agent:

- this nearby content is human-owned and must not be auto-filled
- this nearby content is fixed and must be preserved verbatim
- this nearby section should stay empty or unchanged unless a specific condition is met

Avoid `AGENT_INSTRUCTION` when:

- the rule already belongs in `SKILL.md` or `LLM_GUIDELINES.md`
- the comment merely repeats something the template already makes obvious
- the comment tries to control the entire document instead of one fragile local area

Example:

```md
<!-- AGENT_INSTRUCTION - Keep all checklist boxes below unchecked. Do not change [ ] to [x]. Remove this comment from the final PR body. -->
**Self-checklist**:
+ - [ ] Design
+ - [ ] Test
```

Do not use `AGENT_INSTRUCTION` for general workflow rules that already belong in `SKILL.md` or
`LLM_GUIDELINES.md`.

## Fixed Structure vs Fillable Structure

The key authoring decision is where to draw the line between fixed template structure and
generated content. The deciding question is not "what kind of Markdown block is this?" but
"who should own the value here?"

Keep fixed:

- content whose wording or formatting must stay stable for bots, reviewers, or project policy
- labels, comments, and prompts that define the document structure
- fields whose final value should come from the PR author or another human decision-maker
- any text where the agent should preserve the skeleton and only fill adjacent agent-owned slots

Make fillable:

- content that should be inferred, summarized, selected, or phrased from the current PR itself
- summaries, rationale, technical approach, grouped change descriptions, and test-plan details
- variable fragments inside otherwise fixed lines when that fragment is agent-owned
- reviewer notes or risk notes when they depend on the actual change set

If uncertain, bias toward preserving existing project structure and only replacing the truly
variable, agent-owned content with markers.

## Converting a Real PR Body Into a Template

When the user gives a good example PR body:

1. Keep the stable skeleton.
2. Ask, for each span of content: is this agent-owned or human-owned?
3. Replace only the agent-owned variable spans with `{{FILL: ...}}`.
4. Keep human-owned or externally anchored fields fixed, even if they vary across real PRs.
5. Add frontmatter for the target repo.
6. Add a local `AGENT_INSTRUCTION` comment only if a nearby responsibility or preservation rule is easy to break.

Example conversion:

Input PR body:

```md
**What type of PR is this?**
/kind feature

**What does this PR do / why do we need it**:
Add a new conv2d tiling strategy and register it in the DVM tiling manager.

**Special notes for your reviewers**:
Please focus on cache-size assumptions.
```

Template version:

```md
---
projects:
  - newcompiler
---
**What type of PR is this?**
/kind {{FILL: Select one label based on the dominant change type: bug / task / feature / refactor / clean_code}}

**What does this PR do / why do we need it**:
{{FILL: Describe the goal, motivation, main implementation approach, and key changed modules in Chinese}}

**Special notes for your reviewers**:
{{FILL: List reviewer focus points only if there are notable risks, compatibility concerns, or special review areas; otherwise write `无`}}
```

## Quality Checklist Before Saving

Before writing the template file, confirm:

- frontmatter exists and names the right project
- the body keeps the project's fixed structure intact
- each `{{FILL: ...}}` zone corresponds to agent-owned content, not human-owned content
- variable zones were converted into `{{FILL: ...}}` markers with useful instructions
- any human-owned fields remain fixed rather than being delegated to the agent
- any `AGENT_INSTRUCTION` comments are local, necessary, and clearly removable
- the template is not just a raw PR body pasted without generalization
- the file is saved under `~/.config/gitcode-skills/pr_templates/`

## After Saving

After writing the file:

1. Tell the user the exact saved path.
2. Re-run `python <skill_dir>/scripts/create_pr.py --template-status`.
3. Confirm that the project now resolves to the new template.
