---
name: ms-write-skills
description: Create a new skill for the MindSpore project. Use when adding a new skill, writing a SKILL.md, choosing a skill name, deciding which module a skill belongs to, or registering a skill in CLAUDE.md. Triggers on phrases like "add skill", "create skill", "new skill", "write a skill", "add a ms- skill", "skill naming", "register skill".
---

# MindSpore Skill Authoring Guide

This skill covers the conventions and rules for creating new skills in the MindSpore project.

## Naming Convention

### Directory name: `ms-{module}-{function}`

| Part | Rules | Examples |
|------|-------|---------|
| `ms-` | Fixed prefix for all MindSpore skills | — |
| `{module}` | Functional area (see table below) | `common`, `api`, `jit` |
| `{function}` | What the skill does, kebab-case | `build`, `pr-review`, `write-skills` |

**Module names:**

| Module | Scope | Example skills |
|--------|-------|----------------|
| `common` | Cross-cutting concerns not tied to one subsystem | `ms-common-build`, `ms-common-debug`, `ms-common-document`, `ms-common-pr-review` |
| `api` | Python user-facing API (`mindspore/python/mindspore/`) | `ms-api-develop` |
| `operations` | Operator definition, kernels, infer | `ms-operations-develop` |
| `jit` | JIT compiler, IR passes, fusion | `ms-jit-fusion-pass` |
| `nn` | Neural network layers (`mindspore/nn/`) | (future) |
| `backend` | Device backends, runtime | (future) |

Use `common` when the skill applies broadly across the whole codebase (build system, docs, debugging, code review). Use a specific module when the skill is tightly scoped to one subsystem.

### Frontmatter `name` field

Use a **shortened** form: `ms-{abbreviated-function}` (not the full directory name). This is what appears in the skills list.

```
Directory:          ms-common-pr-review
Frontmatter name:   ms-pr-review

Directory:          ms-operations-develop
Frontmatter name:   ms-operations
```

Keep it short and unambiguous — 2–4 words.

---

## File Structure

```
.claude/skills/{skill-dir-name}/
├── SKILL.md              ← required
└── references/           ← optional, for large reference docs
    └── {topic}.md
```

- `SKILL.md` must stay under **500 lines**. Move large reference tables, full templates, or backend-specific details into `references/` files and reference them from `SKILL.md`.
- Use `references/` subdirectory names that describe the content (e.g., `references/cpu-kernel.md`, `references/yaml-schema.md`).

---

## SKILL.md Frontmatter

Required fields:

```yaml
---
name: ms-{function}
description: <what it does> — <when to trigger>. Triggers on phrases like "...", "...", "...".
---
```

**Description rules:**

1. Start with a one-line summary of what the skill does.
2. Follow with "Use when …" listing concrete scenarios.
3. End with `Triggers on phrases like "...", "..."` — include the actual words a developer would type. Cover both formal ("add a new operator") and casual ("add op", "new kernel") phrasings.
4. Be specific about scope: mention backend names (Ascend/GPU/CPU), subsystem names, or file types where relevant.

**Example (good):**
```
Add a new operator or modify an existing operator in MindSpore. Use when adding a new op,
changing op signatures, adding new backend support (Ascend/CPU/GPU), migrating py_method
dispatch to pyboost, or modifying infer/kernel logic. Triggers on phrases like "add op",
"new operator", "implement op", "add CPU support", "migrate to pyboost", "add GPU kernel",
"register operator".
```

---

## SKILL.md Body Content

### Required sections

1. **Overview paragraph** — one short paragraph explaining what this skill covers.
2. **File map** — a code block listing the key files this skill touches, using real MindSpore paths.
3. **Step-by-step instructions** — numbered steps with concrete code/YAML templates.
4. **Checklist** — a `- [ ]` checklist at the end so the developer can verify completeness.

### Content rules

- Use **imperative form**: "Create", "Add", "Register", not "You should create".
- Reference **real file paths** from the MindSpore codebase, not generic placeholders. Look at similar existing files first.
- Include **code templates** with placeholder tokens like `{op_name}`, `{OpName}`, `{OP_NAME_UPPER}` so the developer can fill them in mechanically.
- Explain the **why** behind non-obvious requirements (e.g., why `pyboost` is preferred over `py_method`).
- When a step has multiple valid choices, explain the tradeoffs (e.g., `pyboost` vs `py_method` dispatch).

### What NOT to include

- Don't copy large amounts of content already in `CLAUDE.md` — link to the relevant CLAUDE.md section instead.
- Don't include debugging or troubleshooting steps — those belong in `ms-common-debug`.
- Don't duplicate content from other skills. If a step requires running code generation, reference `ms-common-build` rather than repeating the steps.

---

## Registration

After creating the skill directory and `SKILL.md`, register it in **two places**:

### 1. `CLAUDE.md` — Skills Available section

Add a line to the `## Skills Available` section (alphabetically ordered within `common`/specific groups):

```markdown
- `ms-{module}-{function}` — One-line description of what it does
```

### 2. System prompt list (automatic)

Skills under `.claude/skills/` are automatically discovered and appear in the available skills list. No extra config needed.

---

## Checklist

- [ ] Directory named `ms-{module}-{function}` under `.claude/skills/`
- [ ] `SKILL.md` with `name` and `description` frontmatter fields
- [ ] `name` is the short form `ms-{function}` (not the full directory name)
- [ ] Description includes: what it does + "Use when …" + "Triggers on phrases like …"
- [ ] Body has: overview, file map with real paths, step-by-step instructions, checklist
- [ ] Registered in `CLAUDE.md` under "Skills Available"
- [ ] Large reference content moved to `references/` if `SKILL.md` exceeds 500 lines
