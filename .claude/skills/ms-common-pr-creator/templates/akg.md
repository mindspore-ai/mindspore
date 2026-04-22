---
projects:
  - akg
---
<!-- AGENT_INSTRUCTION — 你的输出从下一行开始。上方 YAML frontmatter 是模板元数据，不是 PR 模板正文的一部分；最终 PR 正文中不要包含 frontmatter，也不要包含这条注释。 -->
<!--  Thanks for sending a pull request!  Here are some tips for you:

If this is your first time, please read our contributor guidelines: https://github.com/mindspore-ai/mindspore/blob/master/CONTRIBUTING.md
-->

**What type of PR is this?**
`/kind {{FILL: 根据代码变更类型选择一个: feature / bug / task / refactor / clean_code}}`


**What does this PR do / why do we need it**:
{{FILL: 用中文描述 PR 目的、主要方案和关键修改点（参见 LLM_GUIDELINES.md 的内容生成指南）}}

{{FILL: 用中文描述测试计划和验证结果（如本次变更未涉及测试，描述如何验证改动的正确性）}}

**Which issue(s) this PR fixes**:
<!--
*Automatically closes linked issue when PR is merged.
Usage: `Fixes #<issue number>`, or `Fixes (paste link of issue)`.
-->
Fixes #

**Special notes for your reviewers**:
{{FILL: 根据变更文件分析审查注意事项（如是否导致无法前向兼容、是否涉及依赖的三方库变更等）。如无特殊事项写"无"}}
