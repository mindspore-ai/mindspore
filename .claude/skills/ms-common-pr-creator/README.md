# ms-pr-creator

## gitcode 凭据配置

### 方式一：配置文件（推荐）

创建配置文件并写入以下 JSON：

| 平台 | 路径 |
|------|------|
| Linux / macOS | `~/.config/gitcode-skills/config.json` |
| Windows | `%USERPROFILE%\.config\gitcode-skills\config.json` |

```json
{
  "user_id": "your_gitcode_user_id",
  "access_token": "your_gitcode_access_token"
}
```

### 方式二：环境变量

```bash
export GITCODE_USER_ID=your_gitcode_user_id
export GITCODE_ACCESS_TOKEN=your_gitcode_access_token
```

Windows PowerShell：

```powershell
$env:GITCODE_USER_ID="your_gitcode_user_id"
$env:GITCODE_ACCESS_TOKEN="your_gitcode_access_token"
```

### 说明

- `user_id`：你的 GitCode 用户 ID，即个人主页 URL 中的 `user_id`（`https://gitcode.com/{user_id}`），不是页面展示的昵称
- `access_token`：GitCode Personal Access Token，可在 https://gitcode.com/setting/token-classic 创建
- 优先级：环境变量 > 配置文件。如果两者都配置了，使用环境变量的值
- 不要把 `access_token`、环境变量值、配置文件内容或相关截图发送给 AI agent，可能会泄露敏感信息

## 使用前置条件

调用 `ms-pr-creator` 前，请先确认：

- 已安装 Python 3（脚本仅依赖标准库）
- 当前目录是一个 Git 仓库
- 已配置 `upstream` 远端，例如：

```bash
git remote add upstream <upstream_repo_url>
```

- 当前分支已经完成 `git commit`
- 当前分支已经 `git push` 到当前使用者自己的 fork

注意：`ms-pr-creator` 只负责创建 PR，不负责 `git add`、`git commit`、`git push`。如有需要，可先由 Codex 或 Claude Code 协助完成这些 Git 操作，再调用该技能。

## 使用方式

可通过斜杠命令触发：

```bash
/ms-pr-creator --dst <目标分支>
```

或：

```bash
/ms-pr-creator
```

- 提供 `--dst` 时，技能会直接使用该目标分支
- 未提供 `--dst` 时，技能会先询问目标分支

也可使用自然语言触发，例如：

- `创建一个 gitcode pr，提交到 main 分支`
- `帮我创建一个 GitCode 的 PR，目标分支是 br_mfusion`
- `为当前分支发一个 gitcode pull request 到 main`

如果自然语言中已经包含目标分支，效果等价于提供了 `--dst`；如果未明确目标分支，AI 会继续追问。

### PR 模板

模板在运行时自动发现，按以下优先级查找：

1. **用户模板**：`~/.config/gitcode-skills/pr_templates/`（持久化，优先级更高）
2. **内置模板**：`ms-pr-creator/templates/`（随技能发布）

当前内置了以下项目的模板：

| 项目 | 模板文件 | 备注 |
|------|----------|------|
| `mindspore` | `templates/mindspore.md` | 同时作为未匹配项目的默认模板 |
| `akg` | `templates/akg.md` | |
| `docs` | `templates/mindspore.md` | 与 mindspore 共用模板 |

每个模板是一个 Markdown 文件，顶部包含 YAML frontmatter（声明适用的项目名），正文使用 `{{FILL: ...}}` 标记由 LLM 填写的区域。

#### 为新项目配置模板

如果当前项目没有匹配的模板，技能会提供两个选项：

1. 继续使用当前的默认模板
2. 在对话中提供模板内容，由 agent 协助生成项目专用模板

选择第二个选项时，agent 会引导你提供源材料（项目的 PR 模板、一个典型 PR 正文、或需要的章节列表），自动生成模板文件并保存到 `~/.config/gitcode-skills/pr_templates/<project>.md`，随后验证模板生效。

也可以手动创建模板文件：

1. 在 `~/.config/gitcode-skills/pr_templates/` 下新建 `.md` 文件
2. 添加 YAML frontmatter 声明项目名：

```yaml
---
projects:
  - new_project
---
```

3. 编写模板正文，需要 LLM 生成的部分使用 `{{FILL: 提示说明}}` 标记，其余内容会原样保留
4. 保存后，重新调用 `ms-pr-creator` 即可使用新模板
