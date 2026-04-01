# GitCode 凭据首次配置引导

当脚本以 exit code 4 退出时，说明用户尚未配置 GitCode 凭据。此时只做**本地配置引导**，不要让 agent 接触任何 secret。

本文档用于约束 agent 的安全边界和回复方式。用户可见的本地凭据配置步骤以 [../README.md](../README.md) 为准；agent 不要求在聊天中复述整套配置细节。

## Agent Safety Rules

- 明确提醒用户：**不要把 Access Token、环境变量值、配置文件内容或截图发给 agent**，这会带来敏感信息泄露风险。
- agent **不得**要求用户粘贴 `access_token`、`GITCODE_ACCESS_TOKEN`、配置文件全文，或任何包含 secret 的命令输出。
- agent **不得**直接读取环境变量或共享配置文件中的凭据值。
- agent **不得**运行会打印 secret 的命令，例如 `echo $GITCODE_ACCESS_TOKEN`、`Get-ChildItem Env:`、`env`、`printenv`，或 `cat/Get-Content` 凭据文件。
- 如果用户误发了 token，提醒用户立即在 GitCode 撤销并重新生成，不要在后续回复中复述该 token。

## Safe Setup Flow

1. **向用户解释情况**：告知用户该技能需要 GitCode 凭据才能调用 API，只需在本机配置一次，所有 `gitcode-skills` 技能共享。
2. **先做安全提醒**：明确告诉用户不要把 token、环境变量值、配置文件内容或截图发送给 agent。
3. **给最小必要说明**：只需告诉用户在本地完成凭据配置或检查，可使用共享配置文件或环境变量，不要在聊天里回传真实值。
4. **引导用户查看 README.md**：明确告诉用户去看本 skill 的 `README.md` 完成本地配置，不要要求其回传真实值；如用户仍需要最小提示，再补充下面这些要点：
   - 共享配置文件的路径
   - JSON 配置样例
   - `user_id` 的含义和获取方式
   - `access_token` 的创建入口
   - 环境变量配置方式
5. **等待用户确认**：只让用户回复“已完成配置”之类的确认，不要要求其粘贴任何 secret 或配置内容。
6. **重新执行**原始命令。

## Fallback Summary

如果用户需要更具体的本地配置提示，再在聊天中补充下面这些最小关键信息：

- 配置文件路径：
  - Linux / macOS：`~/.config/gitcode-skills/config.json`
  - Windows：`%USERPROFILE%\.config\gitcode-skills\config.json`
- JSON 字段示例：
  ```json
  {
    "user_id": "<你的 GitCode user_id>",
    "access_token": "<你的 Access Token>"
  }
  ```
- `user_id` 是个人主页 URL `https://gitcode.com/{user_id}` 中的 `user_id`
- `access_token` 可在 `https://gitcode.com/setting/token-classic` 创建
