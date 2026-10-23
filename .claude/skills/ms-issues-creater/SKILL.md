---
name: ms-issues-creater
description: Create issues for cann.
---

# MindSpore Pylint Issues Fix Guide

## ARGUMENTS
- issues_files: args firstly, one file or directory of issues
- access_token: args second token of gitcode's user
e
## API for creating issue
- Use api https://api.gitcode.com/api/v5/repos/mindspore/issues to create issues
- body请求体结构如下：
    ```
    {
      'repo': 'org-issues',
      'title': 从issues_files文件中读取完整标题,
      'assignee': 从issues_files文件中读取组件责任人,
      'access_token': access_token,
      'issue_type': 'Bug-Report',
      'body': 将issues_files文件内容作为issue描述,
      'custom_fields': [
        {
          '关联分支': 从issues_files文件中读取关联分支,
          '关联组件': 从issues_files文件中读取关联组件,
          '问题后端类型': 从issues_files文件中读取问题后端类型
        }
      ]
    }
    ```
## Create issues
- 所有动作涉及到编码格式的统一使用utf-8。
- 读取issues_files中以issue为开头命名的md文件
- 使用创建issue的API，在仓库org-issues中，根据读取到的issue[N].md文件, 使用repo仓库中提供的issue模板, 创建一个Bug-Report类型的issue。
- 避免无效或重复创建issue
  - 不执行test_开头的脚本创建测试类issue
  - 不创建同名issue
  - 不创建具有相同'问题描述'的issue
  - 不创建与现有问题无关的issue


## Quick Start
```bash
# base on issues/issues.md which will be got by /ms-log-analysis, create issues.
/ms-issues-creater issues_files=issues/issue.md access_token=<GITCODE_TOKEN>
```
