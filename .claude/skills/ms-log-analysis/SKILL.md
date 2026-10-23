---
name: ms-log-analysis
description: Analyse log file
---

# MindSpore Pylint Issues Fix Guide

## ARGUMENTS
- args firstly is log which may contains error info.

## Create issues files
- 所有动作涉及到编码格式的统一使用utf-8。
- 分析给定的日志文件, 分析失败或错误信息。
- 分析问题主从关系，保留主体问题。
- 将问题分类。
- 在项目根目录，创建issues目录
- 在issues目录下每一类问题创建一个对应的issue.md文件，文件名格式issue+编号,文件以下述信息为头， 避免同一类型问题重复创建issue文件
'''
  ### Checklist  
  - [x] 1. 我已经搜索过相关问题，但没有得到预期的帮助.
  - [x] 2. 最新版本中该错误尚未修复.
  - [x] 3. 请注意，如果您提交的Bug描述缺少相应的环境信息和最小可复现的demo，我们将很难复现和解决该问题，从而降低收到反馈的可能性，甚至该问题将被关闭.
  ### 🐞 问题详细描述
'''
- issue.md 文件中明确issue的标题，标题格式分以下几种情况，根据实际情况选用合适的标题格式
  - 基础标题："标题 [BUG]:"+问题类型
  - 涉及CANN问题标题："标题 [BUG]:CANN"+问题类型
  - 涉及测试用例的标题："标题 [BUG]:"+涉及到的测试用例名，若多个用例需要用"、"隔开+问题类型
  - 涉及CANN+测试用例的标题："标题 [BUG]:CANN "+涉及到的测试用例名，若多个用例需要用"、"隔开+问题类型
- 根据问题涉及的代码，分析文件CODEOWNERS，选择一位最接近问题代码的人作为组件责任人
- issue.md 文件中包含的其他信息
  - 问题描述
  - 错误信息
  - 受影响范围
  - 根本原因
  - 修复建议
  - 环境信息
  - 关联分支
  - 关联组件
  - 问题后端类型
  - 组件责任人


## Quick Start
```bash
/ms-log-analysis <log file or consoleText from jenkins>
# exp: /ms-log-analysis https://build.mindspore.cn/job/MindSpore_Atomgit_Gate_Debug/1866/consoleText
#      /ms-log-analysis https://build.mindspore.cn/blue/rest/organizations/jenkins/pipelines/MindSpore_Atomgit_Gate_Debug/runs/1876/nodes/145/log/?start=0 
# file of issue will be created in <project>/issues/
```
