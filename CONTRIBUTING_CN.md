# MindSpore贡献指南

[View English](./CONTRIBUTING.md)

<!-- TOC -->

- [MindSpore贡献指南](#mindspore贡献指南)
    - [贡献者许可协议](#贡献者许可协议)
    - [快速入门](#快速入门)
    - [贡献流程](#贡献流程)
        - [代码风格](#代码风格)
        - [Fork-Pull开发模型](#fork-pull开发模型)
        - [报告Issue](#报告issue)
        - [提交PR](#提交pr)
        - [本地代码自检](#本地代码自检)

<!-- /TOC -->

## 贡献者许可协议

向MindSpore社区提交代码之前，您需要签署《贡献者许可协议（CLA）》。

个人贡献者请参见[ICLA在线文件](https://www.mindspore.cn/icla)。

## 快速入门

- 在[Github](https://github.com/mindspore-ai/mindspore)或[Gitee](https://gitee.com/mindspore/mindspore)上fork代码仓。
- 参见[README_CN.md](README_CN.md)和[安装页面](https://www.mindspore.cn/install)了解项目信息和构建说明。

## 贡献流程

### 代码风格

请遵循此风格，以便MindSpore审查、维护和开发。

- 编码指南

    MindSpore社区使用[Python PEP 8 编码风格](https://pep8.org/)和[谷歌C++编码风格](http://google.github.io/styleguide/cppguide.html)。建议在IDE中安装以下插件，用于检查代码格式：[CppLint](https://github.com/cpplint/cpplint)、[CppCheck](http://cppcheck.sourceforge.net)、[CMakeLint](https://github.com/cmake-lint/cmake-lint)、[CodeSpell](https://github.com/codespell-project/codespell)、[Lizard](http://www.lizard.ws)、[ShellCheck](https://github.com/koalaman/shellcheck)和[PyLint](https://pylint.org)。

- 单元测试指南

    MindSpore社区使用Python单元测试框架[pytest](http://www.pytest.org/en/latest/)和C++单元测试框架[Google Test Primer](https://github.com/google/googletest/blob/master/docs/primer.md)。注释名称需反映测试用例的设计意图。

- 重构指南

    我们鼓励开发人员重构我们的代码，以消除[代码坏味道](https://zh.wikipedia.org/wiki/%E4%BB%A3%E7%A0%81%E5%BC%82%E5%91%B3)。所有代码都要符合编码风格和测试风格，重构代码也不例外。无注释的代码行（nloc）的[Lizard](http://www.lizard.ws)阈值为100，圈复杂度（cnc）的阈值为20。当收到Lizard警告时，必须重构要合并的代码。

- 文档指南

    我们使用MarkdownLint来检查Markdown文档格式。MindSpore CI基于默认配置修改了以下规则。
    - MD007（无序列表缩进）：参数**indent**设置为**4**，表示无序列表中的所有内容都需要缩进4个空格。
    - MD009（行尾空格）：参数**br_spaces**设置为**2**，表示行尾可以有0或2个空格。
    - MD029（有序列表的序列号）：参数**style**设置为**ordered**，表示升序。

    有关详细信息，请参见[规则](https://github.com/markdownlint/markdownlint/blob/master/docs/RULES.md)。

### Fork-Pull开发模型

- Fork MindSpore代码仓

    在提交代码至MindSpore项目之前，请确保已fork此项目到您自己的代码仓。MindSpore代码仓和您自己的代码仓之间可能会并行开发，请注意它们之间的一致性。

- 克隆远程代码仓

    如果您想将代码下载到本地计算机，最好使用git方法：

    ```shell
    # 在GitHub上：
    git clone https://github.com/{insert_your_forked_repo}/mindspore.git
    git remote add upstream https://github.com/mindspore-ai/mindspore.git
    # 在Gitee上：
    git clone https://gitee.com/{insert_your_forked_repo}/mindspore.git
    git remote add upstream https://gitee.com/mindspore/mindspore.git
    ```

- 本地开发代码。

    为避免分支不一致，建议切换到新分支：

    ```shell
    git checkout -b {新分支名称} origin/master
    ```

    以master分支为例，如果MindSpore需要创建版本分支和下游开发分支，请先修复上游的bug，
    再更改代码。

- 将代码推送到远程代码仓。

    更新代码后，以正式的方式推送更新：

    ```shell
    git add .
    git status # 查看更新状态。
    git commit -m "你的commit标题"
    git commit -s --amend # 添加commit的具体描述。
    git push origin {新分支名称}
    ```

- 将请求拉取到MindSpore代码仓。

    在最后一步中，您需要在新分支和MindSpore主分支之间拉取比较请求。完成拉取请求后，Jenkins CI将自动设置，进行构建测试。拉取请求应该尽快合并到上游master分支中，以降低合并的风险。

### 报告Issue

发现问题后，建议以报告issue的方式为项目作出贡献。错误报告应尽量书写规范，内容详尽，感谢您对项目作出的贡献。

报告issue时，请参考以下格式：

- 说明您使用的环境版本（MindSpore、OS、Python等）。
- 说明是错误报告还是功能需求。
- 说明issue类型，添加标签可以在issue板上突出显示该issue。
- 问题是什么？
- 期望如何处理？
- 如何复现？（尽可能精确具体地描述）
- 给审核员的特别说明。

**Issue咨询：**

- **解决issue时，请先评论**，告知他人由您来负责解决该issue。
- **对于长时间未关闭的issue**，建议贡献者在解决该issue之前进行预先检查。
- **如您自行解决了自己报告的issue**，仍需在关闭该issue之前告知他人。
- **如需issue快速响应**，可为issue添加标签。标签详情，参见[标签列表](https://gitee.com/mindspore/community/blob/master/sigs/dx/docs/labels.md)。

### 提交PR

- 在[GitHub](https://github.com/mindspore-ai/mindspore/issues)或[Gitee](https://gitee.com/mindspore/mindspore/issues)上通过issue提出您的想法。
- 如果是需要大量设计细节的新功能，还应提交设计方案。
- 经issue讨论和设计方案评审达成共识后，在已fork的代码仓开发，并提交PR。
- 任何PR至少需要位2位审批人的LGTM标签。请注意，审批人不允许在自己的PR上添加LGTM标签。
- 经充分讨论后，根据讨论的结果合并、放弃或拒绝PR。

**PR咨询：**

- 避免不相关的更改。
- 确保您的commit历史记录有序。
- 确保您的分支与主分支始终一致。
- 用于修复错误的PR中，确保已关联所有相关问题。

### 本地代码自检

在开发过程中，建议使用pre-push功能进行本地代码自检，可以在本地进行类似CI门禁上Code Check阶段的代码扫描，提高上库时跑门禁的成功率。使用方法请参见[pre-push快速指引](scripts/pre_commit/README_CN.md)。
