<!--  Thanks for sending a pull request!  Here are some tips for you:

1) If this is your first time, please read our contributor guidelines: https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md

2) If you want to contribute your code but don't know who will review and merge, please add label `mindspore-assistant` to the pull request, we will find and do it as soon as possible.
-->

**What type of PR is this?**
> Uncomment only one ` /kind <>` line, hit enter to put that in a new line, and remove leading whitespaces from that line:
>
> /kind bug
> /kind task
> /kind feature


**What does this PR do / why do we need it**:


**Which issue(s) this PR fixes**:
<!-- 
*Automatically closes linked issue when PR is merged.
Usage: `Fixes #<issue number>`, or `Fixes (paste link of issue)`.
-->
Fixes #

**Code review checklist [【代码检视checklist说明】](https://gitee.com/mindspore/community/blob/master/security/code_review_checklist_mechanism.md)**:

- **典型安全编码问题 [【违反安全编码案例】](https://gitee.com/mindspore/community/blob/master/security/security_coding_violation_cases.md)**
    - [ ] 是否进行空指针校验
    - [ ] 是否进行返回值校验 (禁止使用void屏蔽安全函数、自研函数返回值，C++标准库函数确认无问题可以屏蔽)
    - [ ] 是否正确释放new/malloc申请的内存
- **性能分析 (如果涉及某个子项，请概述设计思想/修改内容)**
    - [ ] 是否修改热点***函数 / 算法 / 算子***
    - [ ] 是否考虑并发场景
    - [ ] 是否考虑通信场景
+ - [ ] **是否符合编码规范 [【编码规范】](https://gitee.com/mindspore/community/blob/master/security/coding_guild_cpp_zh_cn.md)**
+ - [ ] **是否遵守 ***SOLID原则 / 迪米特法则*****
+ - [ ] **是否涉及模块/特性间交互 (若涉及请概述实现思路)**
+ - [ ] **是否具备UT测试用例看护 && 测试用例为有效用例 (若新特性无测试用例看护请说明原因)**
+ - [ ] **是否正确加载、释放秘钥**
- **错误处理与记录**
    - [ ] 是否充分考虑接口的异常场景
    - [ ] 是否正确记录错误信息

**Special notes for your reviewers**:
<!-- + - [ ] **是否涉及文档（安装、教程、设计、参考、API、迁移指南、FAQ等）修改** -->
<!-- + - [ ] **是否导致无法前向兼容** -->
<!-- + - [ ] **是否为对外接口变更** -->
<!-- + - [ ] **是否涉及依赖的三方库变更** -->
