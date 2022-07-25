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

**Code review checklist [[illustration]](https://gitee.com/mindspore/community/blob/master/security/code_review_checklist_mechanism.md)**:

- **Typical problems of security coding [[historical security coding cases reference]](https://gitee.com/mindspore/community/blob/master/security/security_coding_violation_cases.md)**
    - [ ] whether to verify the pointer is null/nullptr
    - [ ] whether to verify the function's return value (It is forbidden to use void to mask the return values of security functions and self-developed functions. C++ STL functions can be masked if there is no problem)
    - [ ] whether new/malloc memory is released correctly
- **Performance analysis (if a sub-item is involved, please outline the implementation idea or modification content)**
    - [ ] whether to modify hotspot ***function / algorithm / operation***
    - [ ] whether to consider concurrent scenarios
    - [ ] whether to consider communication scenario
+ - [ ] **Whether to comply with coding specifications [[coding specification reference]](https://gitee.com/mindspore/community/blob/master/security/coding_guild_cpp_zh_cn.md)**
+ - [ ] **Whether to comply with ***SOLID principle / Demeter's law*****
+ - [ ] **Whether the ***interaction between modules / features*** is involved (if yes, please outline the implementation ideas)**
+ - [ ] **Whether there is UT test case && the test case is a valid (if there is no test case, please explain the reason)**
+ - [ ] **whether the secret key is loaded/released correctly**
- **Error handling and recording**
    - [ ] whether the interface exception scenarios are fully considered
    - [ ] whether the error is recorded appropriately

**Special notes for your reviewers**:
<!-- + - [ ] **Whether document (installation, tutorial, design, reference, API, migration guide, FAQ, etc.) modification is involved** -->
<!-- + - [ ] **Whether it causes forward compatibility failure** -->
<!-- + - [ ] **Whether the API change is involved** -->
<!-- + - [ ] **Whether the dependent third-party library change is involved** -->
