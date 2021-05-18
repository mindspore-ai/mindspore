# MindSpore contributing guidelines

<!-- TOC -->

- [MindSpore contributing guidelines](#mindspore-contributing-guidelines)
    - [Contributor License Agreement](#contributor-license-agreement)
    - [Getting Started](#getting-started)
    - [Contribution workflow](#contribution-workflow)
        - [Code style](#code-style)
        - [Fork-Pull development model](#fork-pull-development-model)
        - [Report issues](#report-issues)
        - [Propose PRs](#propose-prs)

<!-- /TOC -->

## Contributor License Agreement

It's required to sign CLA before your first code submission to MindSpore community.

For individual contributor, please refer to [ICLA online document](https://www.mindspore.cn/icla) for the detailed information.

## Getting Started

- Fork the repository on [Github](https://github.com/mindspore-ai/mindspore) or [Gitee](https://gitee.com/mindspore/mindspore).
- Read the [README.md](README.md) and [install page](https://www.mindspore.cn/install/en) for project information and build instructions.

## Contribution Workflow

### Code style

Please follow this style to make MindSpore easy to review, maintain and develop.

- Coding guidelines

    The *Python* coding style suggested by [Python PEP 8 Coding Style](https://pep8.org/) and *C++* coding style suggested by [Google C++ Coding Guidelines](http://google.github.io/styleguide/cppguide.html) are used in MindSpore community. The [CppLint](https://github.com/cpplint/cpplint), [CppCheck](http://cppcheck.sourceforge.net), [CMakeLint](https://github.com/cmake-lint/cmake-lint), [CodeSpell](https://github.com/codespell-project/codespell), [Lizard](http://www.lizard.ws), [ShellCheck](https://github.com/koalaman/shellcheck) and [PyLint](https://pylint.org) are used to check the format of codes, installing these plugins in your IDE is recommended.

- Unittest guidelines

    The *Python* unittest style suggested by [pytest](http://www.pytest.org/en/latest/) and *C++* unittest style suggested by [Googletest Primer](https://github.com/google/googletest/blob/master/docs/primer.md) are used in MindSpore community. The design intent of a testcase should be reflected by its name of comment.

- Refactoring guidelines

    We encourage developers to refactor our code to eliminate the [code smell](https://en.wikipedia.org/wiki/Code_smell). All codes should conform to needs to the coding style and testing style, and refactoring codes are no exception. [Lizard](http://www.lizard.ws) threshold for nloc (lines of code without comments) is 100 and for cnc (cyclomatic complexity number) is 20, when you receive a *Lizard* warning, you have to refactor the code you want to merge.

- Document guidelines

    We use *MarkdownLint* to check the format of markdown documents. MindSpore CI modifies the following rules based on the default configuration.
    - MD007 (unordered list indentation): The **indent** parameter is set to **4**, indicating that all content in the unordered list needs to be indented using four spaces.
    - MD009 (spaces at the line end): The **br_spaces** parameter is set to **2**, indicating that there can be 0 or 2 spaces at the end of a line.
    - MD029 (sequence numbers of an ordered list): The **style** parameter is set to **ordered**, indicating that the sequence numbers of the ordered list are in ascending order.

    For details, please refer to [RULES](https://github.com/markdownlint/markdownlint/blob/master/docs/RULES.md).

### Fork-Pull development model

- Fork MindSpore repository

    Before submitting code to MindSpore project, please make sure that this project have been forked to your own repository. It means that there will be parallel development between MindSpore repository and your own repository, so be careful to avoid the inconsistency between them.

- Clone the remote repository

    If you want to download the code to the local machine, `git` is the best way:

    ```shell
    # For GitHub
    git clone https://github.com/{insert_your_forked_repo}/mindspore.git
    git remote add upstream https://github.com/mindspore-ai/mindspore.git
    # For Gitee
    git clone https://gitee.com/{insert_your_forked_repo}/mindspore.git
    git remote add upstream https://gitee.com/mindspore/mindspore.git
    ```

- Develop code locally

    To avoid inconsistency between multiple branches, checking out to a new branch is `SUGGESTED`:

    ```shell
    git checkout -b {new_branch_name} origin/master
    ```

    Taking the master branch as an example, MindSpore may create version branches and downstream development branches as needed, please fix bugs upstream first.
    Then you can change the code arbitrarily.

- Push the code to the remote repository

    After updating the code, you should push the update in the formal way:

    ```shell
    git add .
    git status # Check the update status
    git commit -m "Your commit title"
    git commit -s --amend #Add the concrete description of your commit
    git push origin {new_branch_name}
    ```

- Pull a request to MindSpore repository

    In the last step, your need to pull a compare request between your new branch and MindSpore `master` branch. After finishing the pull request, the Jenkins CI will be automatically set up for building test. Your pull request should be merged into the upstream master branch as soon as possible to reduce the risk of merging.

### Report issues

A great way to contribute to the project is to send a detailed report when you encounter an issue. We always appreciate a well-written, thorough bug report, and will thank you for it!

When reporting issues, refer to this format:

- What version of env (mindspore, os, python etc) are you using?
- Is this a BUG REPORT or FEATURE REQUEST?
- What kind of issue is, add the labels to highlight it on the issue dashboard.
- What happened?
- What you expected to happen?
- How to reproduce it?(as minimally and precisely as possible)
- Special notes for your reviewers?

**Issues advisory:**

- **If you find an unclosed issue, which is exactly what you are going to solve,** please put some comments on that issue to tell others you would be in charge of it.
- **If an issue is opened for a while,** it's recommended for contributors to precheck before working on solving that issue.
- **If you resolve an issue which is reported by yourself,** it's also required to let others know before closing that issue.
- **If you want the issue to be responded as quickly as possible,** please try to label it, you can find kinds of labels on [Label List](https://gitee.com/mindspore/community/blob/master/sigs/dx/docs/labels.md)

### Propose PRs

- Raise your idea as an *issue* on [GitHub](https://github.com/mindspore-ai/mindspore/issues) or [Gitee](https://gitee.com/mindspore/mindspore/issues)
- If it is a new feature that needs lots of design details, a design proposal should also be submitted.
- After reaching consensus in the issue discussions and design proposal reviews, complete the development on the forked repo and submit a PR.
- None of PRs is not permitted until it receives **2+ LGTM** from approvers. Please NOTICE that approver is NOT allowed to add *LGTM* on his own PR.
- After PR is sufficiently discussed, it will get merged, abandoned or rejected depending on the outcome of the discussion.

**PRs advisory:**

- Any irrelevant changes should be avoided.
- Make sure your commit history being ordered.
- Always keep your branch up with the master branch.
- For bug-fix PRs, make sure all related issues being linked.
