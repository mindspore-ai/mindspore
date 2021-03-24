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

    The *Python* coding style suggested by [Python PEP 8 Coding Style](https://pep8.org/) and *C++* coding style suggested by [Google C++ Coding Guidelines](http://google.github.io/styleguide/cppguide.html) are used in MindSpore community.

- Unittest guidelines

    The *Python* unittest style suggested by [pytest](http://www.pytest.org/en/latest/) and *C++* unittest style suggested by [Googletest Primer](https://github.com/google/googletest/blob/master/docs/primer.md) are used in MindSpore community.

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

    In the last step, your need to pull a compare request between your new branch and MindSpore `master` branch. After finishing the pull request, the Jenkins CI will be automatically set up for building test.

### Report issues

A great way to contribute to the project is to send a detailed report when you encounter an issue. We always appreciate a well-written, thorough bug report, and will thank you for it!

When reporting issues, refer to this format:

- What version of env (mindspore, os, python etc) are you using?
- Is this a BUG REPORT or FEATURE REQUEST?
- What happened?
- What you expected to happen?
- How to reproduce it?(as minimally and precisely as possible)
- Special notes for your reviewers?

**Issues advisory:**

- **If you find an unclosed issue, which is exactly what you are going to solve,** please put some comments on that issue to tell others you would be in charge of it.
- **If an issue is opened for a while,** it's recommended for contributors to precheck before working on solving that issue.
- **If you resolve an issue which is reported by yourself,** it's also required to let others know before closing that issue.

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
