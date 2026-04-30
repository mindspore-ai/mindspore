---
name: ms-pylint-fix
description: Fix issues found by pylint. Use this skill whenever a file has been changed to run pylint checks.
---

# MindSpore Pylint Issues Fix Guide

## Filter

Read the file `.jenkins/check/config/filter_pylint.txt` line by line to get directories, files, or issues that will not be fixed.
Analyze the pylintrc file `.jenkins/rules/pylint/pylintrc` and do not fix the issues specified as disabled.
Don't fix pylint issues: design,similarities,no-self-use,attribute-defined-outside-init,locally-disabled,star-args,pointless-except,bad-option-value,global-statement,fixme,suppressed-message,useless-suppression,locally-enabled,no-member,no-name-in-module,import-error,unsubscriptable-object,unbalanced-tuple-unpacking,undefined-variable,not-context-manager,duplicate-code,consider-using-f-string,import-outside-toplevel,unnecessary-dunder-call,arguments-renamed,broad-exception-raised,c-extension-no-member,consider-using-from-import

## Fix

- Get the changed files by `git diff HEAD~1 HEAD` and run pylint checks with pylintrc `.jenkins/rules/pylint/pylintrc`. Find issues and fix them.
- Get the given issues and fix each file line by line.

## Quick Start

```bash
# With target branch specified, after git-commit, run command:
/ms-pylint-fix
# Find files changed automatically by "git diff",
# then run command "pylint --rcfile=[pylintrc] [files]" to find issues, fix them if exists.
```