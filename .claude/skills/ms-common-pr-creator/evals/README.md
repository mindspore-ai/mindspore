# GitCode PR Evals

This directory keeps the eval definitions for the `ms-pr-creator` skill.

## Files

- `evals.json`: high-level eval prompts and expected outcomes
- `metadata/*.json`: repo-specific assertion lists and grading parameters
- `trigger_evals.json`: description trigger eval queries
- `TEST_MATRIX.md`: layered test strategy for trigger, simulated generation, hermetic script tests, and live smoke tests

## Important

When running evals for this skill, do not assume every assertion is graded the same way.

- Some assertions are graded programmatically by `ms-pr-creator-workspace/grade_assertions.py`
- Some assertions are graded by an LLM judge through the workspace's two-phase judge flow

Before grading or revising the eval pipeline, read:

- `../../ms-pr-creator-workspace/README.md`

That README explains:

- which assertions are programmatic vs LLM-judge
- how to run `--prepare-judge` with `--config`
- how to produce `judge_results_<config>.json`
- how to run `--apply-judge` with `--config` and merge verdicts back into the matching `grading.json`

## Why This Exists

`skill-creator` defines the general eval workflow and shared JSON schemas, but this repo adds
a repo-specific grading protocol. This README exists so an agent working from the eval directory
can discover that protocol without reverse-engineering the grader script first.
