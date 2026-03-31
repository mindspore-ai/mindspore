---
name: ms-common-auto-debug
description: Debug MindSpore issues from any starting point — a failing test, an error message, a coredump stack, or a problem description. First constructs a minimal reproducer, then applies 3-level escalation (static analysis → runtime logging → add C++ logging + rebuild) to locate and fix the root cause. Triggers on phrases like "test failed", "debug this", "coredump", "segfault", "stack trace", "why does this fail", "fix this test", "wrong gradient", "unexpected output", "investigate", "reproduce this error".
---

# MindSpore Auto Debug

A structured escalation workflow for debugging any MindSpore issue. Each phase is attempted in order; only escalate when the current phase fails to reveal root cause.

## Phase 0: Construct a minimal reproducer

**Input can be any of:**
- A failing test name → run it directly to confirm failure
- An error message / exception traceback → extract key info and write a minimal script
- A coredump / segfault stack → identify the crashing function, write a script that exercises that code path
- A natural language problem description → translate to a concrete test case

**Goal:** Produce a short, self-contained script that reproduces the issue.

**Rules:**
- Strip away everything not needed to trigger the bug — minimal imports, hardcoded inputs, no test framework
- Preserve the execution mode (PyNative / Graph) and context settings from the original failure
- Always run the reproducer and confirm it actually fails before proceeding
- If the original input is already a minimal test, skip writing a new script — just confirm it fails

## Phase 1: Static analysis — read code, reason, fix

**When to use:** First attempt after reproducer is confirmed.

**Steps:**
1. Trace the code path from the reproducer's entry point into the implementation (use `ms-common-debug` and `ms-common-conde-analysis` skills for navigation guidance)
2. Reason about the failure: wrong logic? missing condition? type mismatch? wrong compilation pass?
3. Make the fix
4. Run the reproducer to verify the fix

**Exit:** If you can identify and fix the root cause, you're done. Otherwise escalate to Phase 2.

## Phase 2: Runtime logging — enable verbose logs

**When to use:** Static analysis didn't reveal root cause.

**Steps:**
1. Enable framework verbose logging (refer to `ms-common-debug` skill for environment variables and log levels)
2. Run reproducer, capture logs
3. Filter logs for keywords related to the suspected area
4. Analyze execution flow — look for divergence from expected behavior
5. Fix and verify

**Exit:** If logs reveal root cause, fix and verify. Otherwise escalate to Phase 3.

## Phase 3: Add custom C++ logging, rebuild, iterate

**When to use:** Existing framework logs don't cover the critical code path.

**Cycle:**
1. Add temporary log statements at the suspicious function(s) (refer to `ms-common-debug` for logging conventions)
2. Rebuild incrementally (refer to `ms-common-build` for build commands)
3. Install the rebuilt package
4. Run reproducer, analyze new logs
5. Narrow the suspect area based on log output, add more logs if needed
6. Repeat 1–5 until root cause is found
7. Make the actual fix
8. **Remove all temporary debug logging**
9. Rebuild, install, run the reproducer + original test to confirm

## Decision flow

```
Problem input (test / error / coredump / description)
  │
  ├─ Phase 0: Build minimal reproducer → confirm it fails
  │
  ├─ Phase 1: Read code → root cause obvious?
  │    ├─ YES → fix → verify → done
  │    └─ NO ↓
  │
  ├─ Phase 2: Verbose logs → useful info?
  │    ├─ YES → analyze → fix → verify → done
  │    └─ NO ↓
  │
  └─ Phase 3: Add custom logging → rebuild → analyze
       └─ iterate until root cause → fix → clean logs → rebuild → verify
```
