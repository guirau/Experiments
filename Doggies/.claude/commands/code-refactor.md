---
description: Refactor code for clarity, performance, and maintainability without changing external behavior
argument-hint: [file or directory to refactor]
---

Dispatch to the `code-refactor` agent to refactor the target: $ARGUMENTS

If no argument is provided, refactor the files changed since the last commit.

Use the Agent tool with `subagent_type="code-refactor"` and pass $ARGUMENTS as the target scope.
