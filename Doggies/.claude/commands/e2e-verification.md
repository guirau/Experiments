---
description: Validate UI and user flows by driving a real browser via Playwright
argument-hint: [URL or flow to verify, e.g. "http://localhost:3000 login flow"]
---

Dispatch to the `verification-agent` to verify the target: $ARGUMENTS

If no argument is provided, verify the full app at the default local URL.

Use the Agent tool with `subagent_type="verification-agent"` and pass $ARGUMENTS as the target scope.
