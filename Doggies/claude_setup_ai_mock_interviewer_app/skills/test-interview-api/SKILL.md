---
name: test-interview-api
description: Tests the interview API flow end-to-end by simulating a full interview session with mock candidate responses
---

# Test Interview API

Simulates a complete interview session against the backend API to verify the full flow works — no browser needed. Fast smoke test before and after backend changes.

## Quick run
For routine checks, use the script:
```
./scripts/test-flow.sh [base_url]
```
It exercises auth → start → message (approach) → message (code) → end → double-end guard. Requires `curl`, `jq`, and an auth cookie.

## Getting an auth cookie
The API requires a logged-in session. Either:
- Log in via the browser on `localhost:3000`, then copy the session cookie from DevTools → Application → Cookies into the `COOKIE` variable in the script
- Or set `COOKIE=` via environment before invoking the script

If you cannot obtain a cookie, stop and ask — do not attempt to bypass auth.

## Manual flow (when the script is not enough)

### 1. Start interview
```
POST /api/interview/start
Body: { "topic": "arrays", "difficulty": "medium", "language": "javascript" }
Expected: 200 with { id, question_title, question_description, time_limit_seconds }
```
Save the returned `id` — every subsequent request uses it.

### 2. Initial candidate message (reading the problem)
```
POST /api/interview/message
Body: {
  "interview_id": "<id>",
  "message": "I see. So I need to find two numbers that add up to the target. I could use a hash map to do this in O(n) time.",
  "code_snapshot": ""
}
Expected: streaming response with interviewer follow-up
```

### 3. Candidate writes code and asks a clarifying question
```
POST /api/interview/message
Body: {
  "interview_id": "<id>",
  "message": "Can the array contain duplicate values? Here's my initial approach.",
  "code_snapshot": "function twoSum(nums, target) {\n  const map = new Map();\n  for (let i = 0; i < nums.length; i++) {\n    const complement = target - nums[i];\n    if (map.has(complement)) return [map.get(complement), i];\n    map.set(nums[i], i);\n  }\n}"
}
Expected: streaming response — interviewer references the code, asks about edge cases or complexity
```

### 4. Candidate discusses complexity
```
POST /api/interview/message
Body: {
  "interview_id": "<id>",
  "message": "Time is O(n) — single pass. Space is O(n) for the hash map in the worst case.",
  "code_snapshot": "<same as above>"
}
Expected: streaming response — interviewer confirms or pushes back
```

### 5. End interview and get scorecard
```
POST /api/interview/end
Body: { "interview_id": "<id>" }
Expected: 200 with full scorecard:
  - overall_score (1-10)
  - problem_solving_score (1-10)
  - code_quality_score (1-10)
  - communication_score (1-10)
  - time_complexity_analysis (string)
  - space_complexity_analysis (string)
  - strengths (string[])
  - improvements (string[])
  - recommendations (string[])
  - summary (string)
```

## Verification checks
- Every endpoint returns 401 without a valid session — test this first
- `start` creates a row in `interviews` (verify via Supabase MCP)
- Each message creates two rows in `messages` (candidate + interviewer)
- `end` creates a row in `results` with every score field populated and within 1-10
- Streaming responses arrive in chunks (not one large buffered payload) — if the full body arrives at once, streaming is broken
- Interview status transitions `in_progress` → `completed`, never skips or reverses

## Error cases to test
- Start with invalid topic/difficulty → 400
- Message to a nonexistent `interview_id` → 404
- Message to an already-completed interview → 400
- End an already-ended interview → 400
- Missing required fields in any request → 400
- Cross-user access: start an interview as user A, attempt to message it with user B's cookie → 404 or 403 (never 200)

## Reporting
Report per endpoint: status code, whether the expected response shape was returned, and any Supabase row writes observed. Do not attempt to fix failures — log them and hand back to the caller.
