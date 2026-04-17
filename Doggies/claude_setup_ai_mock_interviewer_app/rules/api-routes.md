---
paths:
  - "app/api/**/*.ts"
---

- All API routes use Next.js Route Handlers (export async GET/POST/PUT/DELETE functions)
- Read `node_modules/next/dist/docs/` for Route Handler API before writing any route
- Stream Gemini responses using ReadableStream for chat/interview endpoints
- Always validate session with NextAuth `auth()` before processing requests
- Never expose API keys client-side — use server-side env vars only (process.env)
- Return proper HTTP status codes and JSON error responses via NextResponse.json()
- Interview message endpoint must accept `code_snapshot` alongside the message body
- Parse request body with `await request.json()` and validate required fields
- Wrap handlers in try/catch and return 500 with error details on failure
