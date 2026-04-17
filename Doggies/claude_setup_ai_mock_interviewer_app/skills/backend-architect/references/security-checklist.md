# API Security Checklist

Run through this for every new or modified API route.

## Authentication
- [ ] Route calls `auth()` and checks `session?.user` before any logic
- [ ] Returns 401 with `{ error: "Unauthorized" }` if no valid session

## Input Validation
- [ ] Request body parsed with `await request.json()`
- [ ] All required fields checked — return 400 if missing
- [ ] Enum fields validated against allowed values (topic, difficulty, language, status)
- [ ] String fields trimmed and length-checked where applicable
- [ ] UUIDs validated as proper format before querying

## Data Access
- [ ] Supabase queries filter by `user_id` from session (defense in depth with RLS)
- [ ] No service role key used — anon key + RLS only
- [ ] No string concatenation in queries — always use parameterized `.eq()`, `.in()`, etc.

## Response
- [ ] Success: proper status code (200, 201) with typed JSON body
- [ ] Client errors: 400/404 with `{ error: "Human-readable message" }`
- [ ] Server errors: 500 with `{ error: "Internal server error" }` — no stack traces
- [ ] Streaming: proper `Content-Type` and `ReadableStream` setup

## Rate Limiting
- [ ] Gemini-calling endpoints have rate limits
- [ ] Interview creation limited per user per time window
