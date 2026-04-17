---
paths:
  - "lib/supabase.ts"
---

- Use `@supabase/ssr` for server-side client creation (createServerClient)
- Use `@supabase/supabase-js` for browser client (createBrowserClient)
- Environment vars: `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- Always use row-level security (RLS) — never bypass with service role key in client-facing code
- Database schema tables: `users`, `interviews`, `messages`, `results`
- Always handle Supabase errors — check `error` field on every query response
- Use typed queries with generated types where possible
