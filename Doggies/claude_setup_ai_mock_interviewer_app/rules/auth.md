---
paths:
  - "lib/auth.ts"
  - "app/(auth)/**/*.tsx"
  - "app/api/auth/**/*.ts"
---

- Use NextAuth.js with Google and GitHub OAuth providers
- Read `node_modules/next/dist/docs/` for auth patterns in this Next.js version
- Session provider wraps the app in the root layout
- Protect routes: redirect unauthenticated users to login page
- Store user data in Supabase `users` table via NextAuth adapter
- Export `auth()` helper from lib/auth.ts for server-side session checks
- Configure callbacks for session and JWT to include user ID
