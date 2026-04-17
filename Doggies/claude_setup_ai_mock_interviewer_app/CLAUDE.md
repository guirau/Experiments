# Mock Technical Interview App

@docs/prd/prd.md
@docs/negative-constraints.md
@docs/progress.md
@docs/learnings.md

## Key Context

- **Read `node_modules/next/dist/docs/` before using Next.js APIs** — Next.js 16 has breaking changes from training data
- Gemini API key: `.env.local` → `GEMINI_API_KEY`
- Supabase: `.env.local` → `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- MCPs connected: Supabase (DB operations), shadcn (UI components), Playwright (browser testing)
- `.claude/rules/` has path-specific rules that load automatically per file path
- `.claude/agents/`: `planner`, `git-commit`, `code-refactor`, `verification-agent` (Playwright), `ui-builder`
- `.claude/skills/`: `frontend-designer` (shadcn MCP), `backend-architect` (security patterns), `add-question`

## Rules

- Prefer server components; add `"use client"` only for interactive components (editor, chat, voice, timer)
- Never expose API keys client-side — all Gemini calls go through `/api/` server routes
- Do not install packages without asking first
- Run `npm run build` to verify changes compile before finishing
- Streaming responses: Gemini streaming → route handler `ReadableStream` → client renders tokens incrementally
- Dark theme only — Tailwind dark palette, monospace for code, VS Code aesthetic
- After completing a task, append an entry to `docs/progress.md` (what was done, files touched, status, blockers)
- When you hit a bug, wrong assumption, or non-obvious fix, append an entry to `docs/learnings.md` (mistake, root cause, fix, lesson)
- After every implementation, commit to GitHub using the `git-commit` agent 
- Track bugs and issues in the Notion page "Mock Technical Interviewer" → "Issue Reporting" database via Notion MCP (`mcp__claude_ai_Notion`). Log: issue title, description, severity, status, and related files
