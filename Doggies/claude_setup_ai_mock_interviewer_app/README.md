# Claude Setup Ways — Mock Technical Interview App

A reference Claude Code configuration for building a mock technical interview app with Next.js 16, Supabase, and the Gemini API. This repo is the `.claude`-style configuration layer: agents, skills, rules, and project instructions — not the app code itself.

Use it as a template for structuring your own Claude Code project configs.

## Layout

```
.
├── CLAUDE.md              # Project instructions loaded into every Claude Code session
├── settings.local.json    # Local Claude Code settings (permissions, MCP toggles)
├── agents/                # Subagents invoked via the Agent tool
├── skills/                # Skills invoked via the Skill tool (or /slash command)
└── rules/                 # Path-scoped rules auto-loaded per file being edited
```

## CLAUDE.md

The always-loaded project brief. Declares the stack, env var locations, connected MCPs (Supabase, shadcn, Playwright, Notion), project-wide rules (server-components-first, dark theme, no client-side API keys), and the docs to read before using Next.js 16 APIs.

Edit this when a project-wide invariant changes. Keep it short — it ships in every turn.

## Agents (`agents/`)

Each `.md` file defines a subagent Claude can spawn via the Agent tool. Frontmatter sets `name`, `description`, `model`, and optional `tools`.

| Agent | Model | Purpose |
|---|---|---|
| [planner](agents/planner.md) | — | Breaks work into step-by-step implementation plans |
| [code-refactor](agents/code-refactor.md) | sonnet | Behavior-preserving restructuring with build/lint/test verification |
| [git-commit](agents/git-commit.md) | haiku | Conventional commits with secret scanning and logical splitting |
| [verification-agent](agents/verification-agent.md) | sonnet | Drives Playwright MCP to validate UI flows, console, and network |

## Skills (`skills/`)

Each skill is a directory with a `SKILL.md` plus optional `references/` (docs) and `scripts/` (executables). Invoked via the Skill tool or `/skill-name`.

| Skill | Purpose |
|---|---|
| [add-question](skills/add-question/SKILL.md) | Adds a new LeetCode-style question to `lib/questions.ts` with duplicate and complexity checks |
| [backend-architect](skills/backend-architect/SKILL.md) | Route-handler design with a mandatory security checklist |
| [frontend-designer](skills/frontend-designer/SKILL.md) | UI via the shadcn MCP, dark-theme tokens, and full state coverage |
| [test-interview-api](skills/test-interview-api/SKILL.md) | End-to-end API smoke test with an included bash runner |

Skill references and scripts:
- `skills/backend-architect/references/security-checklist.md` — pre-commit checklist for every new route
- `skills/frontend-designer/references/design-tokens.md` — color, typography, spacing, border tokens
- `skills/test-interview-api/scripts/test-flow.sh` — curl/jq runner for the full interview flow

## Rules (`rules/`)

Path-scoped instructions that Claude Code auto-loads based on which file is being edited. Each file targets a slice of the codebase:

| Rule | Applies to |
|---|---|
| `api-routes.md` | `app/api/**` route handlers |
| `auth.md` | auth flows and session handling |
| `components.md` | React components |
| `gemini.md` | Gemini API calls and streaming |
| `pages.md` | Next.js pages and layouts |
| `supabase.md` | Supabase client usage and RLS |

Keep rules narrow — path-scoped instructions should cover guarantees that a generic rule in `CLAUDE.md` cannot.

## Using this config

1. Drop `agents/`, `skills/`, `rules/`, and `CLAUDE.md` into your project root (or `.claude/` depending on your Claude Code layout)
2. Fill in `.env.local` with `GEMINI_API_KEY`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`
3. Connect the MCPs referenced in `CLAUDE.md`: Supabase, shadcn, Playwright, Notion
4. Start a Claude Code session — the project brief, rules, and skills load automatically

## Conventions

- **Read `node_modules/next/dist/docs/` before using Next.js APIs** — Next.js 16 has breaking changes from training data
- Server components by default; `"use client"` only for interactive pieces (editor, chat, voice, timer)
- All Gemini calls go through `/api/*` server routes — never client-side
- Dark theme only, monospace for code, VS Code aesthetic
- After every implementation: `npm run build`, append to `docs/progress.md`, commit via the `git-commit` agent
- Bugs and learnings go to the Notion "Issue Reporting" database and `docs/learnings.md` respectively
