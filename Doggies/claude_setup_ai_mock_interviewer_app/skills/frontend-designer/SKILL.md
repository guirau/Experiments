---
name: frontend-designer
description: Improves UI design and polish using shadcn components via the shadcn MCP
---

# Frontend Designer

Builds and improves the UI with shadcn components via the shadcn MCP. Design tokens and dark-theme palette live in `references/design-tokens.md` — consult it before writing any styling.

## When to use
- Building new pages or sections that need polished UI
- Replacing raw HTML/Tailwind with proper shadcn component patterns
- Improving visual consistency or accessibility across existing pages

## Process
1. Check if the component already exists in `components/ui/` — reuse before installing
2. Browse the shadcn MCP for the right primitive; install only the minimum set needed
3. Compose into the page/section, wiring handlers and state
4. Apply tokens from `references/design-tokens.md` — do not invent new colors, radii, or spacing values
5. Verify accessibility: keyboard navigation, visible focus rings, ARIA labels on icon-only buttons, dialogs trap focus, form inputs have associated labels

## Server vs client
- Default to server components per CLAUDE.md
- Add `"use client"` only when the component needs interactivity: editor, chat, voice, timer, form state, hover state that JS drives
- Keep the client bundle lean — push data fetching up to the server component parent

## State coverage
For every view, handle and design:
- Loading (skeleton or spinner — prefer skeleton for lists/cards)
- Empty (helpful copy + next action, never a blank panel)
- Error (scoped to the failing region, with retry where possible)
- Success (the happy path)

## Component mapping for this app
- **Buttons** → shadcn Button (CTA, submit, toggle voice)
- **Forms** → shadcn Input, Select, Label, Form (interview setup)
- **Cards** → shadcn Card (dashboard stats, scorecard categories)
- **Tables** → shadcn Table (interview history)
- **Badges** → shadcn Badge (difficulty pills, topic tags)
- **Dialogs** → shadcn Dialog / AlertDialog (confirm end interview)
- **Charts** → shadcn Charts (dashboard analytics)
- **Tabs** → shadcn Tabs (results breakdown)
- **Skeletons** → shadcn Skeleton (loading states)
- **Tooltips** → shadcn Tooltip (icon-only controls)
