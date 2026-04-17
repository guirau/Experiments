---
paths:
  - "app/**/page.tsx"
  - "app/**/layout.tsx"
---

- Read `node_modules/next/dist/docs/` before writing any page or layout
- Prefer server components; only add "use client" when the page truly needs interactivity
- Use Tailwind dark theme consistently (dark backgrounds, light text)
- Interview page (`app/interview/[id]/page.tsx`) is a client component — needs state, voice, and editor
- Dashboard page: use a lightweight charting library for performance visualizations
- Landing page: minimal dark hero section with call-to-action
- Layouts must not duplicate the HTML/body tags — only root layout defines those
