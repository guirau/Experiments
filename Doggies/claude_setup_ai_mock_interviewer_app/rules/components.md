---
paths:
  - "components/**/*.tsx"
---

- All components in this directory are client components — always include "use client" at top
- Use Tailwind CSS exclusively for styling — dark theme with VS Code aesthetic (gray-900, gray-800 backgrounds)
- Monaco editor: use `@monaco-editor/react`, support language switching (JS, Python, etc.)
- Voice controls: use browser-native Web Speech API (SpeechRecognition + SpeechSynthesis) — no extra dependencies
- ChatPanel: support streaming message rendering (token-by-token) using state updates
- All components must be typed with TypeScript interfaces imported from `lib/types.ts`
- No inline styles — Tailwind only
- Handle loading and error states in every component
