---
paths:
  - "lib/gemini.ts"
  - "app/api/interview/**/*.ts"
---

- Use `@google/generative-ai` SDK (GoogleGenerativeAI class)
- API key from env: `GEMINI_API_KEY` — never expose client-side
- System prompt: realistic interviewer persona — asks follow-ups, gives hints when stuck, pushes back on suboptimal approaches
- Support streaming responses via `generateContentStream`
- Include `code_snapshot` context in every message so the model has real-time code awareness
- Scoring: generate detailed scorecard (1-10) across problem-solving, code quality, communication, and complexity analysis
- Use chat sessions (startChat) to maintain conversation history per interview
