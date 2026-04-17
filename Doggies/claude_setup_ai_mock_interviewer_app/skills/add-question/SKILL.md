---
name: add-question
description: Add a new coding interview question to the curated question bank in lib/questions.ts
---

# Add Interview Question

Adds a new LeetCode-style problem to `lib/questions.ts`. The question bank is the source of truth for interview prompts — quality matters more than quantity.

## Before adding
1. Read the canonical `Question` type at the top of `lib/questions.ts` and match it exactly
2. Grep the existing questions for the title and a few distinctive phrases — reject duplicates and near-duplicates
3. Check that the topic and difficulty combination is not already over-represented (aim for balance across `easy | medium | hard` × topic tags)

## Required fields
- `id`: kebab-case slug, unique across the file (e.g., `two-sum`, `longest-palindromic-substring`)
- `title`: short, human-readable (e.g., "Two Sum")
- `description`: the problem statement, written in the same voice as existing questions
- `examples`: at least 2, each with `input`, `output`, and optional `explanation`
- `constraints`: array of string bullets (input size, value ranges, edge case notes)
- `expectedComplexity`: `{ time: string; space: string }` — the target, not the brute force
- `topicTags`: one or more of `arrays`, `strings`, `trees`, `graphs`, `dp`, `sorting`, `hashing`, `two-pointers`, `binary-search`
- `difficulty`: `easy | medium | hard`

## Validation (before saving)
- Walk each example by hand: given the input, does the output match? Trace it.
- Confirm the stated complexity is achievable — do not claim O(n) if the reference solution is O(n log n)
- Constraints must be tight enough to rule out brute-force solutions at the stated difficulty
- Description must be unambiguous — no "obvious" assumptions the candidate has to guess at

## After saving
- Run `npm run build` to catch type errors in the new entry
- If a test file exercises the question bank, run it
