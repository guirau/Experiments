# Design Tokens

## Colors
| Token | Value | Usage |
|-------|-------|-------|
| bg-primary | gray-950 | Page backgrounds |
| bg-surface | gray-900 | Cards, panels |
| bg-elevated | gray-800 | Inputs, hover states |
| text-primary | gray-100 | Headings, body text |
| text-secondary | gray-400 | Labels, placeholders |
| text-muted | gray-500 | Disabled, hints |
| accent | blue-500 | Primary buttons, links, focus rings |
| success | green-500 | Pass indicators, positive scores |
| warning | amber-500 | Medium difficulty, cautions |
| error | red-500 | Errors, hard difficulty, destructive actions |
| border | gray-700 | Subtle dividers, card borders |

## Typography
| Element | Font | Size | Weight |
|---------|------|------|--------|
| Code / Editor | `font-mono` (system monospace) | text-sm | normal |
| Headings | `font-sans` (system sans) | text-xl to text-3xl | bold |
| Body | `font-sans` | text-sm to text-base | normal |
| Labels | `font-sans` | text-xs to text-sm | medium |

## Spacing
- Base unit: 4px
- Component padding: 12px-16px (p-3 to p-4)
- Section gaps: 24px-32px (gap-6 to gap-8)
- Card padding: 16px-24px (p-4 to p-6)

## Borders
- Radius: rounded-lg (cards), rounded-md (buttons/inputs)
- Width: border (1px default)
- Color: border-gray-700

## Shadows
- Cards: shadow-none (rely on border contrast in dark theme)
- Modals/dialogs: shadow-xl with dark overlay
