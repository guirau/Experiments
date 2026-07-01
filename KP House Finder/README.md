# KP House Finder

A personal pipeline for finding a **long-term rental in Koh Phangan**. Rental
listings mostly live in Facebook groups and WhatsApp group chats as free-form
text — impossible to filter. This project scrapes those posts, uses an LLM to
turn each one into structured fields (price, area, bedrooms, long-term
suitability, sublet potential…), stores them in Supabase, and serves a web
dashboard with filters, a map, and a personal "listings I'm pursuing" tracker.

---

## How it works (data flow)

```
 Facebook groups          ┌──────────────────┐        ┌─────────────────────┐
 WhatsApp Web chats  ───▶  │  Chrome extension │  ───▶  │  Supabase: fb_posts │   (raw scraped text)
                          │  (scrape + sync)  │        └─────────┬───────────┘
                          └──────────────────┘                  │
                                                                 ▼
                                                   ┌──────────────────────────┐
                                                   │  src/analyze.py (LLM)     │   incremental + idempotent
                                                   │  extract.py + db.py       │   (Claude Haiku)
                                                   └─────────┬────────────────┘
                                                             ▼
                                              ┌──────────────────────────────┐
                                              │  Supabase: listings_parsed    │   (structured, filterable)
                                              └─────────┬─────────────────────┘
                                                        ▼
                                        ┌───────────────────────────────────┐
                                        │  web/ (Next.js dashboard)          │
                                        │  filters · map · tracker table     │
                                        └────────────┬──────────────────────┘
                                                     ▼
                                        ┌───────────────────────────────────┐
                                        │  Supabase: tracker (manual entry)  │
                                        └───────────────────────────────────┘
```

There are two generations of the parsing path:

- **v1 (current):** extension → `fb_posts` in Supabase → `src/analyze.py` → `listings_parsed`.
- **v0 (legacy CSV):** JSON exports → `src/combine.py` → CSV → `src/extract.py` → CSV.
  Still supported for offline experiments; the web app does **not** read the CSVs.

---

## Repository structure

```
KP House Finder/
├── chrome_extension/KP-Rentals-Exporter/   # Chrome MV3 extension (scraper + Supabase sync)
│   ├── manifest.json, popup.*, background.js
│   └── supabase_schema.sql                  # creates fb_posts (+ RLS) — kept WITH the extension
├── src/                                     # Python pipeline
│   ├── analyze.py    # CLI: fb_posts -> listings_parsed (incremental, idempotent)
│   ├── extract.py    # LLM extraction engine (Claude Haiku, batched)
│   ├── db.py         # Supabase access layer (paginated reads / upsert)
│   └── combine.py    # legacy v0: JSON exports -> deduped CSV
├── sql/                                     # database schema (run in Supabase SQL Editor)
│   ├── listings_parsed.sql                  # structured-listings table (+ RLS)
│   └── tracker.sql                          # personal pipeline table (+ RLS)
├── web/                                      # Next.js 16 dashboard (reads Supabase)
├── notebooks/v0_fb_posts_transform.ipynb    # visual checkpoint for the extract engine
├── tests/                                   # pytest suite for the Python pipeline
├── docs/                                    # FIELD_SCHEMA.md + design/plan docs
├── pyproject.toml / poetry.lock             # Python deps (Poetry, in-project .venv)
└── .env                                     # secrets (gitignored) — see .env.example
```

> **Note on SQL location:** the two shared schemas live in `sql/`. The extension's
> `supabase_schema.sql` intentionally stays inside `chrome_extension/` so the
> extension remains a self-contained, loadable package (its own README references it).

---

## Prerequisites

- **Python** ≥ 3.11 and **[Poetry](https://python-poetry.org/)**
- **Node.js** ≥ 20 and **npm** (for the web app)
- A **[Supabase](https://supabase.com/)** project (free tier is fine)
- An **[Anthropic API key](https://console.anthropic.com/)** (for LLM extraction)
- **Google Chrome** (for the scraper extension)

---

## Setup

### 1. Supabase (database)

1. Create a project at [supabase.com](https://supabase.com/). Note its **Project URL**
   and **anon public key** (Project Settings → API).
2. Open the **SQL Editor** and run these three scripts **in this order** — start
   with `supabase_schema.sql` (it creates the `fb_posts`/`wa_messages` foundation
   the rest of the pipeline reads from), then the two in `sql/`:
   1. `chrome_extension/KP-Rentals-Exporter/supabase_schema.sql` → creates **`fb_posts`** (+ `wa_messages`)
   2. `sql/listings_parsed.sql` → creates **`listings_parsed`**
   3. `sql/tracker.sql` → creates **`tracker`**

   **Within each file, run `STEP 1` before `STEP 2`.** STEP 1 creates the table;
   STEP 2 enables Row-Level Security + anon read/write policies — and STEP 2 will
   error if its table doesn't exist yet. (The three files have no cross-table
   foreign keys, so the order *between* files is a sensible convention rather than
   strictly enforced — but the STEP 1 → STEP 2 order inside each file is required.)

RLS is enabled with permissive anon policies because this is a single-user,
low-stakes tool. The anon key is used everywhere (extension, Python, web).

### 2. Chrome extension (scraper)

The extension scrapes posts from Facebook groups and WhatsApp Web and syncs them
to the `fb_posts` table (with JSON download as a backup). It runs **only when you
click it**, in your own logged-in browser session.

1. Go to `chrome://extensions` → enable **Developer mode**.
2. **Load unpacked** → select `chrome_extension/KP-Rentals-Exporter/`. Pin it.
3. Click the extension → open its **Supabase settings** and paste your Project
   **URL** and **anon key**, enable sync, and hit **Test** (stored in the browser's
   `chrome.storage.local`, never in a file).
4. **Facebook:** open a group feed near the top → set a stop count / cutoff date →
   **Grab posts**. It scrolls, expands "See more", harvests text + permalink + date.
5. **WhatsApp:** open a group chat at `web.whatsapp.com` → **Start grabbing** →
   scroll up by hand → **STOP & DOWNLOAD**.

> Long Facebook grabs pause if the Mac sleeps — run `caffeinate -dimsu` in a
> terminal while scraping (see the extension's own README for details).

### 3. Python pipeline (LLM extraction)

From the project root:

```bash
poetry install                       # creates the in-project .venv, installs deps
cp .env.example .env                 # then fill in the three keys below
```

`.env` (gitignored) needs:

```
ANTHROPIC_API_KEY=sk-ant-...
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_ANON_KEY=<anon-public-key>
```

Run the extraction (always from the project root):

```bash
poetry run python src/analyze.py --inspect   # row counts + samples + schema check, NO writes
poetry run python src/analyze.py --limit 5   # parse at most 5 new rows (cheap trial)
poetry run python src/analyze.py             # parse ALL un-parsed fb_posts
```

`analyze.py` is **incremental and idempotent**: it only processes `fb_posts` rows
whose id isn't already in `listings_parsed`, and each row is sent to the LLM at
most once (batched). Re-running after a scrape only parses the new rows. Rows are
never deleted — non-listings / wanted / short-term posts get a `discard_reason`
and are filtered out by the dashboard's "offers" view.

<details>
<summary>Legacy v0 CSV path (optional, offline only)</summary>

```bash
poetry run python src/combine.py ./sources     # JSON exports -> listings.csv (deduped)
poetry run python src/extract.py               # listings.csv -> parsed_listings.csv
```
The web app does not read these CSVs; use the Supabase path above for the dashboard.
</details>

### 4. Web dashboard

```bash
cd web
npm install
cp .env.example .env.local            # fill in the two NEXT_PUBLIC_ vars
npm run dev                           # http://localhost:3000
```

`web/.env.local`:

```
NEXT_PUBLIC_SUPABASE_URL=https://<project-ref>.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=<anon-public-key>
```

The dashboard reads `listings_parsed` (filterable cards + area map) and `tracker`
(a drag-to-reorder table of listings you're actively pursuing, edited in-app).
See `web/README.md` for build/deploy details (`npm run build`, Vercel).

---

## Database tables

| Table             | Created by                                   | Purpose                                              |
| ----------------- | -------------------------------------------- | ---------------------------------------------------- |
| `fb_posts`        | `chrome_extension/.../supabase_schema.sql`   | Raw scraped posts (text, source, link, date).        |
| `listings_parsed` | `sql/listings_parsed.sql`                    | LLM-extracted structured fields, keyed by `fb_posts.id`. |
| `tracker`         | `sql/tracker.sql`                            | Your personal shortlist (manual entry via the web UI). |

The extension also writes a `wa_messages` table (WhatsApp), created by the same
`supabase_schema.sql`; the current LLM pipeline reads only `fb_posts` so far.

- Raw-table semantics and data-quality gotchas (which dates/columns to trust):
  [`docs/DATA_MODEL.md`](docs/DATA_MODEL.md)
- Extracted field set for `listings_parsed`: [`docs/FIELD_SCHEMA.md`](docs/FIELD_SCHEMA.md)
- Scraper internals (FB/WhatsApp DOM, build notes):
  [`chrome_extension/KP-Rentals-Exporter/DEVELOPMENT.md`](chrome_extension/KP-Rentals-Exporter/DEVELOPMENT.md)

Keep `sql/listings_parsed.sql`, `src/extract.py` (`MODEL_FIELDS`), and
`analyze.build_parsed_row` in sync when changing the schema.

---

## Everyday workflow

1. **Scrape** new posts with the Chrome extension (they land in `fb_posts`).
2. **Extract:** `poetry run python src/analyze.py` (only new rows are parsed).
3. **Browse** the dashboard (`npm run dev` in `web/`), filter by price/area/long-term fit.
4. **Track** promising listings in the tracker table; enter price, visit date, contact.

---

## Testing

```bash
poetry run pytest            # Python pipeline (uses pythonpath=["src"] from pyproject.toml)
cd web && npm test           # web unit tests (vitest)
```

---

## Editor setup (VS Code)

`.vscode/settings.json` points at the in-project venv and wires up pytest. If the
Python extension does **not** auto-select the venv, run once:
**Cmd+Shift+P → "Python: Select Interpreter" → `./.venv/bin/python`**. (VS Code caches
the interpreter choice in per-workspace state, so `python.defaultInterpreterPath` is
only honored the first time — this is a known VS Code behavior, not a misconfiguration.)
Open the **`KP House Finder`** folder directly as the workspace so `.vscode/` applies.
