# KP Rentals — Project Handoff

A handoff summary for continuing this project in Claude Code. It covers what the
system is, the data it produces, the hard-won knowledge behind it, and where work
goes next (analyzing the data in Supabase).

---

## 1. What the project is

A pipeline for finding long-term rentals on Koh Phangan, Thailand. Listings live
in Facebook groups and WhatsApp group chats. The system scrapes those, stores raw
data in Supabase, and (next phase) analyzes it.

Three parts:

1. **KP Rentals Exporter** — a Chrome extension (MV3) that scrapes FB group posts
   and WhatsApp Web messages and syncs them to Supabase. **This is built and
   working (v2.6.1).**
2. **Supabase** — Postgres backend holding the raw scraped data (two tables).
3. **Analysis layer** — *not built yet.* This is the next phase: parsing the raw
   `text` into structured listing fields (price, beds, location, contact, etc.)
   and querying/filtering.

---

## 2. Supabase schemas (source of truth)

Project name:  **KP Rentals** . Access is via the REST API with the anon key, under
RLS policies allowing anon insert/select (normal pattern for a personal
client-side tool).

### `fb_posts`

| column         | type                               | notes                                                                 |
| -------------- | ---------------------------------- | --------------------------------------------------------------------- |
| `id`         | text PK                            | canonical post id `fbid_<postid>`, or fallback (link / text-prefix) |
| `source`     | text NOT NULL                      | which group, e.g.`fb_2478665265528328`                              |
| `text`       | text NOT NULL                      | **the post body — the rich, unstructured field**               |
| `link`       | text                               | a link that reaches the post (photo URL is acceptable)                |
| `date_raw`   | text                               | `YYYY-MM-DD`when readable, else**NULL**(see date caveat)      |
| `url`        | text                               | the post permalink `/groups/<g>/posts/<id>/`                        |
| `created_at` | timestamptz NOT NULL default now() | **first-seen time; preserved across re-syncs**                  |

Indexes: `source`, `created_at`.

### `wa_messages`

| column           | type                               | notes                                                        |
| ---------------- | ---------------------------------- | ------------------------------------------------------------ |
| `id`           | text PK                            | stable message id (WhatsApp `data-id`)                     |
| `source`       | text NOT NULL                      | which chat, e.g.`wa_housing_koh_phangan`                   |
| `chat`         | text                               | human-readable chat name                                     |
| `text`         | text NOT NULL                      | **the message body — the rich field**                 |
| `sender_phone` | text                               | parsed from `data-pre-plain-text`                          |
| `sender_name`  | text                               | parsed sender name                                           |
| `datetime`     | text                               | raw `[HH:MM, D/M/YYYY]`string, as-is                       |
| `ts`           | bigint                             | epoch millis if parsed, else NULL (**reliable**for WA) |
| `url`          | text                               | chat URL at grab time                                        |
| `created_at`   | timestamptz NOT NULL default now() | first-seen time                                              |

Indexes: `source`, `ts`.

The canonical schema file is `supabase_schema.sql` (STEP 1 creates tables, STEP 2
enables RLS + anon policies).

> ⚠️ Verify the live DB matches this before analysis. Earlier dev iterations
> experimented with a `date_inferred` column that was later reverted; a real DB
> may have leftover columns or rows written under older id/date schemes. Run a
> `select *` limit 5 per table and a per-source `count` first.

---

## 3. Data realities that shape analysis

These matter more than the column list — they determine what's reliable to query.

**The `text` column is where the value is.** Both tables store raw, unstructured
content by design. Price, bedrooms, location, contact, conditions, lease length
all live in free text (often multi-line, sometimes Thai, sometimes with emoji).
**The core analysis task is parsing `text` into structured fields.** There is no
structured price/beds/location column — that has to be extracted.

**FB dates are partly unreliable — use `created_at` as the time axis.**

* `date_raw` is exact only for *recent* posts (FB shows relative times like
  "20h"/"2d" which are decoded reliably; v2.6.1 resolves them at capture time so
  they don't drift).
* For *older* posts, FB's date is CSS-obfuscated and genuinely undecodable, so
  `date_raw` is  **NULL** . We deliberately store NULL rather than guess.
* `created_at` (first-seen) is preserved across re-syncs and is a reliable
  secondary time signal — for a regularly-grabbed group it closely tracks when a
  post appeared. **For time-based FB analysis, prefer `created_at`, treat
  `date_raw` as a bonus when present.**

**WA dates are reliable.** `ts` (epoch millis) comes from WhatsApp's clean
`data-pre-plain-text` attribute, no obfuscation. Use `ts` freely for WA time
analysis. (Caveat: date format is M/D/YYYY for the source account; the parser
disambiguates when a number > 12.)

**Dedup is by `id`, independent of dates.** Re-grabbing never creates duplicate
rows — the upsert merges on the `id` primary key. FB `id` is the canonical post
number (`fbid_<n>`) so link-variant differences don't create dupes. So the DB is
safe to re-sync repeatedly; analysis can assume one row per post/message.

**Cross-source dedup is NOT done.** The same listing can appear in both a FB group
and a WhatsApp chat as separate rows. If you want a unified "listings" view,
cross-source dedup (e.g. by contact phone or text similarity) is an analysis-layer
task, not yet built.

---

## 4. The extension (built, v2.6.1) — brief orientation

Only needed if you touch the scraper; the analysis phase mostly doesn't.

* **Files:** `manifest.json`, `popup.html`, `popup.js` (~1340 lines, the grabbers
  * UI), `background.js` (~370 lines, the service worker: storage + Supabase
    sync), `supabase_schema.sql`, `CHANGELOG.md`, icons.
* **Architecture:** site auto-detected from tab URL (FB / WhatsApp Web). Page-
  injected harvesters message the  **service worker** , which owns `chrome.storage`
  and all Supabase REST calls. Worker message types: `kp_save`, `kp_load`,
  `kp_clear`, `kp_get_settings`, `kp_set_settings`, `kp_test_supabase`,
  `kp_sync_now`, `kp_sync_status`, `kp_get_range`, `kp_query_range`.
* **FB grab:** auto-scrolls the `[role="feed"]`, harvests post text + link + date,
  stops at a post-count limit and/or a cutoff date (the stop works off the
  *recent, readable* relative dates).
* **WA grab:** manual scroll (auto-scroll was abandoned — WhatsApp's upward
  virtualization caused message-count instability). A background sweeper harvests
  what you scroll past; "Stop after N" / "Stop at date" act as smart *targets*
  with on-panel indicators.
* **Sync:** upsert via Supabase REST (`Prefer: resolution=merge-duplicates` on
  `id`), debounced after save + 1-min alarm + manual "Sync now". Manual sync also
  queries the DB directly for the current source's record count + date range.
* **Build note:** the dev container resets between sessions; the
  `manifest.json` in outputs has repeatedly reverted to a stale v1.2.0 on
  restore. Always rewrite the full manifest fresh and run the build-guard
  (verifies `background.service_worker`, `alarms` perm, supabase host) before
  zipping. The zip is built FLAT (files at root).

---

## 5. Hard-won knowledge (don't re-derive these)

**Facebook DOM:**

* Feed posts are direct children of `[role="feed"]`, NOT `[role="article"]`.
* The feed is virtualized (~4–17 rendered at once); harvest during scroll.
* Post date is CSS-scrambled in a decoy span block; it is  **not reliably
  decodable** . Multiple approaches were tried (relative-span decode, absolute-
  span-by-`order`, x-position, aria-label) — all failed for older posts. The
  decoy `__cft__` link's spans are a *static* obfuscated UI string, not the date.
  Conclusion: read the recent relative times (reliable), store NULL otherwise.
* Date strings that "look like" months can be false positives: "Julien" (Jul),
  "May be an image of…" (May). The extractor uses strict month+day / relative
  regexes to reject these.

**WhatsApp Web DOM:**

* Messages: `#main [data-id]` rows. `data-pre-plain-text` =
  `[HH:MM, D/M/YYYY] Sender:` — clean, no scrambling.
* Message text is in the `.copyable-text` holder itself, not `span.selectable-text`.
* **Group name is the first `span[dir="auto"]` in the header.** The `span[title]`
  there holds the *participants list* (hundreds of names/phones) — do NOT use it.
* System notices (joins, etc.) lack `data-pre-plain-text` → correctly skipped.

---

## 6. Next phase — analyzing the data (where to start)

The user wants to **start processing and analyzing the Supabase data.** Nothing
here is built yet. Suggested starting points (confirm goals with the user first):

1. **Inspect first.** Pull row counts per `source` and a `select * limit 10` from
   each table. Confirm live schema matches §2; note any leftover columns or
   old-scheme rows.
2. **Build a `text` → structured parser.** This is the core. Extract from the free
   text: `price_thb`, `rental_type` (offer vs wanted), `bedrooms`, `location`,
   `aircon`, `wifi`, `lease_min_months`, `contact`, `is_offer`, etc. The text is
   multilingual (English + Thai) and noisy. (An earlier iteration used an LLM
   extraction script with ~23 fields and a JSON-only prompt — that pattern works
   well here; incremental by `id`/hash so only new rows are parsed.)
3. **Decide the time axis.** FB → prefer `created_at`; WA → use `ts`.
4. **Optional: cross-source unified listings** via contact/text-similarity dedup.
5. **Output.** Likely a filtered/sorted view of current offers matching the
   user's criteria. Format TBD with the user (SQL views, a pandas/notebook
   workflow, or a small dashboard).

**Access pattern:** Supabase REST with the anon key, or direct Postgres
connection for analysis. For a Python workflow, `supabase-py` or plain `psycopg2`/
`httpx` against the REST endpoint both work. Recommend pulling into pandas for the
parsing/analysis rather than doing it all in SQL, given the heavy free-text work.

---

## 7. User working style (from the build history)

* Values **directness and honesty about limitations** over reassurance. When
  something can't be done reliably (e.g. FB date decoding), say so plainly rather
  than ship a clever guess that fails.
* **Tests with live diagnostics** — pasting real DOM/console output. Lean on that:
  inspect real data before committing to an approach.
* Iterative, detail-driven, catches regressions fast and is comfortable reverting.
* Prefers **honest NULL over fabricated data** (chose "exact date or NULL", no
  approximation).
* Increment a version on every change; keep a changelog.
