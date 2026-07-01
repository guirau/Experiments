# Data model — raw scraped tables

Semantics and data-quality realities of the two **raw** Supabase tables the
Chrome extension writes: `fb_posts` and `wa_messages`. The `CREATE TABLE` DDL
lives in [`chrome_extension/KP-Rentals-Exporter/supabase_schema.sql`](../chrome_extension/KP-Rentals-Exporter/supabase_schema.sql);
this file documents what the columns *mean* and which ones you can trust — the
part that shapes every query and the LLM extraction step.

> For the **structured** table produced downstream (`listings_parsed`), see
> [`FIELD_SCHEMA.md`](FIELD_SCHEMA.md).

## The value is in `text`

Both tables store raw, unstructured post/message content by design. Price,
bedrooms, location, contact, conditions, and lease length all live in the free
text — often multi-line, sometimes Thai, sometimes emoji-laden. There is **no**
structured price/beds/location column on the raw tables; it all has to be
extracted (that's what `src/extract.py` + `src/analyze.py` do).

## Column semantics worth knowing

### `fb_posts`
- `id` — canonical FB post id (`fbid_<postid>`), or a fallback (link / text-prefix)
  when the number can't be read. Link-variant differences never create duplicates.
- `source` — the group, e.g. `fb_2478665265528328`.
- `created_at` — **first-seen** time, preserved across re-syncs. This is the
  reliable time axis for FB (see below).
- `date_raw` — `YYYY-MM-DD` **only when FB exposes a readable date**, else `NULL`.

### `wa_messages`
- `id` — stable WhatsApp message id (`data-id`).
- `ts` — epoch millis, parsed from WhatsApp's clean `data-pre-plain-text`. Reliable.
- `datetime` — the raw `[HH:MM, D/M/YYYY]` string, stored as-is.
- `sender_phone` / `sender_name` — parsed from `data-pre-plain-text`.

## Data realities (these decide what's reliable to query)

**FB dates are partly unreliable — use `created_at` as the time axis.**
- `date_raw` is exact only for *recent* posts (FB shows relative times like
  "20h"/"2d", decoded reliably at capture time).
- For *older* posts FB's date is CSS-obfuscated and genuinely undecodable, so
  `date_raw` is stored as **`NULL`** rather than a guess.
- `created_at` (first-seen) is preserved across re-syncs and, for a
  regularly-grabbed group, closely tracks when a post appeared. **Prefer
  `created_at` for FB time analysis; treat `date_raw` as a bonus when present.**

**WA dates are reliable.** Use `ts` freely. Caveat: the source account formats
dates as `M/D/YYYY`; the parser disambiguates when a number is > 12.

**Dedup is by `id`, independent of dates.** Sync upserts with
`Prefer: resolution=merge-duplicates` on the `id` primary key, so re-grabbing
never creates duplicate rows. The DB is safe to re-sync repeatedly; analysis can
assume **one row per post/message**.

**Cross-source dedup is NOT done.** The same listing can appear in both a FB
group and a WhatsApp chat as two separate rows. A unified "listings" view (dedup
by contact phone or text similarity) is an analysis-layer task, not yet built.

## Verify the live DB before a big analysis run

Earlier dev iterations experimented with a `date_inferred` column that was later
reverted, so a long-lived DB may carry leftover columns or rows written under
older id/date schemes. Before trusting a bulk query, run a `select * limit 5` and
a per-`source` `count` on each table (or `python src/analyze.py --inspect`, which
does a live-schema check for `fb_posts`).
