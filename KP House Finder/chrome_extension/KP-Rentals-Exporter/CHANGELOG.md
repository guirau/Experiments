# Changelog — KP Rentals Exporter

## 2.6.1
- Relative dates ("2d", "20h") are now resolved to a fixed YYYY-MM-DD at the
  moment of capture (in-page), instead of at sync time. This removes date drift:
  a post captured as "2d" is locked to that calendar date and stays correct no
  matter when it's synced. Absolute dates still resolve to their date; unreadable
  dates still store as NULL (exact-or-NULL, no guessing). created_at (first-seen)
  remains preserved across re-syncs as a reliable secondary signal.

## 2.6.0
(Built on the 2.4.0 base after reverting the 2.5.x id/inference changes.)
- id is the canonical post number (fbid_<postid>) for clean dedup.
- url holds the POST permalink (/groups/<g>/posts/<id>/); link keeps whatever
  reaches the post (photo URL ok). Every record has a reachable url/link.
- Fixed a dedup-merge bug that dropped id/link/url on near-duplicate fragments.
- Date extraction is strict: accepts only real relative times or month+day
  dates, rejects false positives (names like "Julien", "May be an image of…").
  Undecodable dates store as NULL — never a wrong guess, never today.


Versioning: MAJOR.MINOR.PATCH
- PATCH (x.x.+1): small fixes/tweaks, no behavior change
- MINOR (x.+1.0): new feature or meaningful behavior change
- MAJOR (+1.0.0): big rework / breaking change

## 2.5.1
- Hardened dedup so re-grabbing a Facebook group never creates duplicate DB
  rows. Post id is now a STABLE canonical id derived from the numeric post id in
  the link (posts/NNN, story_fbid, permalink, fbid, multi_permalinks → fbid_NNN),
  so the same post resolves to the same id even if FB serves a different link
  variant on a later grab. Falls back to cleaned link, then text prefix. Dedup is
  entirely independent of dates, so inferred/missing dates don't affect it.
  NOTE: this changes the id scheme — see migration note below.

## 2.5.0
- Dates: since Facebook's older-post dates are CSS-obfuscated and not reliably
  decodable, undated posts now get an INFERRED date from their position in the
  feed (the feed is strictly newest-first, so a post's date lies between its
  nearest dated neighbours — we use the midpoint). Exact dates are still used
  where FB gives a parseable one (recent relative times like "20h"/"2d").
  - New date_inferred boolean column marks whether a row's date_raw is exact or
    position-inferred. Schema updated; run the ALTER in supabase_schema.sql once
    on existing tables (sync auto-falls-back if the column isn't there yet).
  - Cutoff/stop and the panel range now use the effective (exact-or-inferred)
    date, so "stop at date" works for undated posts too.

## 2.4.1
- Reverted the aria-label date extraction from 2.4.0 (it grabbed wrong dates).

## 2.4.0
- Fixed FB dates being saved as "today" for clearly-dated posts. Root cause was
  a silent fallback: when a date couldn't be extracted/parsed, it was stamped as
  today, mislabeling posts that plainly showed e.g. "May 20".
  - resolveFbDate no longer fakes today: relative "XXh"/"XXd"/etc are computed
    from now, any absolute date ("May 20", "May 20 at 3:45 PM") is parsed as-is,
    and a genuinely unparseable/empty date stays NULL instead of today.
  - date_raw now stores the resolved YYYY-MM-DD, or the original raw string if it
    can't be resolved — never a faked today.
  - Date extraction now also reads the post's aria-label/title (clean, full date
    like "Tuesday, May 20, 2025 at 3:45 PM") before the fragile span-decode,
    capturing dates that were previously missed.
  - Panel / saved-state date range no longer counts unparseable dates as today
    (which corrupted the range); shows an "(N undated)" note instead.

## 2.3.3
- Fixed stuck "pending" records on sync. Root cause: a Supabase upsert rejects
  an entire chunk if the same id appears twice in it (duplicate text-hash ids
  from posts with identical text) — so one collision blocked up to 200 rows.
  Now rows are de-duplicated by id before each push, null-id rows are skipped
  cleanly, and if a chunk still fails it's retried row-by-row so good rows get
  through. Any genuine failures now show their exact reason in the popup instead
  of failing silently.

## 2.3.2
- Fixed WhatsApp group name for real, confirmed against the live DOM: the group
  name is the first span[dir="auto"] in the header, while span[title] holds the
  participants list (the opposite of the previous assumption). Now reads
  dir=auto. "Housing | Koh Phangan" → wa_housing_koh_phangan.

## 2.3.1
- Fixed WhatsApp group name: the reader was picking the participants subtitle
  (longest span) instead of the group title. Now takes the FIRST title span in
  the header (the name). "Housing | Koh Phangan" → wa_housing_koh_phangan.
  Normalization also strips leading/trailing underscores (handles emoji/symbols).

## 2.3.0
- WhatsApp source is now the actual group name (wa_<groupname>) instead of
  "wa_current". The chat name is read live from the conversation header with a
  more robust selector, on demand — so it works even before grabbing starts and
  correctly distinguishes different WhatsApp groups in Supabase. The grabber and
  the DB-range query use the same name, so sources match.

## 2.2.3
- Fixed DB record count / date range: "Sync now" now queries Supabase DIRECTLY
  for the current group's source every time (via kp_query_range), instead of
  depending on whether a local batch was processed during sync. Shows the exact
  source string queried and surfaces any DB error, so a source mismatch is
  immediately visible.

## 2.2.2
- "Sync now" / banner now shows the DB record COUNT for the current source
  alongside the date range ("DB records: N" + "DB dates: oldest → newest").

## 2.2.1
- Fixing DB date range: added nullslast to range ordering, a row-count check,
  and clearer diagnostics distinguishing "no rows for source" from "rows but
  unreadable dates" from a working range.

## 2.2.0
- "Sync now" now shows the date range (oldest → newest) of what's saved in the
  Supabase DB for the current source (FB group or WA chat). The worker queries
  the DB's true min/max once per sync (covers older sessions) and caches it.
- WhatsApp panel now warns "⚠ You're scrolling past <date> — already in the
  database" when you manually scroll older than the oldest message already
  saved in the DB for that chat, using the cached range (no per-scroll queries).

## 2.1.0
- WhatsApp now has "Stop after N messages" and "Stop at date" options, like
  Facebook. Because WhatsApp scrolling is manual (auto-scroll dropped messages
  in testing), these act as smart targets: the date filter keeps today back to
  the cutoff (older messages aren't saved), the count is a target, and the page
  panel shows a "✓ Target reached — you can stop scrolling" banner when hit.
- WhatsApp "How to use" rewritten to the concise Facebook style; added a
  "How these work" dropdown explaining the manual-scroll target behavior.

## 2.0.3
- Page panel: split the single button into three — STOP (halts the grab),
  EXPORT JSON (downloads), SYNC SUPABASE (pushes to Supabase now). Applied to
  both Facebook and WhatsApp panels. Stopping still saves everything locally;
  export and sync are now separate explicit actions.
- "How these work" last bullet updated to "press Stop on the page panel".
- WhatsApp panel export no longer copies to clipboard (download only),
  consistent with Facebook.

## 2.0.2
- Added a "⤴ Sync now" button to the Supabase banner: manually push all
  unsynced records to Supabase on demand, with live feedback (how many pushed)
  instead of waiting for the timer.

## 2.0.1
- Page panel: added a close (✕) button to dismiss the panel.
- Renamed the panel button to "Stop & Export" (Facebook) / "Stop & Download"
  (WhatsApp).
- Page panel now shows Supabase sync status (synced / pending counts, or
  "off"), not just local-save status.
- "How these work" rewritten as a clean bulleted list.

## 2.0.0
- v2 milestone: Supabase is the primary destination. The sync banner +
  settings now sit at the TOP of the popup, above the grab controls.
- date_raw always a resolved YYYY-MM-DD on sync (relative "1h"/"20h" computed,
  absolute dates normalized, undated → today; never a bare "1h").
- Prominent sync banner (on/paused dot + live Synced/Pending counts);
  credentials in a compact settings dropdown, remembered/pre-filled.
- JSON export demoted to a small "⬇ JSON backup" button.
- "↻ Resume grabbing (N saved)" button appears whenever there's saved data
  for the current group/chat — click to continue a stopped grab.
- Updated the Facebook how-to-use steps.

## 1.10.0
- date_raw is now always a resolved YYYY-MM-DD when syncing: relative times
  ("1h","20h") are computed to an actual date, absolute dates ("June 19...")
  are normalized, and undated posts resolve to today — never a bare "1h".
- UI now prioritizes Supabase: a prominent sync banner at the top shows
  on/paused state and live Synced/Pending counts. Credentials moved to a
  compact "⚙️ Supabase settings" dropdown (still remembered/pre-filled).
- JSON export demoted to a small "⬇ JSON backup" button (still available).
- New "↻ Resume grabbing (N saved)" button in the popup, shown whenever there
  are saved posts/messages for the current group/chat — click to continue a
  grab that stopped, picking up from what's saved.

## 1.9.0
- v2: optional Supabase sync. Posts/messages now sync to Supabase tables
  (fb_posts, wa_messages) in the background, on a timer and after each save,
  with upsert (no duplicates) keyed on the post id. Local storage + JSON export
  remain as the offline safety net underneath.
- New "Supabase sync" settings section in the popup: Project URL, anon key,
  on/off toggle (on by default), Save, and Test connection. Credentials are
  remembered across sessions and pre-filled automatically.
- Live synced/pending counts shown in the settings section.
- Added alarms permission and https://*.supabase.co host permission.
- Internal fields (_synced/_id) are stripped from JSON exports.
- See supabase_schema.sql for the table-creation SQL.

## 1.8.3
- Diagnostic save-status line on the Facebook panel; persist() falls back to
  direct chrome.storage write if the worker message fails; Export JSON falls
  back to direct storage read.

## 1.8.2
- Saved-data summary falls back to reading chrome.storage directly if the
  background worker isn't reachable, and surfaces the real error if it fails.

## 1.8.1
- Instructions and the "stop after / stop at date" explanation are now tucked
  into collapsible dropdowns, so the popup is cleaner by default.
- The popup now always shows, under the export/clear buttons, how many
  posts/messages are currently saved for this group/chat and their date range.
  Refreshes after Clear.
- New icon (isometric beach house with palms) at 16/48/128px.
- (Versioning note: from here, small refinements bump the patch number
  (0.0.1); only substantial features bump the minor (0.1.0).)

## 1.8.0
- FIXED resume/persistence: storage now runs through a real background
  service worker (background.js). The page-injected harvesters couldn't
  reliably write chrome.storage, so saving silently failed and resume never
  detected prior grabs — now fixed. (The extension also has an inspectable
  service worker in chrome://extensions now.)
- FIXED missing posts: lowered the minimum post-text length from 25 to 8
  characters so short/compact posts (e.g. Thai) are no longer dropped.
- Every post/message now carries a stable id; the worker dedupes by id when
  merging, so resume continues cleanly without duplicates.
- New icon (palm + house + beach badge) at 16/48/128px.
- Redesigned popup: tropical header with icon, card layout, modern inputs and
  buttons, refreshed instructions, "Export JSON" naming throughout.

## 1.7.0
- Fixed slow/broken STOP & EXPORT on Facebook: the button now exports
  immediately from collected posts instead of waiting for the scroll loop
  (which did nothing if scraping had already finished). Sleeps are now
  interruptible, so stopping mid-wait is near-instant.
- Popup: removed the "Copy JSON to clipboard" checkbox (and the clipboard
  copy on export). Renamed "Export saved" → "Export JSON"; it downloads the
  saved batch filtered to the cutoff date.
- A finished grab and the Export JSON button both just download the .json.

## 1.6.1
- Docs: added a "Running long scrapes on Mac" section to the README
  (caffeinate command to prevent sleep). No code change.

## 1.6.0
- Resume indicator: when a Facebook grab loads previously-saved posts for the
  group, the panel now shows "↻ Resuming from N saved posts". (Resume is
  automatic — every grab of a group loads that group's saved batch and
  continues; a crash needs no special handling, you just grab again.)
- Clarified the date model: the stop date means "grab from today back to this
  date." Export keeps everything from today to the cutoff; older posts are
  dropped; recent/undated posts are always kept. The batch still stores
  everything per group regardless of the export filter.
- (Groundwork for v2: posts persist per-group in storage, ready for a future
  Supabase sync that streams them to a database for dashboards.)

## 1.5.0
- Panel date display: posts/messages with no parseable date (e.g. very recent
  "1h"/"20h" relative posts, or blank) now count as today in the date-range
  display, instead of being skipped. ("20h" still resolves accurately and may
  show yesterday if 20h ago was the previous day.)
- Added a STOP & EXPORT button to the Facebook on-page panel: stop the grab
  at any time and export what's been collected. (WhatsApp already had Stop.)
  Works mid-run even during the slow wake-up waits; data is also saved
  continuously, so "Export saved" recovers it even if the popup was closed.

## 1.4.0
- Fixed "Stop after N posts": leaving it blank now means NO post limit (so a
  date-only grab runs until it reaches the date), instead of silently
  defaulting to 100. Blank count + date = stop at date; count + blank date =
  stop at count; both blank = whole feed; both set = whichever comes first.

## 1.3.0
- Resume fast-skip (Facebook): on resume, posts already saved are recognized
  by text and skipped — no re-expanding "See more", no re-processing. Scrolling
  back through the already-saved region is now fast; real work resumes only at
  genuinely new posts. (Facebook always reloads the feed from the top, so the
  scroll-through is unavoidable, but it's now cheap.)
- Verified the Facebook date-stop end to end: dates decode reliably from the
  feed (CSS-unscrambled), parse correctly ("June 19 at 4:16 PM" → 2026-06-19),
  and the cutoff stops at the right post. No change needed — confirmed working.

## 1.2.0
- Crash-resilience: both grabbers now stream harvested items to
  chrome.storage.local as they go, so a crash/reload never loses progress.
- Resume: re-grabbing the SAME group/chat continues from what's saved
  (deduped); a DIFFERENT group/chat gets its own separate batch.
- New "Export saved" and "Clear saved" buttons in each mode — export works
  even if a grab died mid-run (crash recovery).
- Facebook now shows an on-page status panel (count + date range + saved
  status), matching WhatsApp. WhatsApp panel now shows the full date range.
- Added storage + unlimitedStorage permissions.

## 1.1.0
- Facebook: rewrote scroll stop-detection to fix random early stops. A step
  with no new posts is now treated as a load pause, not the end: it does up to
  two firm wake-up jumps, and only counts a "real stall" if those fail AND the
  page is at the document bottom. Requires 6 real stalls in a row to stop.
  Longer waits (1.2s step, 2.6s wake-up).

## 1.0.0
- Initial combined extension: Facebook post grabber (auto-scroll, text +
  permalink + CSS-decoded date, stop by post count or cutoff date) and
  WhatsApp Web grabber (manual-scroll background sweeper, sender + exact
  timestamp). Site auto-detected. Tropical house icon.
