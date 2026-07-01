# KP Rentals Exporter — development notes

Hard-won internals for anyone **modifying the scraper**. For install/usage see
[`README.md`](README.md); for version history see [`CHANGELOG.md`](CHANGELOG.md).
Normal usage and the analysis pipeline don't need any of this.

## Architecture

- Site is auto-detected from the tab URL (Facebook vs WhatsApp Web); the popup
  shows the matching controls.
- Page-injected harvesters message the **service worker** (`background.js`),
  which owns `chrome.storage` and every Supabase REST call. Local storage is the
  always-on safety net; Supabase is the durable store.
- Worker message types: `kp_save`, `kp_load`, `kp_clear`, `kp_get_settings`,
  `kp_set_settings`, `kp_test_supabase`, `kp_sync_now`, `kp_sync_status`,
  `kp_get_range`, `kp_query_range`.
- **Sync:** upsert via Supabase REST (`Prefer: resolution=merge-duplicates` on
  `id`), debounced after save + a 1-min alarm + manual "Sync now". Manual sync
  also queries the DB for the current source's record count and date range.
- Supabase URL + anon key are stored in `chrome.storage.local`
  (`kp_supabase_settings = { url, anonKey, enabled }`), configured in the popup —
  never committed to a file.

## Facebook DOM (don't re-derive)

- Feed posts are direct children of `[role="feed"]`, **not** `[role="article"]`.
- The feed is virtualized (~4–17 posts rendered at once) — harvest *during* scroll.
- The grabber auto-scrolls `[role="feed"]`, expands "See more", and stops at a
  post-count limit and/or a cutoff date (the date stop works off the *recent,
  readable* relative times only).
- **Post dates for older posts are CSS-scrambled and NOT reliably decodable.**
  Every approach was tried and failed: relative-span decode, absolute-span by
  `order`, x-position, aria-label. The decoy `__cft__` link's spans are a *static*
  obfuscated UI string, not the date. Conclusion: read recent relative times
  (reliable), store `NULL` otherwise — never guess.
- Month-like false positives to reject: "Julien" (looks like Jul), "May be an
  image of…" (looks like May). The extractor uses strict month+day / relative
  regexes to filter these out.

## WhatsApp Web DOM (don't re-derive)

- Messages are `#main [data-id]` rows. `data-pre-plain-text` =
  `[HH:MM, D/M/YYYY] Sender:` — clean, no scrambling (source of `ts`/sender).
- Message text lives in the `.copyable-text` holder itself, **not**
  `span.selectable-text`.
- **Group name is the first `span[dir="auto"]` in the header.** The `span[title]`
  there holds the *participants list* (hundreds of names/phones) — do NOT use it.
- System notices (joins, etc.) lack `data-pre-plain-text` and are correctly skipped.
- Grab is **manual scroll**: auto-scroll was abandoned because WhatsApp's upward
  virtualization made message counts unstable. A background sweeper harvests what
  you scroll past; "Stop after N" / "Stop at date" act as on-panel *targets*, not
  hard stops.

## Build note

The dev container resets between sessions and `manifest.json` has repeatedly
reverted to a stale older version on restore. Always rewrite the full manifest
fresh and run the build-guard (verify `background.service_worker`, the `alarms`
permission, and the Supabase host permission) before zipping. The zip is built
**flat** — files at the root, manifest not in a subfolder.
