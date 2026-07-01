# KP Rentals Exporter

One Chrome extension that exports rental listing posts to JSON from both
**Facebook groups** and **WhatsApp Web** group chats, for the Koh Phangan
rental finder. It detects which site the active tab is on and shows the right
controls. Runs only when you click it, in your own logged-in session.

## Install (once)
1. Put `manifest.json`, `popup.html`, `popup.js` in one folder (manifest at
   the top level — not in a subfolder).
2. Chrome → `chrome://extensions` → turn on "Developer mode".
3. "Load unpacked" → select the folder.
4. Pin it.

## Facebook (on a group feed)
1. Open the group, start near the top.
2. Click the extension. Set "Stop after N posts" and/or a cutoff date.
3. Click "Grab posts". It scrolls down, expands "See more", and harvests each
   post (text, permalink, decoded date), stopping at the count, the date, or
   the end of the feed.
4. Copies JSON to clipboard and/or downloads `fb_extension_YYYY-MM-DD.json`.

## WhatsApp (on a group chat at web.whatsapp.com)
1. Open the group chat.
2. Click the extension → "Start grabbing". A green panel appears on the page.
3. Scroll UP by hand as far back as you want; the panel shows a live message
   count and the oldest date reached.
4. Click "STOP & DOWNLOAD" on the panel. Copies JSON to clipboard and
   downloads `wa_<chat>_YYYY-MM-DD.json`.

## Feeding the parser
Drop the downloaded `.json` files into the rental scraper's `sources/` folder
and run `python parse_listings.py`. Files are auto-detected:
- `fb_*.json` → Facebook posts (text, link, date)
- `wa_*.json` → WhatsApp messages (grouped by sender)

## Notes
- Keep the tab in front while either mode runs.

## Running long scrapes on Mac (keep the laptop awake)
A long Facebook grab pauses if the Mac sleeps. To keep it awake, open Terminal
and run this before/while scraping, then press Ctrl+C when done:

    caffeinate -dimsu

Or auto-release after N seconds (e.g. 2 hours):

    caffeinate -dimsu -t 7200

Flags: -d display, -i system idle, -m disk, -s even on battery, -u user-active.
NOTE: closing the laptop lid still sleeps a MacBook regardless of caffeinate —
keep the lid open for an uninterrupted run (or use clamshell mode with an
external display + keyboard).

Safety net: even if the Mac sleeps or the tab crashes mid-scrape, every post
collected so far is already saved to storage. Just reopen the group and click
Grab — it shows "↻ Resuming from N saved posts" and continues.

## More notes
- Facebook dates are decoded from the page (best-effort; some recent posts
  show relative times). WhatsApp dates are exact.
- If a grab returns nothing after a site redesign: for Facebook the
  `[role="feed"]` selector and the `position:relative` date decode are the
  things to re-check; for WhatsApp it's the `data-pre-plain-text` /
  `.copyable-text` selectors.
