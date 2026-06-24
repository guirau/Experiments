// Background service worker for KP Rentals Exporter.
// Owns chrome.storage (the page-injected harvesters can't access it reliably)
// AND owns Supabase sync. Local storage is the always-on safety net; Supabase
// is the durable store. Posts are marked synced once Supabase accepts them.

const KEY = (batchKey) => "batch_" + batchKey;
const SETTINGS_KEY = "kp_supabase_settings";   // { url, anonKey, enabled }

// ---- settings helpers ----
async function getSettings() {
  const got = await chrome.storage.local.get(SETTINGS_KEY);
  return got[SETTINGS_KEY] || { url: "", anonKey: "", enabled: true };
}
async function setSettings(s) {
  await chrome.storage.local.set({ [SETTINGS_KEY]: s });
}

// Which Supabase table a batch goes to, based on source kind.
function tableFor(batch) {
  return batch.source === "whatsapp" ? "wa_messages" : "fb_posts";
}

// Map a stored item to a DB row for its table.
// Parse an FB date string to epoch millis (relative "1h"/"20h", absolute
// "June 19 at 4:16 PM", "yesterday", etc.). Returns null if unparseable.
function fbDateToTs(raw, now) {
  if (!raw) return null;
  if (/^\d{4}-\d{2}-\d{2}$/.test(String(raw))) return Date.parse(raw + "T00:00:00");
  const s = String(raw).trim().toLowerCase();
  let m = s.match(/^(\d{1,2})\s?(m|min|h|hr|hrs|hour|d|day|w|week)s?\b/);
  if (m) {
    const v = parseInt(m[1], 10), u = m[2]; let ms = 0;
    if (/^m/.test(u)) ms = v * 60e3; else if (/^h/.test(u)) ms = v * 3600e3;
    else if (/^d/.test(u)) ms = v * 86400e3; else if (/^w/.test(u)) ms = v * 7 * 86400e3;
    return now - ms;
  }
  if (/yesterday/.test(s)) return now - 86400e3;
  if (/today|just now/.test(s)) return now;
  const months = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec";
  if (!new RegExp("\\b(" + months + ")", "i").test(s)) return null;
  const cleaned = String(raw).replace(/\bat\b/i, "").replace(/\s+/g, " ").trim();
  const yr = new Date(now).getFullYear();
  let t = Date.parse(cleaned + " " + yr);
  if (isNaN(t)) return null;
  if (t > now + 86400e3) t = Date.parse(cleaned + " " + (yr - 1));
  return isNaN(t) ? null : t;
}

// Resolve any FB date string to a YYYY-MM-DD calendar date. Relative times
// ("1h","20h") and absolute ("June 19...") are both normalized; an undated
// post (newest-feed) resolves to today. Never returns a bare "1h".
// Resolve a FB date string to YYYY-MM-DD.
//   - relative "XXh"/"XXd"/"XXm"/"XXw" → computed from now
//   - "yesterday"/"today"/"just now"   → that day
//   - any absolute date ("May 20", "May 20 at 3:45 PM") → parsed as-is
// Returns null if the string is empty or genuinely unparseable — we do NOT
// fake "today", because that silently mislabels clearly-dated posts. The caller
// decides what to do with null (store null rather than a wrong date).
function resolveFbDate(raw, now) {
  const ts = fbDateToTs(raw, now);
  if (ts == null) return null;              // unknown → null, NOT today
  return new Date(ts).toISOString().slice(0, 10);
}

function rowFor(batch, batchKey, it) {
  if (batch.source === "whatsapp") {
    return {
      id: it.id || it._id,
      source: batchKey,
      chat: batch.chat || "",
      text: it.text || "",
      sender_phone: it.senderPhone || null,
      sender_name: it.senderName || null,
      datetime: it.datetime || null,
      ts: it.ts != null ? it.ts : null,
      url: batch.url || null,
    };
  }
  return {
    id: it.id,
    source: batchKey,
    text: it.text || "",
    link: it.link || it.url || null,                 // always a reachable link
    date_raw: resolveFbDate(it.date, Date.now()) || (it.date ? String(it.date) : null),
    url: it.url || it.link || batch.url || null,      // post permalink preferred
  };
}

// POST a batch of rows to a Supabase table with upsert (merge-duplicates on id).
async function pushRows(settings, table, rows) {
  const endpoint = settings.url.replace(/\/+$/, "") + "/rest/v1/" + table;
  const resp = await fetch(endpoint, {
    method: "POST",
    headers: {
      "apikey": settings.anonKey,
      "Authorization": "Bearer " + settings.anonKey,
      "Content-Type": "application/json",
      "Prefer": "resolution=merge-duplicates,return=minimal",
    },
    body: JSON.stringify(rows),
  });
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new Error("HTTP " + resp.status + " " + body.slice(0, 200));
  }
  return true;
}

// Sync all batches: push any items not yet marked synced, in chunks.
let syncing = false;
// Query Supabase for the oldest/newest saved date of a given source, and
// cache it locally so the panels can compare without re-querying each scroll.
// FB tables sort by date_raw (YYYY-MM-DD text); WA by ts (bigint).
const RANGE_KEY = "kp_db_range";   // { <source>: { oldest, newest, table } }

async function fetchDbRange(settings, table, source) {
  const base = settings.url.replace(/\/+$/, "") + "/rest/v1/" + table;
  const orderCol = table === "wa_messages" ? "ts" : "date_raw";
  const sel = table === "wa_messages" ? "ts,datetime" : "date_raw";
  const hdr = { "apikey": settings.anonKey, "Authorization": "Bearer " + settings.anonKey };
  const enc = encodeURIComponent(source);
  let count = null;
  async function one(dir) {
    const url = base + "?source=eq." + enc + "&select=" + sel +
                "&order=" + orderCol + "." + dir + ".nullslast&limit=1";
    const r = await fetch(url, { headers: hdr });
    if (!r.ok) {
      const body = await r.text().catch(() => "");
      throw new Error("HTTP " + r.status + " " + body.slice(0, 120));
    }
    const arr = await r.json();
    return arr && arr[0] ? arr[0] : null;
  }
  // also get a count of rows for this source, to tell "no rows" apart from "bad dates"
  try {
    const cr = await fetch(base + "?source=eq." + enc + "&select=id", {
      headers: { ...hdr, "Prefer": "count=exact", "Range": "0-0" },
    });
    const crange = cr.headers.get("content-range") || "";
    const m = crange.match(/\/(\d+)$/);
    if (m) count = parseInt(m[1], 10);
  } catch (_) {}
  const newest = await one("desc");
  const oldest = await one("asc");
  function toDate(row) {
    if (!row) return null;
    if (table === "wa_messages") {
      return row.ts != null ? new Date(row.ts).toISOString().slice(0, 10) : null;
    }
    return row.date_raw || null;
  }
  return { oldest: toDate(oldest), newest: toDate(newest), table, count, source };
}

async function refreshDbRange(settings, table, source) {
  try {
    const range = await fetchDbRange(settings, table, source);
    const got = await chrome.storage.local.get(RANGE_KEY);
    const map = got[RANGE_KEY] || {};
    map[source] = range;
    await chrome.storage.local.set({ [RANGE_KEY]: map });
    return range;
  } catch (e) {
    return null;
  }
}

async function syncAll() {
  if (syncing) return { ok: true, skipped: "already running" };
  const settings = await getSettings();
  if (!settings.enabled || !settings.url || !settings.anonKey) {
    return { ok: false, error: "sync disabled or not configured" };
  }
  syncing = true;
  let pushed = 0, failed = 0;
  const ranges = {};
  const errors = [];        // collect real failure reasons
  try {
    const all = await chrome.storage.local.get(null);
    for (const [key, batch] of Object.entries(all)) {
      if (!key.startsWith("batch_") || !batch || !Array.isArray(batch.items)) continue;
      const batchKey = key.slice("batch_".length);
      const table = tableFor(batch);
      const pending = batch.items.filter((it) => !it._synced && (it.id || it._id));

      // chunk to keep requests modest
      const CHUNK = 200;
      let anyOk = false;
      for (let i = 0; i < pending.length; i += CHUNK) {
        const slice = pending.slice(i, i + CHUNK);
        // Build rows and DEDUPE by id within this batch — Supabase upsert
        // rejects a whole chunk if the same id appears twice ("ON CONFLICT DO
        // UPDATE cannot affect row a second time"). Keep the last occurrence.
        const seen = new Map();
        for (const it of slice) {
          const row = rowFor(batch, batchKey, it);
          if (row.id == null || row.id === "") continue;   // skip null-id rows
          seen.set(row.id, { row, it });
        }
        const rows = Array.from(seen.values()).map((x) => x.row);
        const itemsForRows = Array.from(seen.values()).map((x) => x.it);
        if (rows.length === 0) { for (const it of slice) it._synced = true; continue; }
        try {
          await pushRows(settings, table, rows);
          for (const it of slice) it._synced = true;   // whole slice done
          pushed += rows.length;
          anyOk = true;
        } catch (e) {
          const chunkMsg = (e && e.message ? e.message : String(e));
          if (errors.length < 5) errors.push(batchKey + " [" + table + "] chunk: " + chunkMsg);
          for (let j = 0; j < rows.length; j++) {
            try {
              await pushRows(settings, table, [rows[j]]);
              itemsForRows[j]._synced = true;
              pushed += 1;
              anyOk = true;
            } catch (e2) {
              failed += 1;
              const m = (e2 && e2.message ? e2.message : String(e2));
              if (errors.length < 8) {
                errors.push("row id=" + (rows[j] && rows[j].id) + ": " + m.slice(0, 160));
              }
            }
          }
        }
      }
      if (anyOk) {
        await chrome.storage.local.set({ [key]: batch });
      }
      const range = await refreshDbRange(settings, table, batchKey);
      if (range) ranges[batchKey] = range;
    }
    return { ok: true, pushed, failed, ranges, errors };
  } catch (e) {
    return { ok: false, error: String(e), errors };
  } finally {
    syncing = false;
  }
}

// Debounced sync trigger (after saves) + periodic alarm.
let syncTimer = null;
function scheduleSync(delayMs = 4000) {
  if (syncTimer) clearTimeout(syncTimer);
  syncTimer = setTimeout(() => { syncTimer = null; syncAll(); }, delayMs);
}
chrome.alarms.create("kp_sync", { periodInMinutes: 1 });
chrome.alarms.onAlarm.addListener((a) => { if (a.name === "kp_sync") syncAll(); });

// Test the connection: a HEAD/GET against one table.
async function testConnection(settings) {
  try {
    const endpoint = settings.url.replace(/\/+$/, "") + "/rest/v1/fb_posts?select=id&limit=1";
    const resp = await fetch(endpoint, {
      headers: { "apikey": settings.anonKey, "Authorization": "Bearer " + settings.anonKey },
    });
    if (resp.ok) return { ok: true };
    const body = await resp.text().catch(() => "");
    return { ok: false, error: "HTTP " + resp.status + " " + body.slice(0, 200) };
  } catch (e) {
    return { ok: false, error: String(e) };
  }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    try {
      if (msg.type === "kp_save") {
        const key = KEY(msg.batchKey);
        const got = await chrome.storage.local.get(key);
        const existing = (got[key] && got[key].items) || [];
        const byId = new Map();
        for (const it of existing) byId.set(it.id || it._id, it);
        for (const it of msg.items) {
          const id = it.id || it._id;
          const prev = byId.get(id);
          // preserve _synced flag if we already had this item
          byId.set(id, prev && prev._synced ? { ...it, _synced: true } : it);
        }
        const merged = Array.from(byId.values());
        const obj = {};
        obj[key] = {
          source: msg.source || (got[key] && got[key].source) || "",
          url: msg.url || (got[key] && got[key].url) || "",
          chat: msg.chat || (got[key] && got[key].chat) || "",
          updated_at: new Date().toISOString(),
          items: merged,
        };
        await chrome.storage.local.set(obj);
        scheduleSync();                       // push to Supabase shortly after
        sendResponse({ ok: true, count: merged.length });
        return;
      }

      if (msg.type === "kp_load") {
        const key = KEY(msg.batchKey);
        const got = await chrome.storage.local.get(key);
        sendResponse({ ok: true, batch: got[key] || null });
        return;
      }

      if (msg.type === "kp_clear") {
        await chrome.storage.local.remove(KEY(msg.batchKey));
        sendResponse({ ok: true });
        return;
      }

      if (msg.type === "kp_get_settings") {
        sendResponse({ ok: true, settings: await getSettings() });
        return;
      }

      if (msg.type === "kp_set_settings") {
        await setSettings(msg.settings);
        sendResponse({ ok: true });
        return;
      }

      if (msg.type === "kp_test_supabase") {
        sendResponse(await testConnection(msg.settings || (await getSettings())));
        return;
      }

      if (msg.type === "kp_sync_now") {
        sendResponse(await syncAll());
        return;
      }

      if (msg.type === "kp_query_range") {
        const settings = await getSettings();
        if (!settings.url || !settings.anonKey) {
          sendResponse({ ok: false, error: "not configured" });
          return;
        }
        try {
          const range = await fetchDbRange(settings, msg.table || "fb_posts", msg.batchKey);
          // also cache it for the WA scroll-warning feature
          const got = await chrome.storage.local.get(RANGE_KEY);
          const map = got[RANGE_KEY] || {};
          map[msg.batchKey] = range;
          await chrome.storage.local.set({ [RANGE_KEY]: map });
          sendResponse({ ok: true, range });
        } catch (e) {
          sendResponse({ ok: false, error: String(e) });
        }
        return;
      }

      if (msg.type === "kp_get_range") {
        const got = await chrome.storage.local.get(RANGE_KEY);
        const map = got[RANGE_KEY] || {};
        sendResponse({ ok: true, range: map[msg.batchKey] || null });
        return;
      }

      if (msg.type === "kp_sync_status") {
        // count synced vs pending across all batches
        const all = await chrome.storage.local.get(null);
        let synced = 0, pending = 0;
        for (const [key, batch] of Object.entries(all)) {
          if (!key.startsWith("batch_") || !batch || !Array.isArray(batch.items)) continue;
          for (const it of batch.items) (it._synced ? synced++ : pending++);
        }
        sendResponse({ ok: true, synced, pending });
        return;
      }

      sendResponse({ ok: false, error: "unknown message type" });
    } catch (e) {
      sendResponse({ ok: false, error: String(e) });
    }
  })();
  return true;
});
