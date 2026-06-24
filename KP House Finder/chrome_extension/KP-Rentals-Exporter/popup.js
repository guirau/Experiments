const $ = (id) => document.getElementById(id);
const status = (msg) => { $("status").textContent = msg; };

// show extension version (read from manifest so it never goes stale)
try {
  const v = chrome.runtime.getManifest().version;
  const el = $("ver");
  if (el) el.textContent = "v" + v;
} catch (_) {}

// Detect which site the active tab is on, then show the matching panel.
(async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const url = (tab && tab.url) || "";
  if (/web\.whatsapp\.com/.test(url)) {
    $("wa").classList.remove("hidden");
    $("siteLabel").textContent = "WhatsApp Web";
    wireWhatsApp(tab);
    showSavedState(tab, "whatsapp");
  } else if (/facebook\.com/.test(url)) {
    $("fb").classList.remove("hidden");
    $("siteLabel").textContent = "Facebook";
    wireFacebook(tab);
    showSavedState(tab, "facebook");
  } else {
    $("none").classList.remove("hidden");
    $("siteLabel").textContent = "";
  }
})();

// ---------------- Supabase settings UI ----------------
(async () => {
  const urlEl = $("supaUrl"), keyEl = $("supaKey"), enEl = $("supaEnabled");
  const stateEl = $("supaState"), statusEl = $("supaStatus");
  if (!urlEl) return;

  // Load saved settings and pre-fill (remembered across sessions).
  let settings = { url: "", anonKey: "", enabled: true };
  try {
    const resp = await chrome.runtime.sendMessage({ type: "kp_get_settings" });
    if (resp && resp.ok && resp.settings) settings = resp.settings;
  } catch (_) {}
  urlEl.value = settings.url || "";
  keyEl.value = settings.anonKey || "";
  enEl.checked = settings.enabled !== false;

  function paintState() {
    const configured = !!(urlEl.value.trim() && keyEl.value.trim());
    const dot = $("supaDot"), bannerText = $("supaBannerText"), counts = $("supaCounts");
    if (!configured) {
      stateEl.textContent = "not set up";
      if (dot) dot.className = "supa-dot";
      if (bannerText) bannerText.textContent = "Supabase — not configured";
      if (counts) counts.innerHTML = '<span style="opacity:.8">Open ⚙️ settings below to connect.</span>';
      return;
    }
    const on = enEl.checked;
    stateEl.textContent = on ? "syncing on" : "paused";
    if (dot) dot.className = "supa-dot " + (on ? "on" : "off");
    if (bannerText) bannerText.textContent = on ? "Supabase — syncing" : "Supabase — paused";
  }
  paintState();

  async function refreshSyncStatus() {
    try {
      const r = await chrome.runtime.sendMessage({ type: "kp_sync_status" });
      if (r && r.ok) {
        const counts = $("supaCounts");
        if (counts && (urlEl.value.trim() && keyEl.value.trim())) {
          counts.innerHTML =
            '<span class="synced">Synced <b>' + r.synced + '</b></span>' +
            '<span class="pending">Pending <b>' + r.pending + '</b></span>';
        }
        // also keep the in-settings status box in sync
        if (statusEl && statusEl.textContent.indexOf("saved") === -1) {
          statusEl.className = "saved";
          statusEl.innerHTML =
            '<div class="row"><span class="lbl">Synced</span><span class="val">' + r.synced + '</span></div>' +
            '<div class="row"><span class="lbl">Pending</span><span class="val">' + r.pending + '</span></div>';
        }
      }
    } catch (_) {}
  }
  if (settings.url && settings.anonKey) refreshSyncStatus();

  async function save() {
    settings = { url: urlEl.value.trim(), anonKey: keyEl.value.trim(), enabled: enEl.checked };
    await chrome.runtime.sendMessage({ type: "kp_set_settings", settings });
    paintState();
    statusEl.className = "saved empty";
    statusEl.textContent = "Settings saved.";
  }
  $("supaSave").addEventListener("click", save);
  enEl.addEventListener("change", save);   // toggling on/off saves immediately

  $("supaTest").addEventListener("click", async () => {
    statusEl.className = "saved empty";
    statusEl.textContent = "Testing…";
    const s = { url: urlEl.value.trim(), anonKey: keyEl.value.trim(), enabled: enEl.checked };
    await chrome.runtime.sendMessage({ type: "kp_set_settings", settings: s });
    const r = await chrome.runtime.sendMessage({ type: "kp_test_supabase", settings: s });
    if (r && r.ok) {
      statusEl.className = "saved";
      statusEl.innerHTML = '<div style="color:#1a7f37;font-weight:600">✓ Connected — tables reachable.</div>';
      refreshSyncStatus();
    } else {
      statusEl.className = "saved empty";
      statusEl.textContent = "✗ " + ((r && r.error) || "connection failed");
    }
  });

  // Manual "Sync now": push all unsynced records to Supabase immediately.
  const syncNowBtn = $("supaSyncNow");
  if (syncNowBtn) syncNowBtn.addEventListener("click", async () => {
    if (!urlEl.value.trim() || !keyEl.value.trim()) {
      statusEl.className = "saved empty";
      statusEl.textContent = "Set your Supabase URL and key first.";
      $("supaWrap").open = true;
      return;
    }
    const label = syncNowBtn.textContent;
    syncNowBtn.disabled = true;
    syncNowBtn.textContent = "Syncing…";
    try {
      const r = await chrome.runtime.sendMessage({ type: "kp_sync_now" });
      if (r && r.ok) {
        syncNowBtn.textContent = "✓ Synced " + (r.pushed || 0)
          + (r.failed ? " (" + r.failed + " failed)" : "");
        await showDbRange(r.ranges);
        // If anything failed, surface the real reason(s) in the status box.
        if (r.failed && r.errors && r.errors.length) {
          statusEl.className = "saved";
          statusEl.innerHTML = '<div style="color:#b3261e;font-weight:600;margin-bottom:4px">' +
            r.failed + ' failed to sync. Reasons:</div>' +
            '<div style="font-size:11px;white-space:pre-wrap;word-break:break-word">' +
            r.errors.map((x) => "• " + x).join("\n") + '</div>';
          $("supaWrap").open = true;
        }
      } else {
        syncNowBtn.textContent = "✗ " + ((r && r.error) || "sync failed");
        if (r && r.errors && r.errors.length) {
          statusEl.className = "saved";
          statusEl.innerHTML = '<div style="font-size:11px;white-space:pre-wrap">' +
            r.errors.map((x) => "• " + x).join("\n") + '</div>';
          $("supaWrap").open = true;
        }
      }
    } catch (e) {
      syncNowBtn.textContent = "✗ " + (e && e.message ? e.message : "sync failed");
    }
    await refreshSyncStatus();
    setTimeout(() => { syncNowBtn.disabled = false; syncNowBtn.textContent = label; }, 2500);
  });

  // Resolve the current tab's batch key and show its DB date range in the banner.
  async function showDbRange(ranges) {
    const rangeEl = $("supaRange");
    if (!rangeEl) return;
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      let bk = null, table = "fb_posts";
      if (tab && /web\.whatsapp\.com/.test(tab.url || "")) {
        table = "wa_messages";
        bk = await waBatchKeyForTab(tab);
      } else if (tab) {
        bk = batchKeyForUrl(tab.url || "");
      }
      if (!bk) { rangeEl.textContent = ""; return; }

      // Query the DB directly for THIS source every time — don't depend on
      // whether a local batch was processed during sync.
      rangeEl.textContent = "Reading DB for " + bk + "…";
      const rr = await chrome.runtime.sendMessage({
        type: "kp_query_range", batchKey: bk, table,
      });
      if (!rr || !rr.ok) {
        rangeEl.textContent = "DB read error: " + ((rr && rr.error) || "unknown")
          + " (source " + bk + ")";
        return;
      }
      const range = rr.range || {};
      if (range.count != null && range.count > 0) {
        rangeEl.innerHTML =
          "DB records (" + bk + "): <b>" + range.count + "</b><br>" +
          "DB dates: <b>" + (range.oldest || "—") + "</b> → <b>" + (range.newest || "—") + "</b>";
      } else if (range.count === 0) {
        rangeEl.textContent = "DB has 0 records for source “" + bk + "”. "
          + "(Check the source column in your table.)";
      } else {
        rangeEl.textContent = "DB: no count returned for " + bk;
      }
    } catch (e) {
      rangeEl.textContent = "DB read failed: " + (e && e.message ? e.message : e);
    }
  }
  // populate on open too if we already have a cached range
  showDbRange(null);
})();

// Load the saved batch for the current group/chat and render a small summary
// (item count + date range) under the export/clear buttons. Refreshable.
async function showSavedState(tab, mode) {
  const boxId = mode === "facebook" ? "fbsaved" : "wasaved";
  const box = $(boxId);
  if (!box) return;
  try {
    let bk;
    if (mode === "whatsapp") {
      bk = await waBatchKeyForTab(tab);
    } else {
      bk = batchKeyForUrl(tab.url || "");
    }
    let batch = null;
    try {
      const resp = await chrome.runtime.sendMessage({ type: "kp_load", batchKey: bk });
      batch = resp && resp.ok ? resp.batch : null;
    } catch (err) {
      // Worker not reachable — fall back to reading storage directly (the
      // popup context can access chrome.storage).
      try {
        const got = await chrome.storage.local.get("batch_" + bk);
        batch = got["batch_" + bk] || null;
      } catch (_) {}
    }
    // Belt-and-suspenders: if the message resolved but returned nothing, also
    // try direct storage before concluding it's empty.
    if (!batch) {
      try {
        const got = await chrome.storage.local.get("batch_" + bk);
        batch = got["batch_" + bk] || null;
      } catch (_) {}
    }
    const items = (batch && batch.items) || [];
    const resumeBtn = $(mode === "facebook" ? "fbresume" : "waresume");
    if (items.length === 0) {
      box.className = "saved empty";
      box.textContent = mode === "facebook"
        ? "No saved posts for this group yet."
        : "No saved messages for this chat yet.";
      if (resumeBtn) resumeBtn.classList.add("hidden");
      return;
    }
    // date range
    const nowTs = Date.now();
    let oldest = null, newest = null;
    for (const it of items) {
      let ts;
      if (mode === "facebook") ts = fbDateToTs(it.date, nowTs);
      else ts = it.ts != null ? it.ts : null;
      if (ts == null) continue;          // unparseable → don't fake today, skip
      if (oldest == null || ts < oldest) oldest = ts;
      if (newest == null || ts > newest) newest = ts;
    }
    const fmt = (t) => new Date(t).toISOString().slice(0, 10);
    const noun = mode === "facebook" ? "posts" : "messages";
    box.className = "saved";
    if (resumeBtn) {
      resumeBtn.classList.remove("hidden");
      resumeBtn.textContent = "↻ Resume grabbing (" + items.length + " saved)";
    }
    box.innerHTML =
      '<div class="row"><span class="lbl">Saved ' + noun + '</span>' +
      '<span class="val">' + items.length + '</span></div>' +
      '<div class="range">' + fmt(oldest) + ' &rarr; ' + fmt(newest) + '</div>';
  } catch (e) {
    box.className = "saved empty";
    box.textContent = "Couldn't read saved data: " + (e && e.message ? e.message : String(e));
  }
}

/* ===================== Facebook mode ===================== */
// Popup-side copy of the date parser, for filtering exports by cutoff.
// (The in-page parseFbDate isn't reachable from the popup context.)
function fbDateToTs(raw, now) {
  if (!raw) return null;
  const iso = String(raw).match(/^\d{4}-\d{2}-\d{2}$/);
  if (iso) return Date.parse(raw + "T00:00:00");
  const s = raw.trim().toLowerCase();
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
  const cleaned = raw.replace(/\bat\b/i, "").replace(/\s+/g, " ").trim();
  const yr = new Date(now).getFullYear();
  let t = Date.parse(cleaned + " " + yr);
  if (isNaN(t)) return null;
  if (t > now + 86400e3) t = Date.parse(cleaned + " " + (yr - 1));
  return isNaN(t) ? null : t;
}

// Derive a stable batch key from a tab URL: fb_<groupId> or wa_<chatId-ish>.
// Injected into the WhatsApp tab to read the CURRENT chat name from the
// conversation header, robustly (tries several selectors). Returns "" if none.
function readWaChatNameInPage() {
  // The conversation header is the <header> inside #main. Confirmed structure:
  // the group NAME is the first span[dir="auto"] (e.g. "Housing | Koh Phangan"),
  // while span[title] holds the PARTICIPANTS list — so we must read dir=auto,
  // NOT span[title].
  const main = document.querySelector("#main");
  if (!main) return "";
  const header = main.querySelector("header");
  if (!header) return "";

  // Group name = first dir=auto span in the header.
  const spans = Array.from(header.querySelectorAll('span[dir="auto"]'))
    .map((s) => (s.textContent || "").trim())
    .filter(Boolean);
  if (spans.length) return spans[0];

  // Last-resort fallback: a heading element, if present.
  const h = header.querySelector('h1,h2,[role="heading"]');
  return h ? (h.textContent || "").trim() : "";
}

// Resolve the WhatsApp batch key for the current tab by reading the chat name
// live (works even before grabbing has started). Falls back to wa_current.
async function waBatchKeyForTab(tab) {
  try {
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: readWaChatNameInPage,
    });
    const name = (result || "").trim();
    if (name) {
      return "wa_" + name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 40);
    }
  } catch (_) {}
  return "wa_current";
}

function batchKeyForUrl(url) {
  const fb = url.match(/facebook\.com\/groups\/(\d+)/);
  if (fb) return "fb_" + fb[1];
  if (/web\.whatsapp\.com/.test(url)) return "wa_current";  // refined in-page
  const any = url.match(/facebook\.com\/([^/?#]+)/);
  return any ? "fb_" + any[1] : "fb_unknown";
}

function wireFacebook(tab) {
  const fbResume = $("fbresume");
  if (fbResume) fbResume.addEventListener("click", () => $("fbgrab").click());
  $("fbgrab").addEventListener("click", async () => {
    const maxRaw = parseInt($("maxposts").value, 10);
    const opts = {
      maxPosts: Number.isFinite(maxRaw) && maxRaw > 0 ? maxRaw : Infinity,
      cutoffDate: $("cutoff").value || null,
      batchKey: batchKeyForUrl(tab.url || ""),
    };

    $("fbgrab").disabled = true;
    status("Working… progress is saved as it goes.\nSafe to reopen this popup; a crash\nwon't lose collected posts.");

    let result;
    try {
      const [{ result: r }] = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: grabPostsInPage,
        args: [opts],
      });
      result = r;
    } catch (e) {
      status("Injection failed: " + e.message
        + "\n(Collected posts are still saved — click Export saved.)");
      $("fbgrab").disabled = false;
      return;
    }

    if (!result || !result.posts || result.posts.length === 0) {
      status("No posts found. Try scrolling more, or check you're on a group feed.");
      $("fbgrab").disabled = false;
      return;
    }

    const cleanPosts = result.posts.map((p) => {
      const c = {};
      for (const k in p) if (!k.startsWith("_")) c[k] = p[k];
      return c;
    });
    const payload = {
      grabbed_at: new Date().toISOString(),
      url: tab.url,
      count: cleanPosts.length,
      posts: cleanPosts,
    };
    const json = JSON.stringify(payload, null, 2);
    const dataUrl = "data:application/json;charset=utf-8," + encodeURIComponent(json);
    const fname = "fb_extension_" + new Date().toISOString().slice(0, 10) + ".json";
    chrome.downloads.download({ url: dataUrl, filename: fname, saveAs: true });

    status(`Done: ${result.posts.length} posts grabbed\nDownload started.`);
    $("fbgrab").disabled = false;
  });

  wireExportControls(tab, "facebook");
}

/* ===================== WhatsApp mode ===================== */
function wireWhatsApp(tab) {
  const waResume = $("waresume");
  if (waResume) waResume.addEventListener("click", () => $("wagrab").click());
  $("wagrab").addEventListener("click", async () => {
    const maxRaw = parseInt($("wamax").value, 10);
    const opts = {
      maxMsgs: Number.isFinite(maxRaw) && maxRaw > 0 ? maxRaw : Infinity,
      cutoffDate: $("wacutoff").value || null,
    };
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: startSweeper,
        args: [opts],
      });
      status("Sweeper started. Scroll UP by hand.\nThe panel shows progress and when a\ntarget is reached. Saves continuously.");
    } catch (e) {
      status("Injection failed: " + e.message);
    }
  });

  wireExportControls(tab, "whatsapp");
}

/* ===================== Export / manage saved batches ===================== */
// Reads the saved batch for the current tab from chrome.storage.local and
// exports it to JSON, independent of whether a grab is running. This is the
// crash-recovery path: even if a grab died, whatever was saved is exportable.
function wireExportControls(tab, mode) {
  const exportBtn = $(mode === "facebook" ? "fbexport" : "waexport");
  const clearBtn = $(mode === "facebook" ? "fbclear" : "waclear");

  async function currentBatchKey() {
    if (mode === "whatsapp") {
      return await waBatchKeyForTab(tab);
    }
    return batchKeyForUrl(tab.url || "");
  }

  if (exportBtn) exportBtn.addEventListener("click", async () => {
    const bk = await currentBatchKey();
    let batch = null;
    try {
      const resp = await chrome.runtime.sendMessage({ type: "kp_load", batchKey: bk });
      batch = resp && resp.ok ? resp.batch : null;
    } catch (_) {}
    if (!batch) {
      try {
        const got = await chrome.storage.local.get("batch_" + bk);
        batch = got["batch_" + bk] || null;
      } catch (_) {}
    }
    if (!batch || !batch.items || batch.items.length === 0) {
      status("Nothing saved to export for this group yet.");
      return;
    }
    const isWa = batch.source === "whatsapp";
    let items = batch.items;
    // For Facebook, apply the same "today back to cutoff" date filter as a
    // live grab, using the popup's current cutoff field.
    if (!isWa) {
      const cutoff = $("cutoff") && $("cutoff").value
        ? Date.parse($("cutoff").value + "T00:00:00") : null;
      if (cutoff != null) {
        const nowTs = Date.now();
        items = items.filter((p) => {
          const ts = fbDateToTs(p.date, nowTs);
          return ts == null || ts >= cutoff;
        });
      }
    }
    const payload = {
      grabbed_at: new Date().toISOString(),
      url: batch.url || (tab.url || ""),
      chat: batch.chat || "",
      count: items.length,
    };
    payload[isWa ? "messages" : "posts"] = items.map((it) => {
      const clean = {};
      for (const k in it) if (!k.startsWith("_")) clean[k] = it[k];
      return clean;
    });
    const json = JSON.stringify(payload, null, 2);
    const stamp = new Date().toISOString().slice(0, 10);
    const safe = (batch.chat || "").replace(/[^a-z0-9]+/gi, "_").slice(0, 30);
    const fname = isWa
      ? "wa_" + (safe || "chat") + "_" + stamp + ".json"
      : "fb_extension_" + stamp + ".json";
    const dataUrl = "data:application/json;charset=utf-8," + encodeURIComponent(json);
    chrome.downloads.download({ url: dataUrl, filename: fname, saveAs: true });
    status("Exported " + items.length + " items.\nDownload started.");
  });

  if (clearBtn) clearBtn.addEventListener("click", async () => {
    const bk = await currentBatchKey();
    await chrome.runtime.sendMessage({ type: "kp_clear", batchKey: bk });
    status("Cleared saved data for this group.");
    showSavedState(tab, mode);   // refresh the saved summary (now empty)
  });
}

/* ===================== Injected: Facebook ===================== */

async function grabPostsInPage(opts) {
  // Sleep that wakes early if STOP is pressed, so stopping is instant even
  // mid-wait. Polls the flag every 100ms instead of one long timer.
  const sleep = (ms) => new Promise((resolve) => {
    const start = Date.now();
    const tick = () => {
      if (window.__kpStop || Date.now() - start >= ms) resolve();
      else setTimeout(tick, 100);
    };
    tick();
  });
  const batchKey = opts.batchKey;

  // Persist current progress to chrome.storage.local each cycle, so a crash
  // never loses what's been harvested. Stored deduped by post link+text.
  // Persist current progress via the background service worker (the page
  // context can't be trusted to write chrome.storage directly). Sends the
  // full current set each cycle; the worker dedupes by id and saves.
  let __saveInfo = "saving locally…";
  let __syncInfo = "";
  async function persist(items) {
    if (!batchKey) { __saveInfo = "⚠ no batch key"; return; }
    try {
      const resp = await chrome.runtime.sendMessage({
        type: "kp_save",
        batchKey,
        source: "facebook",
        url: location.href,
        items,
      });
      if (resp && resp.ok) __saveInfo = "✓ saved locally (" + resp.count + ")";
      else __saveInfo = "⚠ local save issue";
    } catch (e) {
      try {
        const key = "batch_" + batchKey;
        const got = await chrome.storage.local.get(key);
        const existing = (got[key] && got[key].items) || [];
        const byId = new Map();
        for (const it of existing) byId.set(it.id, it);
        for (const it of items) byId.set(it.id, it);
        const merged = Array.from(byId.values());
        const obj = {}; obj[key] = { source: "facebook", url: location.href, updated_at: new Date().toISOString(), items: merged };
        await chrome.storage.local.set(obj);
        __saveInfo = "✓ saved locally (" + merged.length + ")";
      } catch (e2) {
        __saveInfo = "⚠ save failed";
      }
    }
    // ask the worker for Supabase sync status (synced/pending across batches)
    try {
      const s = await chrome.runtime.sendMessage({ type: "kp_sync_status" });
      const st = await chrome.runtime.sendMessage({ type: "kp_get_settings" });
      const enabled = st && st.ok && st.settings && st.settings.enabled &&
                      st.settings.url && st.settings.anonKey;
      if (!enabled) __syncInfo = "☁ Supabase: off";
      else if (s && s.ok) __syncInfo = "☁ Supabase: " + s.synced + " synced, " + s.pending + " pending";
      else __syncInfo = "☁ Supabase: …";
    } catch (_) { __syncInfo = ""; }
  }

  // On-page status panel (survives popup closing). Shows live progress.
  let panel = document.getElementById("__kp_panel");
  if (!panel) {
    panel = document.createElement("div");
    panel.id = "__kp_panel";
    panel.style.cssText =
      "position:fixed;top:70px;right:20px;z-index:2147483647;background:#1877f2;color:#fff;" +
      "font:13px/1.4 -apple-system,system-ui,sans-serif;padding:12px 14px;border-radius:10px;" +
      "box-shadow:0 4px 16px rgba(0,0,0,.3);width:230px;";
    document.body.appendChild(panel);
  }
  window.__kpStop = false;   // set true by the Stop button to end the grab early
  function fmtDate(d) { return d ? new Date(d).toISOString().slice(0, 10) : "—"; }
  function updatePanel(items) {
    const todayTs = Date.now();
    let oldest = null, newest = null;
    let undated = 0;
    for (const it of items) {
      // Only count posts whose date actually parses. An unparseable/blank date
      // is NOT assumed to be today — that would corrupt the range. We track the
      // undated count separately so the panel can show it honestly.
      const ts = parseFbDate(it.date, todayTs);
      if (ts == null) { undated++; continue; }
      if (oldest == null || ts < oldest) oldest = ts;
      if (newest == null || ts > newest) newest = ts;
    }
    panel.innerHTML =
      '<button id="__kp_closebtn" title="Close panel" style="position:absolute;top:6px;right:8px;' +
      'width:20px;height:20px;border:0;border-radius:5px;background:rgba(255,255,255,.18);' +
      'color:#fff;font-size:13px;line-height:1;cursor:pointer;padding:0">✕</button>' +
      '<div style="font-weight:700;margin-bottom:6px;padding-right:18px">KP Exporter · Facebook</div>' +
      (resumedCount > 0
        ? '<div style="font-size:11px;background:rgba(255,255,255,.2);padding:3px 6px;' +
          'border-radius:4px;margin-bottom:6px">↻ Resuming from ' + resumedCount +
          ' saved posts</div>'
        : '') +
      '<div>posts: <b>' + items.length + '</b></div>' +
      '<div style="font-size:11px;opacity:.9;margin-top:2px">range: ' +
        fmtDate(oldest) + ' → ' + fmtDate(newest) +
        (undated ? ' <span style="opacity:.8">(' + undated + ' undated)</span>' : '') + '</div>' +
      '<div style="font-size:11px;opacity:.95;margin-top:4px">' + __saveInfo + '</div>' +
      (__syncInfo ? '<div style="font-size:11px;opacity:.95;margin-top:2px">' + __syncInfo + '</div>' : '') +
      '<button id="__kp_stopbtn" style="width:100%;margin-top:8px;padding:8px;border:0;' +
      'border-radius:6px;background:#fff;color:#1877f2;font-weight:700;cursor:pointer">' +
      'STOP</button>' +
      '<div style="display:flex;gap:6px;margin-top:6px">' +
      '<button id="__kp_exportbtn" style="flex:1;padding:7px;border:0;border-radius:6px;' +
      'background:rgba(255,255,255,.2);color:#fff;font-weight:600;font-size:11px;cursor:pointer">' +
      'EXPORT JSON</button>' +
      '<button id="__kp_syncbtn" style="flex:1;padding:7px;border:0;border-radius:6px;' +
      'background:rgba(255,255,255,.2);color:#fff;font-weight:600;font-size:11px;cursor:pointer">' +
      'SYNC SUPABASE</button>' +
      '</div>';
    const cb = document.getElementById("__kp_closebtn");
    if (cb) cb.onclick = () => { panel.remove(); };
    const sb = document.getElementById("__kp_stopbtn");
    if (sb) sb.onclick = () => {
      window.__kpStop = true;          // halt the scroll loop
      sb.textContent = "stopped";
      sb.disabled = true;
    };
    const eb = document.getElementById("__kp_exportbtn");
    if (eb) eb.onclick = () => {
      eb.textContent = "exporting…"; eb.disabled = true;
      try { exportNow(); eb.textContent = "exported ✓"; }
      catch (e) { eb.textContent = "failed"; }
      setTimeout(() => { eb.textContent = "EXPORT JSON"; eb.disabled = false; }, 2500);
    };
    const syb = document.getElementById("__kp_syncbtn");
    if (syb) syb.onclick = async () => {
      syb.textContent = "syncing…"; syb.disabled = true;
      try {
        const r = await chrome.runtime.sendMessage({ type: "kp_sync_now" });
        syb.textContent = (r && r.ok) ? ("synced " + (r.pushed || 0)) : "sync failed";
      } catch (e) { syb.textContent = "sync failed"; }
      setTimeout(() => { syb.textContent = "SYNC SUPABASE"; syb.disabled = false; }, 2500);
    };
  }

  // Build the filtered JSON from current `collected` and trigger a download,
  // independent of the scroll loop. Called by the STOP button so export is
  // instant and works even if the loop already finished.
  function exportNow() {
    const now = Date.now();
    let out = collected;
    if (cutoffTs != null) {
      out = collected.filter((p) => {
        const ts = parseFbDate(p.date, now);
        return ts == null || ts >= cutoffTs;
      });
    }
    out = out.slice(0, opts.maxPosts);
    const cleanOut = out.map((p) => {
      const c = {};
      for (const k in p) if (!k.startsWith("_")) c[k] = p[k];
      return c;
    });
    const payload = {
      grabbed_at: new Date().toISOString(),
      url: location.href,
      count: cleanOut.length,
      posts: cleanOut,
    };
    const json = JSON.stringify(payload, null, 2);
    const a = document.createElement("a");
    a.href = "data:application/json;charset=utf-8," + encodeURIComponent(json);
    a.download = "fb_extension_" + new Date().toISOString().slice(0, 10) + ".json";
    document.body.appendChild(a); a.click(); a.remove();
  }
  const expandLabels = ["see more", "read more", "ver más", "mehr anzeigen", "voir plus", "see more."];

  // Posts collected across scroll steps. An expanded post is the same post
  // with longer text sharing the same prefix, so we dedup by containment
  // rather than a fixed-window fingerprint (which breaks when expansion
  // changes the early characters).
  const collected = [];   // array of {text, link, date}, deduped by text containment
  let expanded = 0;

  // Resume: load any previously stored items for this batch so a re-grab of
  // the same group continues instead of starting over.
  const savedPrefixes = new Set();   // normalized text prefixes already saved
  const normPrefix = (s) => (s || "").replace(/\s+/g, " ").trim().toLowerCase().slice(0, 80);
  try {
    const resp = await chrome.runtime.sendMessage({ type: "kp_load", batchKey });
    if (resp && resp.ok && resp.batch && Array.isArray(resp.batch.items)) {
      for (const it of resp.batch.items) {
        collected.push(it);
        savedPrefixes.add(normPrefix(it.text));
      }
    }
  } catch (_) {}
  const resumedCount = collected.length;   // posts loaded from a previous run

  // The group id from the current URL, used to build clean post permalinks.
  const GROUP_ID = (location.href.match(/groups\/(\d+)/) || [])[1] || "";

  // Pull the canonical numeric post id from any FB link variant.
  function postIdFromLink(link) {
    if (!link) return "";
    const m = link.match(/(?:\/posts\/|story_fbid=|\/permalink\/|[?&]fbid=|multi_permalinks=)(\d{5,})/);
    return m ? m[1] : "";
  }

  // Build the best reachable URL for a post:
  //   - if we can get the numeric id + group id → clean /groups/<g>/posts/<id>/
  //   - else fall back to whatever link we extracted (photo url etc.)
  // Guarantees a non-empty string whenever any link exists.
  function postUrlFor(link) {
    const pid = postIdFromLink(link);
    if (pid && GROUP_ID) return "https://www.facebook.com/groups/" + GROUP_ID + "/posts/" + pid + "/";
    return link || "";
  }

  function addPost(txt, link, date) {
    const norm = (s) => s.replace(/\s+/g, " ").trim().toLowerCase();
    const n = norm(txt);
    // Merge with an existing near-duplicate, preserving id/link/url/date and
    // upgrading any field that was previously missing or weaker.
    function mergeInto(idx, takeNewerText) {
      const cur = { ...collected[idx] };
      if (takeNewerText) cur.text = txt;
      if (link && !cur.link) cur.link = link;
      if (link && (!cur.url || cur.url === cur.link)) {
        const u = postUrlFor(link); if (u) cur.url = u;
      }
      if (!cur.date && date) cur.date = date;
      // upgrade id to a canonical post number once we have a link that yields one
      const pid = postIdFromLink(link || cur.link);
      if (pid && !/^fbid_/.test(cur.id || "")) cur.id = "fbid_" + pid;
      collected[idx] = cur;
    }
    for (let i = 0; i < collected.length; i++) {
      const c = norm(collected[i].text);
      if (c.includes(n)) { mergeInto(i, false); return; }   // existing is fuller
      if (n.includes(c)) { mergeInto(i, true); return; }    // this is fuller
      const minLen = Math.min(c.length, n.length);
      if (minLen > 60 && c.slice(0, 60) === n.slice(0, 60)) {
        mergeInto(i, txt.length > collected[i].text.length);
        return;
      }
    }
    // New post. id = canonical post number when available, else cleaned link,
    // else text prefix. link = whatever we found. url = best post permalink.
    const pid = postIdFromLink(link);
    collected.push({
      id: pid ? ("fbid_" + pid) : (link || normPrefix(txt)),
      text: txt,
      link: link || "",
      url: postUrlFor(link),         // post permalink (or link fallback)
      date: date || "",
    });
  }

  // Pull the real user-written text out of one post container using
  // div[dir="auto"] (FB marks user text this way; avoids image alt junk).
  function extractPost(container) {
    const seeMore = /\s*(?:see more|read more|ver más|mehr anzeigen|voir plus)\.?\s*$/i;
    let blocks = Array.from(container.querySelectorAll('div[dir="auto"]'))
      .map((d) => (d.innerText || "").trim().replace(seeMore, "").trim())
      .filter((t) => t.length > 0);
    if (blocks.length === 0) return "";
    // Nested dir=auto report overlapping text; drop any block contained in
    // a longer sibling, then join what remains.
    const uniq = [];
    blocks.sort((a, b) => b.length - a.length);
    for (const b of blocks) {
      if (!uniq.some((u) => u.includes(b))) uniq.push(b);
    }
    return uniq.join("\n").trim();
  }

  // Find the post's permalink. FB scatters several links in each post; we
  // want one that survives stripping the __cft__/__tn__ tracking junk into a
  // real post URL. Bare "?cft...#hash" links clean to nothing (no path/id),
  // so we score by whether a usable id/path remains.
  function cleanHref(href) {
    let h = href.split("__cft__")[0].split("?__cft__")[0];
    h = h.replace(/[?&]__tn__[^&]*/g, "").replace(/[?&#]+$/, "");
    if (h.startsWith("/")) h = "https://www.facebook.com" + h;
    return h;
  }
  function extractLink(container) {
    const cands = Array.from(container.querySelectorAll('a[href*="__cft__"]'))
      .map((a) => a.getAttribute("href") || "");
    const score = (raw) => {
      const h = cleanHref(raw);
      if (/\/posts\/\d|permalink\/?\d|story_fbid=\d|multi_permalinks=\d/.test(raw)) return 4;
      if (/\/photo\/?\?.*fbid=\d/.test(raw)) return 3;   // photo post, has id
      if (/\/groups\/\d+\/user\//.test(raw)) return 0;   // author profile
      // anything that still has a real path+id after cleaning
      if (/facebook\.com\/.+\d/.test(h)) return 2;
      return 0;                                          // cleans to nothing
    };
    let best = "", bestScore = 0;
    for (const raw of cands) {
      const s = score(raw);
      if (s > bestScore) { bestScore = s; best = raw; }
    }
    return best ? cleanHref(best) : "";
  }

  // Read the post's timestamp. Two formats exist:
  //  - recent posts (< ~1 week) show PLAIN relative text: "9h","2d","5d",
  //    "Yesterday at 3 PM" — read directly.
  //  - older posts show a CSS-SCRAMBLED absolute date ("May 26 at 12:25 PM"):
  //    real chars are position:relative, decoys are absolute; sort by CSS
  //    `order` to reconstruct.
  // Returns { raw, ts } where raw is the display string and ts is a JS
  // timestamp (ms) or null if unparseable. Best-effort throughout.
  function readDateString(container) {
    const links = Array.from(
      container.querySelectorAll('a[href*="/posts/"], a[href*="/permalink/"], a[href*="__cft__"], a[href*="story_fbid"]')
    );
    // STRICT relative: "20h", "2 d", "45m", "3w", "5 days", etc. Must be the
    // WHOLE string (a number + unit), so names/sentences can't match.
    const relRe = /^\d{1,2}\s?(h|hr|hrs|hour|hours|d|day|days|m|min|mins|w|wk|wks|week|weeks)$/i;
    // STRICT absolute: a month name adjacent to a day number, e.g. "May 20",
    // "20 May", "May 20, 2025", optionally with a time. Requires the
    // month+number pairing so "Julien" (Jul) or "May be an image" can't match.
    const absRe = /\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b|\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)/i;
    const isDateText = (t) =>
      relRe.test(t) || /^(yesterday|today|just now)\b/i.test(t) || absRe.test(t);

    // PASS 1 — the real timestamp link's own text ("20h", "May 20").
    for (const a of links) {
      const t = (a.textContent || "").trim();
      if (t && t.length <= 30 && isDateText(t)) return t;
    }
    // PASS 2 — a full date in an aria-label/title, but ONLY if it has the
    // month+day pairing (rejects "May be an image of…", names, etc.).
    for (const a of links) {
      const nodes = [a, ...a.querySelectorAll("[aria-label],[title]")];
      for (const node of nodes) {
        const al = (node.getAttribute && (node.getAttribute("aria-label") || node.getAttribute("title")) || "").trim();
        if (al && al.length <= 40 && absRe.test(al)) return al;
      }
    }
    // PASS 3 — scrambled-span decode (last resort). Only accept if the decoded
    // string itself passes the strict date test, so garbage is never stored.
    let best = null, bestCount = -1;
    for (const a of links) {
      const n = a.querySelectorAll("span").length;
      if (n > bestCount) { bestCount = n; best = a; }
    }
    if (!best) return "";
    const spans = Array.from(best.querySelectorAll("span"))
      .filter((s) => s.children.length === 0 && (s.textContent || "").length <= 2);
    if (spans.length < 4) return "";
    const decoded = spans
      .map((s) => {
        const cs = getComputedStyle(s);
        return { ch: s.textContent, order: parseFloat(cs.order) || 0, pos: cs.position };
      })
      .filter((x) => x.pos === "relative")
      .sort((a, b) => a.order - b.order)
      .map((x) => x.ch)
      .join("")
      .trim();
    // Strict acceptance: only a real month+day or HH:MM-bearing string counts.
    if (absRe.test(decoded) || /\b\d{1,2}:\d{2}\b/.test(decoded)) return decoded;
    return "";
  }

  // Parse a FB date string to a JS timestamp (ms). Handles relative ("2d",
  // "9h", "Yesterday") and absolute ("May 26 at 12:25 PM"). Returns null if
  // it can't parse. now is passed so the whole grab uses one reference time.
  // Resolve a raw FB date string to a fixed YYYY-MM-DD at capture time. This is
  // what we STORE, so relative times ("2d") are anchored to the scrape moment
  // and never drift when synced later. Returns "" if the date isn't readable
  // (we store nothing rather than guess).
  function resolveAtCapture(raw, now) {
    const ts = parseFbDate(raw, now);
    if (ts == null) return "";
    return new Date(ts).toISOString().slice(0, 10);
  }

  function parseFbDate(raw, now) {
    if (!raw) return null;
    // Already-resolved ISO date (we now store dates as YYYY-MM-DD at capture).
    const iso = String(raw).match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (iso) return Date.parse(raw + "T00:00:00");
    const s = raw.trim().toLowerCase();
    // relative: "9h", "17h", "2d", "5d", "3w", "10m"/"10 min"
    let m = s.match(/^(\d{1,2})\s?(m|min|h|hr|hrs|hour|d|day|w|week)s?\b/);
    if (m) {
      const v = parseInt(m[1], 10), u = m[2];
      let ms = 0;
      if (/^m/.test(u)) ms = v * 60e3;
      else if (/^h/.test(u)) ms = v * 3600e3;
      else if (/^d/.test(u)) ms = v * 86400e3;
      else if (/^w/.test(u)) ms = v * 7 * 86400e3;
      return now - ms;
    }
    if (/yesterday/.test(s)) return now - 86400e3;
    if (/today|just now/.test(s)) return now;
    // absolute: must contain a month name to be trusted (Date.parse is too
    // lenient otherwise and will accept junk). Year is usually omitted, so we
    // set it explicitly: current year, or last year if that lands in future.
    const months = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec";
    if (!new RegExp("\\b(" + months + ")", "i").test(s)) return null;
    const cleaned = raw.replace(/\bat\b/i, "").replace(/\s+/g, " ").trim();
    const yr = new Date(now).getFullYear();
    let t = Date.parse(cleaned + " " + yr);
    if (isNaN(t)) return null;
    if (t > now + 86400e3) t = Date.parse(cleaned + " " + (yr - 1));
    return isNaN(t) ? null : t;
  }

  function getFeed() { return document.querySelector('[role="feed"]'); }


  // Harvest everything currently rendered in the feed. Returns the oldest
  // parseable post timestamp seen in THIS pass (or null), for cutoff logic.
  function harvest(now, cutoffTs) {
    const feed = getFeed();
    if (!feed) return { oldestTs: null, sawOlder: 0 };
    let oldestTs = null, sawOlder = 0;
    for (const kid of feed.children) {
      if ((kid.innerText || "").length < 50) continue; // empty placeholder
      const txt = extractPost(kid);
      if (txt.length < 8) continue;   // keep short posts (e.g. compact Thai); only skip empty fragments
      if (savedPrefixes.size && savedPrefixes.has(normPrefix(txt))) continue; // already saved
      const raw = readDateString(kid);
      // Resolve the date NOW, at capture time, so relative times ("2d") become
      // a fixed calendar date anchored to the scrape moment — no drift from a
      // later sync. Absolute dates resolve to their date; unreadable → "".
      const resolved = resolveAtCapture(raw, now);
      addPost(txt, extractLink(kid), resolved);
      if (raw) {
        const ts = parseFbDate(raw, now);
        if (ts != null) {
          if (oldestTs == null || ts < oldestTs) oldestTs = ts;
          if (cutoffTs != null && ts < cutoffTs) sawOlder++;
        }
      }
    }
    return { oldestTs, sawOlder };
  }

  // Click "See more" buttons currently in the feed to expand truncated posts.
  // On resume, skip posts whose text is already saved — no need to expand or
  // re-process them, which makes scrolling back through the saved region fast.
  async function expandVisible() {
    const feed = getFeed();
    if (!feed) return;
    let clicked = 0;
    for (const kid of feed.children) {
      if ((kid.innerText || "").length < 50) continue;
      // already-saved? skip expansion entirely for this post
      if (savedPrefixes.size && savedPrefixes.has(normPrefix(extractPost(kid)))) continue;
      const buttons = Array.from(kid.querySelectorAll('[role="button"], span'))
        .filter((el) => {
          const t = (el.textContent || "").trim().toLowerCase();
          return expandLabels.includes(t) && el.offsetParent !== null;
        });
      for (const b of buttons) {
        try { b.click(); expanded++; clicked++; } catch (_) {}
      }
    }
    if (clicked) await sleep(500);
  }

  // Main loop. Facebook loads the feed in BURSTS with pauses between them, so
  // "a step added no new posts" does NOT mean the end — it usually means FB is
  // mid-fetch. Treating that as the end is what caused random early stops.
  // Fix: a step adding nothing triggers a firm wake-up (big jump + long wait);
  // only if the wake-up ALSO yields nothing do we count a real stall, and we
  // require several real stalls in a row AND being at the document bottom
  // before concluding the feed has ended.
  const stepPx = Math.round(window.innerHeight * 0.5);
  const maxSteps = 800;            // safety cap; real stop ends it normally
  const stopAfterRealStall = 6;    // consecutive failed wake-ups => ended

  const now = Date.now();
  const cutoffTs = opts.cutoffDate ? Date.parse(opts.cutoffDate + "T00:00:00") : null;
  let olderStreak = 0;

  const atBottom = () =>
    (window.innerHeight + window.scrollY) >= (document.body.scrollHeight - 80);

  // scroll to top so we capture the whole feed from the newest post down
  window.scrollTo(0, 0);
  await sleep(1300);
  await expandVisible();
  harvest(now, cutoffTs);

  let realStall = 0, prevCount = collected.length;
  for (let i = 0; i < maxSteps; i++) {
    if (window.__kpStop) break;                  // user pressed STOP
    if (collected.length >= opts.maxPosts) break;

    window.scrollBy(0, stepPx);
    await sleep(1200);
    await expandVisible();
    let h = harvest(now, cutoffTs);
    await persist(collected);
    updatePanel(collected);

    if (cutoffTs != null) {
      if (h.sawOlder > 0) olderStreak += h.sawOlder; else olderStreak = 0;
      if (olderStreak >= 3) break;        // clearly past the cutoff date
    }

    if (collected.length > prevCount) {
      realStall = 0;
      prevCount = collected.length;
      continue;
    }

    // Plain step added nothing — could just be a load pause. Wake FB up:
    // jump down 3 screens, wait long, harvest; try twice before giving up.
    let recovered = false;
    for (let w = 0; w < 2 && !recovered; w++) {
      if (window.__kpStop) break;
      window.scrollBy(0, window.innerHeight * 3);
      await sleep(2600);
      await expandVisible();
      h = harvest(now, cutoffTs);
      if (cutoffTs != null && h.sawOlder > 0) {
        olderStreak += h.sawOlder;
        if (olderStreak >= 3) { recovered = true; break; }
      }
      if (collected.length > prevCount) {
        realStall = 0;
        prevCount = collected.length;
        recovered = true;
      }
    }
    if (recovered) continue;

    // Wake-up failed. Only NOW does it count as a real stall — and only if we
    // are actually at the bottom of the document. If we're not at the bottom,
    // FB is likely just throttling; keep trying without counting a stall.
    if (atBottom()) {
      realStall++;
      if (realStall >= stopAfterRealStall) break;   // genuinely ended
    }
  }

  // one final pass
  await expandVisible();
  harvest(now, cutoffTs);
  await persist(collected);
  updatePanel(collected);

  // Build result. The batch (collected) keeps EVERYTHING; we only filter the
  // EXPORTED set. The model is "everything from today back to the cutoff
  // date", so we drop posts OLDER than the cutoff. Undated/recent posts are
  // always kept (they're at the newest end, always in range).
  let out = collected;
  if (cutoffTs != null) {
    out = collected.filter((p) => {
      const ts = parseFbDate(p.date, now);
      return ts == null || ts >= cutoffTs;   // keep undated + newer-than-cutoff
    });
  }
  return { posts: out.slice(0, opts.maxPosts), expanded };
}

/* ===================== Injected: WhatsApp ===================== */
function startSweeper(opts) {
  opts = opts || {};
  const maxMsgs = (typeof opts.maxMsgs === "number" && opts.maxMsgs > 0) ? opts.maxMsgs : Infinity;
  const cutoffTs = opts.cutoffDate ? Date.parse(opts.cutoffDate + "T00:00:00") : null;
  // avoid double-injection
  if (window.__waSweeperOn) {
    const p = document.getElementById("__wa_panel");
    if (p) p.style.display = "block";
    return;
  }
  window.__waSweeperOn = true;

  const main = document.querySelector("#main");
  const collected = new Map();   // data-id -> {sender, datetime, ts, text}

  let chatName = "";
  const header = main && main.querySelector("header");
  if (header) {
    const spans = Array.from(header.querySelectorAll('span[dir="auto"]'))
      .map((s) => (s.textContent || "").trim()).filter(Boolean);
    if (spans.length) chatName = spans[0];            // first dir=auto = group name
    else {
      const h = header.querySelector('h1,h2,[role="heading"]');
      chatName = h ? (h.textContent || "").trim() : "";
    }
  }

  // Batch key from chat name (stable per chat). Exposed so the popup's
  // Export button can find this chat's saved data.
  const batchKey = "wa_" + (chatName || "current").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 40);
  window.__kpBatchKey = batchKey;

  let __waSync = "";
  let __dbOldest = null;   // oldest YYYY-MM-DD already saved in DB for this chat
  async function refreshDbRange() {
    try {
      const rr = await chrome.runtime.sendMessage({ type: "kp_get_range", batchKey });
      if (rr && rr.ok && rr.range && rr.range.oldest) __dbOldest = rr.range.oldest;
    } catch (_) {}
  }
  refreshDbRange();
  async function persist() {
    try {
      const items = Array.from(collected.values()).map((m) => ({
        id: m._id || (m.datetime + "|" + (m.text || "").slice(0, 40)),
        ...m,
      }));
      await chrome.runtime.sendMessage({
        type: "kp_save",
        batchKey,
        source: "whatsapp",
        url: location.href,
        chat: chatName,
        items,
      });
    } catch (_) {}
    try {
      const s = await chrome.runtime.sendMessage({ type: "kp_sync_status" });
      const st = await chrome.runtime.sendMessage({ type: "kp_get_settings" });
      const enabled = st && st.ok && st.settings && st.settings.enabled &&
                      st.settings.url && st.settings.anonKey;
      if (!enabled) __waSync = "☁ Supabase: off";
      else if (s && s.ok) __waSync = "☁ Supabase: " + s.synced + " synced, " + s.pending + " pending";
      else __waSync = "";
    } catch (_) { __waSync = ""; }
  }

  // Resume: load any previously saved messages for this chat via the worker.
  (async () => {
    try {
      const resp = await chrome.runtime.sendMessage({ type: "kp_load", batchKey });
      if (resp && resp.ok && resp.batch && Array.isArray(resp.batch.items)) {
        for (const it of resp.batch.items) {
          const id = it._id || it.id || (it.datetime + "|" + (it.text || "").slice(0, 40));
          if (!collected.has(id)) collected.set(id, it);
        }
        updatePanel();
      }
    } catch (_) {}
  })();

  function parsePrePlain(s) {
    const m = s.match(/^\[(\d{1,2}):(\d{2})(?::\d{2})?\s*(?:[ap]m)?,\s*(\d{1,2})\/(\d{1,2})\/(\d{2,4})\]\s*(.*?):\s*$/i);
    if (!m) return { senderPhone: "", senderName: "", ts: null, datetime: "" };
    const hh = +m[1], mm = +m[2], a = +m[3], b = +m[4];
    const rawSender = (m[6] || "").trim();
    let year = +m[5]; if (year < 100) year += 2000;
    let day, mon;
    if (a > 12) { day = a; mon = b; }
    else if (b > 12) { mon = a; day = b; }
    else { mon = a; day = b; }
    const ts = new Date(year, mon - 1, day, hh, mm).getTime();
    const datetime = s.replace(/^\[/, "").replace(/\].*$/, "").trim();
    // WhatsApp shows EITHER a phone number (when not a saved contact) OR a
    // display name (saved contact / business account). A phone sender looks
    // like "+31 6 47488900" — starts with + or is mostly digits/spaces.
    const isPhone = /^\+?[\d][\d\s\-()]{5,}$/.test(rawSender);
    return {
      senderPhone: isPhone ? rawSender : "",
      senderName: isPhone ? "" : rawSender,
      ts: isNaN(ts) ? null : ts,
      datetime,
    };
  }

  let __targetReached = false;
  let __reachedReason = "";
  function sweep() {
    if (!main) return;
    for (const row of main.querySelectorAll("[data-id]")) {
      const id = row.getAttribute("data-id");
      if (!id || collected.has(id)) continue;
      const holder = row.querySelector("[data-pre-plain-text]");
      if (!holder) continue;                  // system notice / divider / media-only
      const preStr = holder.getAttribute("data-pre-plain-text") || "";
      const text = (holder.innerText || holder.textContent || "").trim();
      if (!text) continue;                     // media-only / empty
      const { senderPhone, senderName, ts, datetime } = parsePrePlain(preStr);
      // Date target: skip messages older than the cutoff (mirrors FB: keep
      // today back to the cutoff; undated messages are always kept).
      if (cutoffTs != null && ts != null && ts < cutoffTs) {
        __targetReached = true;
        __reachedReason = "reached stop date";
        continue;                              // don't save older-than-cutoff
      }
      // Count target: stop saving once we hit maxMsgs.
      if (collected.size >= maxMsgs) {
        __targetReached = true;
        __reachedReason = "reached " + maxMsgs + " messages";
        break;
      }
      collected.set(id, { _id: id, senderPhone, senderName, datetime, ts, text });
    }
    if (collected.size >= maxMsgs) { __targetReached = true; __reachedReason = "reached " + maxMsgs + " messages"; }
    updatePanel();
  }

  // floating control panel
  const panel = document.createElement("div");
  panel.id = "__wa_panel";
  panel.style.cssText =
    "position:fixed;top:70px;right:20px;z-index:999999;background:#00a884;color:#fff;" +
    "font:13px/1.4 -apple-system,system-ui,sans-serif;padding:12px 14px;border-radius:10px;" +
    "box-shadow:0 4px 16px rgba(0,0,0,.3);width:210px;";
  panel.innerHTML =
    '<button id="__wa_close" title="Close panel" style="position:absolute;top:6px;right:8px;' +
    'width:20px;height:20px;border:0;border-radius:5px;background:rgba(255,255,255,.18);' +
    'color:#fff;font-size:13px;line-height:1;cursor:pointer;padding:0">✕</button>' +
    '<div style="font-weight:700;margin-bottom:6px;padding-right:18px">WA Grabber</div>' +
    '<div id="__wa_count" style="margin-bottom:4px">messages: 0</div>' +
    '<div id="__wa_oldest" style="font-size:11px;opacity:.9;margin-bottom:4px">range: —</div>' +
    '<div id="__wa_target" style="font-size:11px;font-weight:700;margin-bottom:6px;display:none;background:rgba(255,255,255,.25);padding:3px 6px;border-radius:4px"></div>' +
    '<div id="__wa_dbwarn" style="font-size:11px;font-weight:600;margin-bottom:6px;display:none;background:rgba(255,200,0,.3);padding:3px 6px;border-radius:4px"></div>' +
    '<div id="__wa_sync" style="font-size:11px;opacity:.95;margin-bottom:8px"></div>' +
    '<button id="__wa_stop" style="width:100%;padding:8px;border:0;border-radius:6px;background:#fff;color:#00a884;font-weight:700;cursor:pointer">STOP</button>' +
    '<div style="display:flex;gap:6px;margin-top:6px">' +
    '<button id="__wa_export" style="flex:1;padding:7px;border:0;border-radius:6px;background:rgba(255,255,255,.2);color:#fff;font-weight:600;font-size:11px;cursor:pointer">EXPORT JSON</button>' +
    '<button id="__wa_sync_btn" style="flex:1;padding:7px;border:0;border-radius:6px;background:rgba(255,255,255,.2);color:#fff;font-weight:600;font-size:11px;cursor:pointer">SYNC SUPABASE</button>' +
    '</div>' +
    '<div style="font-size:11px;opacity:.9;margin-top:6px">Scroll UP by hand to load older messages.</div>';
  document.body.appendChild(panel);
  document.getElementById("__wa_close").addEventListener("click", () => {
    window.__waSweeperOn = false;
    if (typeof timer !== "undefined") clearInterval(timer);
    panel.remove();
  });

  function updatePanel() {
    const c = document.getElementById("__wa_count");
    const o = document.getElementById("__wa_oldest");
    if (c) c.textContent = "messages: " + collected.size + "  (✓ saved)";
    if (o) {
      let oldest = null, newest = null;
      const todayTs = Date.now();
      for (const v of collected.values()) {
        let ts = v.ts;
        if (ts == null) ts = todayTs;   // undated = recent = today
        if (oldest == null || ts < oldest) oldest = ts;
        if (newest == null || ts > newest) newest = ts;
      }
      const f = (t) => t ? new Date(t).toISOString().slice(0, 10) : "—";
      o.textContent = "range: " + f(oldest) + " → " + f(newest);
    }
    const sy = document.getElementById("__wa_sync");
    if (sy) sy.textContent = __waSync;
    const tg = document.getElementById("__wa_target");
    if (tg) {
      if (__targetReached) {
        tg.style.display = "block";
        tg.textContent = "✓ Target " + __reachedReason + " — you can stop scrolling";
      } else {
        tg.style.display = "none";
      }
    }
    // "scrolling into already-saved DB territory" warning: if the oldest
    // message currently collected is older than the oldest date already in
    // the DB, you've scrolled past what's saved.
    const dbw = document.getElementById("__wa_dbwarn");
    if (dbw) {
      let oldestOnScreen = null;
      for (const v of collected.values()) {
        if (v.ts == null) continue;
        if (oldestOnScreen == null || v.ts < oldestOnScreen) oldestOnScreen = v.ts;
      }
      if (__dbOldest && oldestOnScreen != null) {
        const oldestDate = new Date(oldestOnScreen).toISOString().slice(0, 10);
        if (oldestDate <= __dbOldest) {
          dbw.style.display = "block";
          dbw.textContent = "⚠ You're scrolling past " + __dbOldest +
            " — messages this old are already in the database.";
        } else {
          dbw.style.display = "none";
        }
      } else {
        dbw.style.display = "none";
      }
    }
  }

  const timer = setInterval(() => { sweep(); persist(); }, 1000);
  sweep();

  function waExportNow() {
    const msgs = Array.from(collected.values())
      .sort((x, y) => (x.ts || 0) - (y.ts || 0))
      .map(({ _id, ...rest }) => {
        const c = {};
        for (const k in rest) if (!k.startsWith("_")) c[k] = rest[k];
        return c;
      });
    const payload = {
      grabbed_at: new Date().toISOString(),
      url: location.href,
      chat: chatName,
      count: msgs.length,
      messages: msgs,
    };
    const json = JSON.stringify(payload, null, 2);
    const safe = (chatName || "chat").replace(/[^a-z0-9]+/gi, "_").slice(0, 30);
    const a = document.createElement("a");
    a.href = "data:application/json;charset=utf-8," + encodeURIComponent(json);
    a.download = "wa_" + safe + "_" + new Date().toISOString().slice(0, 10) + ".json";
    document.body.appendChild(a); a.click(); a.remove();
    return msgs.length;
  }

  document.getElementById("__wa_stop").addEventListener("click", (e) => {
    clearInterval(timer);
    sweep();
    persist();
    window.__waSweeperOn = false;
    const b = e.target; b.textContent = "stopped"; b.disabled = true;
  });

  document.getElementById("__wa_export").addEventListener("click", (e) => {
    const b = e.target; b.textContent = "exporting…"; b.disabled = true;
    try { const n = waExportNow(); b.textContent = "exported " + n; }
    catch (_) { b.textContent = "failed"; }
    setTimeout(() => { b.textContent = "EXPORT JSON"; b.disabled = false; }, 2500);
  });

  document.getElementById("__wa_sync_btn").addEventListener("click", async (e) => {
    const b = e.target; b.textContent = "syncing…"; b.disabled = true;
    try {
      await persist();   // make sure latest is saved before syncing
      const r = await chrome.runtime.sendMessage({ type: "kp_sync_now" });
      b.textContent = (r && r.ok) ? ("synced " + (r.pushed || 0)) : "sync failed";
    } catch (_) { b.textContent = "sync failed"; }
    setTimeout(() => { b.textContent = "SYNC SUPABASE"; b.disabled = false; }, 2500);
  });
}
