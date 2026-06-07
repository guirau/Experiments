const $ = (id) => document.getElementById(id);
const status = (msg) => { $("status").textContent = msg; };

// Detect which site the active tab is on, then show the matching panel.
(async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const url = (tab && tab.url) || "";
  if (/web\.whatsapp\.com/.test(url)) {
    $("wa").classList.remove("hidden");
    $("siteLabel").textContent = "WhatsApp Web";
    wireWhatsApp(tab);
  } else if (/facebook\.com/.test(url)) {
    $("fb").classList.remove("hidden");
    $("siteLabel").textContent = "Facebook";
    wireFacebook(tab);
  } else {
    $("none").classList.remove("hidden");
    $("siteLabel").textContent = "";
  }
})();

/* ===================== Facebook mode ===================== */
function wireFacebook(tab) {
  $("fbgrab").addEventListener("click", async () => {
    const opts = {
      maxPosts: parseInt($("maxposts").value, 10) || 100,
      cutoffDate: $("cutoff").value || null,
    };
    const wantClip = $("fbclip").checked;
    const wantDl = $("fbdl").checked;

    $("fbgrab").disabled = true;
    status("Working… keep this tab in front.");

    let result;
    try {
      const [{ result: r }] = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: grabPostsInPage,
        args: [opts],
      });
      result = r;
    } catch (e) {
      status("Injection failed: " + e.message);
      $("fbgrab").disabled = false;
      return;
    }

    if (!result || !result.posts || result.posts.length === 0) {
      status("No posts found. Try scrolling more, or check you're on a group feed.");
      $("fbgrab").disabled = false;
      return;
    }

    const payload = {
      grabbed_at: new Date().toISOString(),
      url: tab.url,
      count: result.posts.length,
      posts: result.posts,
    };
    const json = JSON.stringify(payload, null, 2);

    if (wantClip) { try { await navigator.clipboard.writeText(json); } catch (_) {} }
    if (wantDl) {
      const dataUrl = "data:application/json;charset=utf-8," + encodeURIComponent(json);
      const fname = "fb_extension_" + new Date().toISOString().slice(0, 10) + ".json";
      chrome.downloads.download({ url: dataUrl, filename: fname, saveAs: true });
    }

    status(`Done: ${result.posts.length} posts grabbed`
      + (wantClip ? "\nCopied to clipboard." : "")
      + (wantDl ? "\nDownload started." : ""));
    $("fbgrab").disabled = false;
  });
}

/* ===================== WhatsApp mode ===================== */
function wireWhatsApp(tab) {
  $("wagrab").addEventListener("click", async () => {
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: startSweeper,
      });
      status("Sweeper started on the page.\nScroll UP by hand as far back as you\nwant, then click STOP & DOWNLOAD on\nthe green panel.");
    } catch (e) {
      status("Injection failed: " + e.message);
    }
  });
}

/* ===================== Injected: Facebook ===================== */
async function grabPostsInPage(opts) {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const expandLabels = ["see more", "read more", "ver más", "mehr anzeigen", "voir plus", "see more."];

  // Posts collected across scroll steps. An expanded post is the same post
  // with longer text sharing the same prefix, so we dedup by containment
  // rather than a fixed-window fingerprint (which breaks when expansion
  // changes the early characters).
  const collected = [];   // array of {text, link, date}, deduped by text containment
  let expanded = 0;

  function addPost(txt, link, date) {
    const norm = (s) => s.replace(/\s+/g, " ").trim().toLowerCase();
    const n = norm(txt);
    for (let i = 0; i < collected.length; i++) {
      const c = norm(collected[i].text);
      if (c.includes(n)) {                         // already have longer text
        if (!collected[i].link && link) collected[i].link = link;
        if (!collected[i].date && date) collected[i].date = date;
        return;
      }
      if (n.includes(c)) {                         // this is fuller
        collected[i] = {
          text: txt,
          link: link || collected[i].link,
          date: date || collected[i].date,
        };
        return;
      }
      const minLen = Math.min(c.length, n.length);
      if (minLen > 60 && c.slice(0, 60) === n.slice(0, 60)) {
        if (txt.length > collected[i].text.length) {
          collected[i] = {
            text: txt,
            link: link || collected[i].link,
            date: date || collected[i].date,
          };
        } else {
          if (!collected[i].link && link) collected[i].link = link;
          if (!collected[i].date && date) collected[i].date = date;
        }
        return;
      }
    }
    collected.push({ text: txt, link: link || "", date: date || "" });
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
      container.querySelectorAll('a[href*="/posts/"], a[href*="__cft__"]')
    );
    let best = null, bestCount = -1;
    for (const a of links) {
      const n = a.querySelectorAll("span").length;
      if (n > bestCount) { bestCount = n; best = a; }
    }
    if (!best) return "";

    // try plain text first (recent posts)
    const plain = (best.textContent || "").trim();
    if (plain && plain.length <= 25 &&
        (/^\d{1,2}\s?(h|hr|hrs|hour|d|day|days|m|min|w)s?\b/i.test(plain)
         || /yesterday|today|just now/i.test(plain)
         || /\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)/i.test(plain))) {
      return plain;
    }

    // fall back to scrambled-span decode (older posts)
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
    const looksDate = /\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)/i.test(decoded)
      || /\d{1,2}:\d{2}/.test(decoded)
      || /yesterday|today/i.test(decoded);
    return looksDate ? decoded : "";
  }

  // Parse a FB date string to a JS timestamp (ms). Handles relative ("2d",
  // "9h", "Yesterday") and absolute ("May 26 at 12:25 PM"). Returns null if
  // it can't parse. now is passed so the whole grab uses one reference time.
  function parseFbDate(raw, now) {
    if (!raw) return null;
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
      if (txt.length < 25) continue;
      const raw = readDateString(kid);
      addPost(txt, extractLink(kid), raw);
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
  async function expandVisible() {
    const feed = getFeed();
    if (!feed) return;
    const buttons = Array.from(feed.querySelectorAll('[role="button"], span'))
      .filter((el) => {
        const t = (el.textContent || "").trim().toLowerCase();
        return expandLabels.includes(t) && el.offsetParent !== null;
      });
    for (const b of buttons) {
      try { b.click(); expanded++; } catch (_) {}
    }
    if (buttons.length) await sleep(500);
  }

  // Main loop. Facebook's measured page height plateaus even while it keeps
  // virtualizing in new posts below, so we do NOT stop on scrollHeight.
  // We stop when (a) we've collected maxPosts, or (b) many consecutive steps
  // add no new posts (the real "reached the end" signal). When a step adds
  // nothing we do a big jump + long wait, which is what reliably forces FB to
  // fetch and render the next batch (validated to pull 150+ posts).
  const stepPx = Math.round(window.innerHeight * 0.5);
  const maxSteps = 400;       // safety cap so it can never loop forever
  const stopAfterIdle = 10;   // consecutive no-progress steps => bottom

  const now = Date.now();
  // cutoffTs: stop once posts are older than this. null = no date cutoff.
  const cutoffTs = opts.cutoffDate ? Date.parse(opts.cutoffDate + "T00:00:00") : null;
  // require this many consecutive older-than-cutoff posts before stopping,
  // so a single pinned/misdecoded post can't end the run early.
  let olderStreak = 0;

  // scroll to top so we capture the whole feed from the newest post down
  window.scrollTo(0, 0);
  await sleep(1200);
  await expandVisible();
  harvest(now, cutoffTs);

  let idle = 0, prevCount = collected.length;
  for (let i = 0; i < maxSteps; i++) {
    if (collected.length >= opts.maxPosts) break;   // reached post cap

    window.scrollBy(0, stepPx);
    await sleep(1100);
    await expandVisible();
    let h = harvest(now, cutoffTs);

    // date cutoff: feed is chronological, so once we keep seeing posts older
    // than the cutoff we've gone far enough back.
    if (cutoffTs != null) {
      if (h.sawOlder > 0) olderStreak += h.sawOlder; else olderStreak = 0;
      if (olderStreak >= 2) break;
    }

    if (collected.length > prevCount) {
      idle = 0;
      prevCount = collected.length;
    } else {
      idle++;
      window.scrollBy(0, window.innerHeight * 3);
      await sleep(2500);
      await expandVisible();
      h = harvest(now, cutoffTs);
      if (cutoffTs != null && h.sawOlder > 0) { olderStreak += h.sawOlder; if (olderStreak >= 2) break; }
      if (collected.length > prevCount) { idle = 0; prevCount = collected.length; }
      if (idle >= stopAfterIdle) break;   // genuinely reached the end
    }
  }

  // one final pass
  await expandVisible();
  harvest(now, cutoffTs);

  // Build result. If a date cutoff is set, drop posts clearly older than it;
  // posts with an unparseable/blank date are KEPT (we don't discard on
  // uncertainty). Then trim to the post cap.
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
function startSweeper() {
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
    const t = header.querySelector('span[dir="auto"], span[title]');
    chatName = t ? (t.getAttribute("title") || t.textContent || "").trim() : "";
  }

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

  function sweep() {
    if (!main) return;
    for (const row of main.querySelectorAll("[data-id]")) {
      const id = row.getAttribute("data-id");
      if (!id || collected.has(id)) continue;
      // The .copyable-text holder carries BOTH the date attribute and the
      // message text. (span.selectable-text is empty in current WA Web.)
      const holder = row.querySelector("[data-pre-plain-text]");
      if (!holder) continue;                  // system notice / divider / media-only
      const preStr = holder.getAttribute("data-pre-plain-text") || "";
      // text lives inside the holder itself; strip nothing, just read it
      const text = (holder.innerText || holder.textContent || "").trim();
      if (!text) continue;                     // media-only / empty
      const { senderPhone, senderName, ts, datetime } = parsePrePlain(preStr);
      collected.set(id, { senderPhone, senderName, datetime, ts, text });
    }
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
    '<div style="font-weight:700;margin-bottom:6px">WA Grabber</div>' +
    '<div id="__wa_count" style="margin-bottom:4px">messages: 0</div>' +
    '<div id="__wa_oldest" style="font-size:11px;opacity:.9;margin-bottom:8px">oldest: —</div>' +
    '<button id="__wa_stop" style="width:100%;padding:8px;border:0;border-radius:6px;background:#fff;color:#00a884;font-weight:700;cursor:pointer">STOP &amp; DOWNLOAD</button>' +
    '<div style="font-size:11px;opacity:.9;margin-top:6px">Scroll UP by hand to load older messages.</div>';
  document.body.appendChild(panel);

  function updatePanel() {
    const c = document.getElementById("__wa_count");
    const o = document.getElementById("__wa_oldest");
    if (c) c.textContent = "messages: " + collected.size;
    if (o) {
      let oldest = null;
      for (const v of collected.values()) if (v.ts != null && (oldest == null || v.ts < oldest)) oldest = v.ts;
      o.textContent = "oldest: " + (oldest ? new Date(oldest).toISOString().slice(0, 10) : "—");
    }
  }

  const timer = setInterval(sweep, 800);
  sweep();

  document.getElementById("__wa_stop").addEventListener("click", () => {
    clearInterval(timer);
    sweep();
    window.__waSweeperOn = false;

    const msgs = Array.from(collected.values()).sort((x, y) => (x.ts || 0) - (y.ts || 0));

    const payload = {
      grabbed_at: new Date().toISOString(),
      url: location.href,
      chat: chatName,
      count: msgs.length,
      messages: msgs,
    };
    const json = JSON.stringify(payload, null, 2);
    try { navigator.clipboard.writeText(json); } catch (_) {}
    const safe = (chatName || "chat").replace(/[^a-z0-9]+/gi, "_").slice(0, 30);
    const a = document.createElement("a");
    a.href = "data:application/json;charset=utf-8," + encodeURIComponent(json);
    a.download = "wa_" + safe + "_" + new Date().toISOString().slice(0, 10) + ".json";
    document.body.appendChild(a); a.click(); a.remove();

    panel.innerHTML = '<div style="font-weight:700">Done</div><div>' + msgs.length +
      ' messages saved.<br>Copied to clipboard +<br>download started.</div>';
    setTimeout(() => panel.remove(), 6000);
  });
}
