"use client";
import { useCallback, useEffect, useState } from "react";
import { triggerDiscover, triggerEnrich, fetchJobStatus, type JobState, type JobStatus } from "@/lib/prospectingApi";

const COST_PER_ENRICH_USD = 0.015;

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

// One-line human summary of a job's current state.
function jobLine(name: string, s: JobState): string | null {
  if (s.status === "idle") return null;
  if (s.status === "running") return `${name}: running…`;
  if (s.status === "error") return `${name}: error — ${s.error}`;
  const parts = s.result ? Object.entries(s.result).map(([k, v]) => `${v} ${k}`).join(", ") : "done";
  return `${name}: done (${parts})`;
}

export interface ProspectingControlsProps {
  queuedCount: number;
  onRefresh: () => void;
}

export function ProspectingControls({ queuedCount, onRefresh }: ProspectingControlsProps) {
  const [limit, setLimit] = useState("");
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [polling, setPolling] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  // While a run is in flight, poll /status; stop + refresh prospects when both jobs settle.
  useEffect(() => {
    if (!polling) return;
    let on = true;
    const tick = async () => {
      try {
        const s = await fetchJobStatus();
        if (!on) return;
        setStatus(s);
        if (s.discover.status !== "running" && s.enrich.status !== "running") {
          setPolling(false);
          onRefresh();
        }
      } catch (e) {
        if (on) { setMsg(errMsg(e)); setPolling(false); }
      }
    };
    tick();
    const id = setInterval(tick, 2000);
    return () => { on = false; clearInterval(id); };
  }, [polling, onRefresh]);

  const discover = useCallback(async () => {
    setMsg(null);
    try {
      await triggerDiscover(limit.trim() === "" ? null : Math.max(1, Math.round(Number(limit))));
      setPolling(true);
    } catch (e) { setMsg(errMsg(e)); }
  }, [limit]);

  const enrich = useCallback(async () => {
    setMsg(null);
    if (queuedCount === 0) { setMsg("No places queued — select pins on the map and press Queue first."); return; }
    const cost = (queuedCount * COST_PER_ENRICH_USD).toFixed(2);
    if (!window.confirm(`Enrich ${queuedCount} queued place(s)? ~$${cost} in SerpApi calls.`)) return;
    try { await triggerEnrich(); setPolling(true); }
    catch (e) { setMsg(errMsg(e)); }
  }, [queuedCount]);

  const lines = status ? [jobLine("Discover", status.discover), jobLine("Enrich", status.enrich)].filter(Boolean) : [];

  return (
    <div className="mb-3 flex flex-wrap items-center gap-3 rounded-2xl border p-3" style={{ borderColor: "var(--line)", background: "var(--surface)" }}>
      <div className="flex items-center gap-2">
        <button onClick={discover} disabled={polling} className="card-btn">🔎 Discover places</button>
        <input type="number" inputMode="numeric" min={1} value={limit} onChange={(e) => setLimit(e.target.value)}
          placeholder="limit" aria-label="Discover limit (new places)"
          className="w-20 rounded-lg border px-2 py-1 text-sm" style={{ borderColor: "var(--line)" }} />
      </div>
      <span aria-hidden className="h-6 w-px" style={{ background: "var(--line)" }} />
      <button onClick={enrich} disabled={polling || queuedCount === 0} className={`card-btn${queuedCount > 0 ? " card-btn-on" : ""}`}>
        💲 Enrich queued ({queuedCount})
      </button>

      <div className="ml-auto flex flex-col items-end text-xs" style={{ color: "var(--muted)" }}>
        {polling && <span>working… (polling)</span>}
        {lines.map((l, i) => (
          <span key={i} style={{ color: l!.includes("error") ? "var(--warn)" : "var(--muted)" }}>{l}</span>
        ))}
        {msg && <span style={{ color: "var(--warn)" }}>{msg}</span>}
      </div>
    </div>
  );
}
