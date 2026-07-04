"use client";
import { useState } from "react";
import dynamic from "next/dynamic";
import type { useTracker } from "@/hooks/useTracker";
import { useProspects } from "@/hooks/useProspects";
import { TrackerTable } from "@/components/tracker/TrackerTable";
import { Skeleton } from "@/components/ui/Skeleton";
import { ProspectCard } from "./ProspectCard";
import { ProspectingControls } from "./ProspectingControls";

// Leaflet touches the DOM/window, so the map only loads in the browser.
const ProspectingMap = dynamic(() => import("./ProspectingMap"), {
  ssr: false,
  loading: () => <div className="rounded-2xl border p-10 text-center text-sm" style={{ borderColor: "var(--line)", color: "var(--muted)" }}>Loading map…</div>,
});

type SubView = "map" | "cards" | "tracker";

export function ProspectingPanel({ tracker }: { tracker: ReturnType<typeof useTracker> }) {
  const [sub, setSub] = useState<SubView>("cards");
  const {
    prospects, mappable, trackedIds, selected, loading, error, queueing, queuedCount,
    toggleSelect, queueSelected, addToTracker, refresh,
  } = useProspects();

  const tabs: { v: SubView; label: string }[] = [
    { v: "map", label: `🗺 Map (${mappable.length})` },
    { v: "cards", label: `▤ Cards (${prospects.length})` },
    { v: "tracker", label: `🗂 Tracker (${tracker.rows.length})` },
  ];

  return (
    <div>
      <div className="mb-3 flex items-center gap-2">
        <div role="tablist" aria-label="Prospecting view" className="inline-flex overflow-hidden rounded-lg border" style={{ borderColor: "var(--line)" }}>
          {tabs.map(({ v, label }) => (
            <button key={v} role="tab" aria-selected={sub === v} onClick={() => setSub(v)} className="px-3 py-1 text-sm"
              style={sub === v ? { background: "var(--accent)", color: "var(--accent-ink)" } : { background: "var(--surface)" }}>
              {label}
            </button>
          ))}
        </div>
      </div>

      <ProspectingControls queuedCount={queuedCount} onRefresh={refresh} />

      {error && <div className="mb-3 rounded-xl border p-3 text-sm" style={{ borderColor: "var(--warn)" }}>Prospecting error: {error}</div>}

      {sub === "tracker" ? (
        <TrackerTable tracker={tracker} />
      ) : loading ? (
        <Skeleton />
      ) : prospects.length === 0 ? (
        <p className="rounded-2xl border p-10 text-center text-sm" style={{ borderColor: "var(--line)", color: "var(--muted)" }}>
          No prospects yet — run <code>poetry run python src/discover.py</code> to find Koh Phangan places on Google Maps.
        </p>
      ) : sub === "map" ? (
        <ProspectingMap prospects={mappable} selected={selected} onToggle={toggleSelect} onQueue={queueSelected} queueing={queueing} />
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {prospects.map((p) => (
            <ProspectCard key={p.place_id} prospect={p} tracked={trackedIds.has(p.place_id)} onAddToTracker={addToTracker} />
          ))}
        </div>
      )}
    </div>
  );
}
