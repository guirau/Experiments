"use client";
import { MapContainer, TileLayer, CircleMarker, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import type { Prospect } from "@/lib/types";

const KP_CENTER: [number, number] = [9.75, 100.03];

// Marker colour communicates state at a glance: selection wins, then enrichment status.
function markerColor(p: Prospect, selected: boolean): string {
  if (selected) return "var(--accent)";
  if (p.enrich_status === "enriched") return "var(--good)";
  if (p.enrich_status === "queued") return "var(--warn)";
  return "var(--muted)";
}

interface ProspectingMapProps {
  prospects: Prospect[];
  selected: Set<string>;
  onToggle: (placeId: string) => void;
  onQueue: () => void;
  queueing: boolean;
}

export default function ProspectingMap({ prospects, selected, onToggle, onQueue, queueing }: ProspectingMapProps) {
  const count = selected.size;
  return (
    <div className="relative overflow-hidden rounded-2xl border" style={{ borderColor: "var(--line)", height: "70vh" }}>
      <MapContainer center={KP_CENTER} zoom={12} scrollWheelZoom style={{ height: "100%", width: "100%" }}>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {prospects.map((p) => {
          const isSel = selected.has(p.place_id);
          const color = markerColor(p, isSel);
          return (
            <CircleMarker
              key={p.place_id}
              center={[p.lat as number, p.lng as number]}
              radius={isSel ? 9 : 6}
              pathOptions={{ color, fillColor: color, fillOpacity: isSel ? 0.9 : 0.6, weight: isSel ? 3 : 1 }}
              eventHandlers={{ click: () => onToggle(p.place_id) }}
            >
              <Tooltip>
                <strong>{p.name ?? "Unnamed"}</strong>
                {p.google_rating != null && <> · {p.google_rating}★</>}
                {p.suitability_score != null && <> · fit {p.suitability_score}/10</>}
                {p.rate_per_night_thb != null && <> · ~฿{p.rate_per_night_thb.toLocaleString()}/night</>}
                <br />
                <span style={{ opacity: 0.7 }}>{isSel ? "Selected — click to deselect" : "Click to select"}</span>
              </Tooltip>
            </CircleMarker>
          );
        })}
      </MapContainer>

      {/* floating action bar: reflects the current selection + fires the queue action */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 z-[1000] flex justify-center p-3">
        <div className="pointer-events-auto flex items-center gap-3 rounded-full border px-4 py-2 shadow-[var(--shadow)]"
          style={{ borderColor: "var(--line)", background: "color-mix(in oklch, var(--surface) 92%, transparent)" }}>
          <span className="text-sm" style={{ color: "var(--muted)" }}>
            {count === 0 ? "Click pins to select places to price" : `${count} selected`}
          </span>
          <button onClick={onQueue} disabled={count === 0 || queueing}
            className={`card-btn${count > 0 ? " card-btn-on" : ""}`}>
            {queueing ? "Queuing…" : `Queue ${count || ""} for enrichment`}
          </button>
        </div>
      </div>
    </div>
  );
}
