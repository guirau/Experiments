"use client";
import { useState } from "react";
import { areaName } from "@/lib/areas";
import { AREA_POINTS, ISLAND_PATH, NON_MAP_AREAS } from "@/lib/areaGeo";

function radius(count: number): number {
  return 3 + Math.min(5, Math.sqrt(count)); // ~3 (empty) … ~8 (busy)
}

export function AreaMap({ counts, selected, onToggle }:
  { counts: Record<string, number>; selected: string[]; onToggle: (slug: string) => void }) {
  const [hover, setHover] = useState<string | null>(null);

  return (
    <div className="rounded-2xl border p-4" style={{ borderColor: "var(--line)", background: "var(--surface)" }}>
      <p className="mb-2 text-sm" style={{ color: "var(--muted)" }}>
        Koh Phangan areas — click to filter (multi-select). Numbers are matching listings.
      </p>
      <svg viewBox="0 0 100 100" className="mx-auto block h-auto w-full max-w-xl" role="group" aria-label="Koh Phangan area map">
        <path d={ISLAND_PATH} fill="var(--bg)" stroke="var(--line)" strokeWidth={0.6} />
        {AREA_POINTS.map(({ slug, x, y }) => {
          const count = counts[slug] ?? 0;
          const isSel = selected.includes(slug);
          const isHover = hover === slug;
          const r = radius(count);
          const name = areaName(slug);
          return (
            <g key={slug} role="button" tabIndex={0} aria-pressed={isSel}
              aria-label={`${name}: ${count} listings`}
              style={{ cursor: "pointer" }}
              onClick={() => onToggle(slug)}
              onMouseEnter={() => setHover(slug)} onMouseLeave={() => setHover(null)}
              onFocus={() => setHover(slug)} onBlur={() => setHover(null)}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onToggle(slug); } }}>
              <title>{`${name}: ${count} listings`}</title>
              <circle cx={x} cy={y} r={r}
                fill={isSel ? "var(--accent)" : "var(--surface)"}
                stroke={isSel ? "var(--accent)" : "var(--accent)"}
                strokeWidth={isHover || isSel ? 1.1 : 0.7}
                opacity={isHover && !isSel ? 0.9 : 1} />
              <text x={x} y={y + r * 0.45} textAnchor="middle" fontSize={r * 0.95} fontWeight={700}
                fill={isSel ? "var(--accent-ink)" : "var(--ink)"} style={{ pointerEvents: "none" }}>
                {count}
              </text>
              <text x={x} y={y + r + 2.6} textAnchor="middle" fontSize={2.3}
                fill={isHover || isSel ? "var(--ink)" : "var(--muted)"} fontWeight={isHover || isSel ? 600 : 400}
                style={{ pointerEvents: "none" }}>
                {name}
              </text>
            </g>
          );
        })}
      </svg>

      <div className="mt-3 flex flex-wrap items-center gap-1.5">
        <span className="text-xs" style={{ color: "var(--muted)" }}>No location:</span>
        {NON_MAP_AREAS.map((slug) => {
          const isSel = selected.includes(slug);
          return (
            <button key={slug} type="button" aria-pressed={isSel} onClick={() => onToggle(slug)}
              className="rounded-full border px-2.5 py-1 text-xs"
              style={isSel ? { background: "var(--accent)", color: "var(--accent-ink)", borderColor: "var(--accent)" } : { borderColor: "var(--line)" }}>
              {areaName(slug)} ({counts[slug] ?? 0})
            </button>
          );
        })}
      </div>
    </div>
  );
}
