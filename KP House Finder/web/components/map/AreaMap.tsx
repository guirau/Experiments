"use client";
import { useState } from "react";
import Image from "next/image";
import { areaName } from "@/lib/areas";
import { AREA_POINTS, NON_MAP_AREAS } from "@/lib/areaGeo";
import mapImg from "@/assets/map.png";

const MARKER = 34; // px — every marker is the same size

export function AreaMap({ counts, selected, onToggle }:
  { counts: Record<string, number>; selected: string[]; onToggle: (slug: string) => void }) {
  const [hover, setHover] = useState<string | null>(null);

  return (
    <div className="rounded-2xl border p-4" style={{ borderColor: "var(--line)", background: "var(--surface)" }}>
      <p className="mb-2 text-sm" style={{ color: "var(--muted)" }}>
        Koh Phangan areas — click to filter (multi-select). Numbers are matching listings.
      </p>

      <div className="relative mx-auto w-full max-w-2xl" style={{ aspectRatio: `${mapImg.width} / ${mapImg.height}` }}>
        <Image src={mapImg} alt="Map of Koh Phangan" fill priority sizes="(max-width: 768px) 100vw, 700px"
          className="rounded-xl object-contain" />

        {AREA_POINTS.map(({ slug, x, y }) => {
          const count = counts[slug] ?? 0;
          const isSel = selected.includes(slug);
          const isHover = hover === slug;
          const name = areaName(slug);
          return (
            <button key={slug} type="button" aria-pressed={isSel} aria-label={`${name}: ${count} listings`}
              onClick={() => onToggle(slug)}
              onMouseEnter={() => setHover(slug)} onMouseLeave={() => setHover(null)}
              onFocus={() => setHover(slug)} onBlur={() => setHover(null)}
              className="absolute flex items-center justify-center rounded-full border text-xs font-bold transition-transform"
              style={{
                left: `${x}%`, top: `${y}%`, width: MARKER, height: MARKER,
                transform: `translate(-50%, -50%) scale(${isHover && !isSel ? 1.12 : 1})`,
                background: isSel ? "var(--accent)" : "color-mix(in oklch, var(--surface) 88%, transparent)",
                color: isSel ? "var(--accent-ink)" : "var(--ink)",
                borderColor: "var(--accent)",
                borderWidth: isSel || isHover ? 2 : 1,
                boxShadow: "var(--shadow)",
                zIndex: isHover || isSel ? 2 : 1,
              }}>
              {count}
              {(isHover || isSel) && (
                <span className="absolute left-1/2 top-full mt-1 -translate-x-1/2 whitespace-nowrap rounded px-1.5 py-0.5 text-[10px] font-medium"
                  style={{ background: "var(--ink)", color: "var(--bg)" }}>
                  {name}
                </span>
              )}
            </button>
          );
        })}
      </div>

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
