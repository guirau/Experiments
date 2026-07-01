"use client";
import type { SortKey } from "@/lib/types";
const OPTS: { k: SortKey; label: string }[] = [
  { k: "newest", label: "Newest" }, { k: "recent", label: "Recently added" },
  { k: "price_asc", label: "Price ↑" }, { k: "price_desc", label: "Price ↓" },
  { k: "confidence", label: "Confidence" },
];
export function SortBar({ count, sort, onSort }: { count: number; sort: SortKey; onSort: (s: SortKey) => void }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm" style={{ color: "var(--muted)" }}>{count.toLocaleString()} listings</span>
      <label className="text-sm">Sort{" "}
        <select value={sort} onChange={(e) => onSort(e.target.value as SortKey)} className="rounded-lg border px-2 py-1" style={{ borderColor: "var(--line)" }}>
          {OPTS.map((o) => <option key={o.k} value={o.k}>{o.label}</option>)}
        </select>
      </label>
    </div>
  );
}
