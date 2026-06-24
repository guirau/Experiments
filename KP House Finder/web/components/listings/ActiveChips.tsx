"use client";
import type { FilterState } from "@/lib/types";
import { defaultFilters } from "@/lib/filters";
import { areaName } from "@/lib/areas";
import { Chip } from "@/components/ui/Chip";
export function ActiveChips({ filters, setFilters }: { filters: FilterState; setFilters: (f: FilterState) => void }) {
  const d = defaultFilters();
  const chips: { label: string; clear: Partial<FilterState> }[] = [];
  if (JSON.stringify(filters.listingType) !== JSON.stringify(d.listingType))
    chips.push({ label: filters.listingType.length ? filters.listingType.join(" + ") : "rent + sale", clear: { listingType: d.listingType } });
  if (filters.parsedWithin) chips.push({ label: `parsed ≤ ${filters.parsedWithin}d`, clear: { parsedWithin: "" } });
  if (filters.priceMin != null) chips.push({ label: `≥ ฿${filters.priceMin}`, clear: { priceMin: null } });
  if (filters.priceMax != null) chips.push({ label: `≤ ฿${filters.priceMax}`, clear: { priceMax: null } });
  filters.areas.forEach((a) => chips.push({ label: areaName(a), clear: { areas: filters.areas.filter((x) => x !== a) } }));
  if (filters.yearRound.length) chips.push({ label: `year-round: ${filters.yearRound.join("/")}`, clear: { yearRound: d.yearRound } });
  if (filters.subletting.length) chips.push({ label: `sublet: ${filters.subletting.join("/")}`, clear: { subletting: d.subletting } });
  filters.amenities.forEach((a) => chips.push({ label: a, clear: { amenities: filters.amenities.filter((x) => x !== a) } }));
  if (!chips.length) return null;
  return <div className="mb-3 flex flex-wrap gap-1.5">{chips.map((c) => <Chip key={c.label} label={c.label} onRemove={() => setFilters({ ...filters, ...c.clear })} />)}</div>;
}
