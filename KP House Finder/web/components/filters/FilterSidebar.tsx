"use client";
import type { FilterState } from "@/lib/types";
import { AREA_ENUM, PROPERTY_TYPES, SEASONS, CONFIDENCES, LANGUAGES } from "@/lib/constants";
import { areaName } from "@/lib/areas";
import { PriceRange } from "./PriceRange";
import { MultiSelect } from "./MultiSelect";
import { TriStateToggle } from "./TriStateToggle";
import { NumberMax } from "./NumberMax";
import { AmenityToggles } from "./AmenityToggles";

function Section({ children }: { children: React.ReactNode }) {
  return <h3 className="mb-2 mt-4 text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--muted)" }}>{children}</h3>;
}

export function FilterSidebar({ filters, setFilters, onClear }:
  { filters: FilterState; setFilters: (f: FilterState) => void; onClear: () => void }) {
  const set = (patch: Partial<FilterState>) => setFilters({ ...filters, ...patch });
  return (
    <aside className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">Filters</h2>
        <button onClick={onClear} className="text-xs underline" style={{ color: "var(--muted)" }}>Clear all</button>
      </div>
      <Section>Core</Section>
      <PriceRange min={filters.priceMin} max={filters.priceMax} onChange={(priceMin, priceMax) => set({ priceMin, priceMax })} />
      <MultiSelect label="Area" options={AREA_ENUM} selected={filters.areas} onChange={(areas) => set({ areas })} render={areaName} />
      <MultiSelect label="Type" options={PROPERTY_TYPES} selected={filters.propertyTypes} onChange={(propertyTypes) => set({ propertyTypes })} />
      <NumberMax label="Min bedrooms" value={filters.bedroomsMin} onChange={(bedroomsMin) => set({ bedroomsMin })} />
      <NumberMax label="Min bathrooms" value={filters.bathroomsMin} onChange={(bathroomsMin) => set({ bathroomsMin })} />
      <Section>Long-term rental</Section>
      <TriStateToggle label="Year-round" value={filters.yearRound} onChange={(yearRound) => set({ yearRound })} />
      <MultiSelect label="Season" options={SEASONS} selected={filters.seasons} onChange={(seasons) => set({ seasons })} />
      <NumberMax label="Max min-stay (months)" value={filters.minStayMax} onChange={(minStayMax) => set({ minStayMax })} />
      <Section>Sublet potential</Section>
      <TriStateToggle label="Subletting allowed" value={filters.subletting} onChange={(subletting) => set({ subletting })} />
      <Section>Cost</Section>
      <NumberMax label="Max deposit ฿" value={filters.depositMax} onChange={(depositMax) => set({ depositMax })} />
      <TriStateToggle label="Water included" value={filters.waterIncluded} onChange={(waterIncluded) => set({ waterIncluded })} />
      <TriStateToggle label="Internet included" value={filters.internetIncluded} onChange={(internetIncluded) => set({ internetIncluded })} />
      <Section>Amenities</Section>
      <AmenityToggles selected={filters.amenities} onChange={(amenities) => set({ amenities })} />
      <Section>Quality / language</Section>
      <MultiSelect label="Confidence" options={CONFIDENCES} selected={filters.confidences} onChange={(confidences) => set({ confidences })} />
      <MultiSelect label="Language" options={LANGUAGES} selected={filters.languages} onChange={(languages) => set({ languages })} />
    </aside>
  );
}
