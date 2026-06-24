"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useListings } from "@/hooks/useListings";
import { useFilters } from "@/hooks/useFilters";
import { applyFilters, sortListings, countByArea } from "@/lib/filters";
import { FilterSidebar } from "@/components/filters/FilterSidebar";
import { AreaMap } from "@/components/map/AreaMap";
import { ListingGrid } from "./ListingGrid";
import { SortBar } from "./SortBar";
import { ActiveChips } from "./ActiveChips";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";

type View = "list" | "map";

export function Dashboard() {
  const { listings, loading, error } = useListings();
  const { filters, setFilters, reset } = useFilters();
  const [drawer, setDrawer] = useState(false);
  const [view, setView] = useState<View>("list");
  const closeBtnRef = useRef<HTMLButtonElement>(null);
  const results = useMemo(() => sortListings(applyFilters(listings, filters), filters.sort), [listings, filters]);
  // counts per area reflect all OTHER active filters (area filter blanked), so the map
  // stays informative even while areas are selected.
  const areaCounts = useMemo(() => countByArea(applyFilters(listings, { ...filters, areas: [] })), [listings, filters]);
  const toggleArea = (slug: string) =>
    setFilters({ ...filters, areas: filters.areas.includes(slug) ? filters.areas.filter((a) => a !== slug) : [...filters.areas, slug] });

  // Mobile filter drawer: close on Escape and move focus into it when opened.
  useEffect(() => {
    if (!drawer) return;
    closeBtnRef.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setDrawer(false); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [drawer]);

  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      <header className="sticky top-0 z-10 -mx-4 mb-4 flex items-center justify-between border-b px-4 py-3 backdrop-blur" style={{ borderColor: "var(--line)", background: "color-mix(in oklch, var(--bg) 80%, transparent)" }}>
        <h1 className="text-lg font-bold tracking-tight">KP House Finder</h1>
        <button className="rounded-full border px-3 py-1 text-sm lg:hidden" style={{ borderColor: "var(--line)" }} onClick={() => setDrawer(true)}>Filters</button>
      </header>

      {error && <div className="rounded-xl border p-4 text-sm" style={{ borderColor: "var(--warn)" }}>Couldn&apos;t load listings: {error}</div>}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[260px_1fr]">
        <div className="hidden lg:block"><div className="sticky top-20 max-h-[calc(100vh-6rem)] overflow-y-auto overscroll-contain pr-1"><FilterSidebar filters={filters} setFilters={setFilters} onClear={reset} /></div></div>

        <main>
          <div className="mb-3 flex items-center justify-between gap-3">
            <div role="tablist" aria-label="View" className="inline-flex overflow-hidden rounded-lg border" style={{ borderColor: "var(--line)" }}>
              {(["list", "map"] as const).map((v) => (
                <button key={v} role="tab" aria-selected={view === v} onClick={() => setView(v)}
                  className="px-3 py-1 text-sm capitalize"
                  style={view === v ? { background: "var(--accent)", color: "var(--accent-ink)" } : { background: "var(--surface)" }}>{v}</button>
              ))}
            </div>
            {view === "list"
              ? <SortBar count={results.length} sort={filters.sort} onSort={(s) => setFilters({ ...filters, sort: s })} />
              : <span className="text-sm" style={{ color: "var(--muted)" }}>{results.length.toLocaleString()} listings</span>}
          </div>
          <div className="mt-3"><ActiveChips filters={filters} setFilters={setFilters} /></div>
          {view === "map"
            ? <AreaMap counts={areaCounts} selected={filters.areas} onToggle={toggleArea} />
            : loading ? <Skeleton /> : results.length === 0 ? <EmptyState onClear={reset} /> : <ListingGrid listings={results} />}
        </main>
      </div>

      {drawer && (
        <div className="fixed inset-0 z-20 lg:hidden">
          <div className="absolute inset-0 bg-black/30" onClick={() => setDrawer(false)} />
          <div role="dialog" aria-modal="true" aria-label="Filters" className="absolute left-0 top-0 h-full w-80 overflow-y-auto p-4" style={{ background: "var(--surface)" }}>
            <button ref={closeBtnRef} aria-label="Close filters" className="mb-2 text-sm underline" onClick={() => setDrawer(false)}>Close</button>
            <FilterSidebar filters={filters} setFilters={setFilters} onClear={reset} />
          </div>
        </div>
      )}
    </div>
  );
}
