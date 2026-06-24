"use client";
import { useMemo, useState } from "react";
import { useListings } from "@/hooks/useListings";
import { useFilters } from "@/hooks/useFilters";
import { applyFilters, sortListings } from "@/lib/filters";
import { FilterSidebar } from "@/components/filters/FilterSidebar";
import { ListingGrid } from "./ListingGrid";
import { SortBar } from "./SortBar";
import { ActiveChips } from "./ActiveChips";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";

export function Dashboard() {
  const { listings, loading, error } = useListings();
  const { filters, setFilters, reset } = useFilters();
  const [drawer, setDrawer] = useState(false);
  const results = useMemo(() => sortListings(applyFilters(listings, filters), filters.sort), [listings, filters]);

  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      <header className="sticky top-0 z-10 -mx-4 mb-4 flex items-center justify-between border-b px-4 py-3 backdrop-blur" style={{ borderColor: "var(--line)", background: "color-mix(in oklch, var(--bg) 80%, transparent)" }}>
        <h1 className="text-lg font-bold tracking-tight">KP House Finder</h1>
        <button className="rounded-full border px-3 py-1 text-sm lg:hidden" style={{ borderColor: "var(--line)" }} onClick={() => setDrawer(true)}>Filters</button>
      </header>

      {error && <div className="rounded-xl border p-4 text-sm" style={{ borderColor: "var(--warn)" }}>Couldn&apos;t load listings: {error}</div>}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[260px_1fr]">
        <div className="hidden lg:block"><div className="sticky top-20"><FilterSidebar filters={filters} setFilters={setFilters} onClear={reset} /></div></div>

        <main>
          <SortBar count={results.length} sort={filters.sort} onSort={(s) => setFilters({ ...filters, sort: s })} />
          <div className="mt-3"><ActiveChips filters={filters} setFilters={setFilters} /></div>
          {loading ? <Skeleton /> : results.length === 0 ? <EmptyState onClear={reset} /> : <ListingGrid listings={results} />}
        </main>
      </div>

      {drawer && (
        <div className="fixed inset-0 z-20 lg:hidden">
          <div className="absolute inset-0 bg-black/30" onClick={() => setDrawer(false)} />
          <div className="absolute left-0 top-0 h-full w-80 overflow-y-auto p-4" style={{ background: "var(--surface)" }}>
            <button className="mb-2 text-sm underline" onClick={() => setDrawer(false)}>Close</button>
            <FilterSidebar filters={filters} setFilters={setFilters} onClear={reset} />
          </div>
        </div>
      )}
    </div>
  );
}
