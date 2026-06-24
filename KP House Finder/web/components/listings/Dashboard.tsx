"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useListings } from "@/hooks/useListings";
import { useFilters } from "@/hooks/useFilters";
import { useCollections } from "@/hooks/useCollections";
import type { Listing } from "@/lib/types";
import { applyFilters, sortListings, countByArea } from "@/lib/filters";
import { updatePrice, updateListing } from "@/lib/supabase";
import { FilterSidebar } from "@/components/filters/FilterSidebar";
import { AreaMap } from "@/components/map/AreaMap";
import { EditModal } from "./EditModal";
import { ListingGrid } from "./ListingGrid";
import { SortBar } from "./SortBar";
import { ActiveChips } from "./ActiveChips";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";

type View = "list" | "map" | "saved" | "contacted" | "removed";

export function Dashboard() {
  const { listings, loading, error, updateLocal } = useListings();
  const { filters, setFilters, reset } = useFilters();
  const [drawer, setDrawer] = useState(false);
  const [view, setView] = useState<View>("list");
  const [editing, setEditing] = useState<Listing | null>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);
  const { removed, saved, contacted, remove, restore, undoRemove, toggleSave, toggleContacted } = useCollections();
  const removedSet = useMemo(() => new Set(removed), [removed]);
  const savedSet = useMemo(() => new Set(saved), [saved]);
  const contactedSet = useMemo(() => new Set(contacted), [contacted]);

  // main list: filtered + sorted, hiding removed (dismissed) and contacted (already actioned)
  const visible = useMemo(
    () => sortListings(applyFilters(listings, filters), filters.sort).filter((l) => !removedSet.has(l.id) && !contactedSet.has(l.id)),
    [listings, filters, removedSet, contactedSet]);
  // collections (removed takes precedence over the others)
  const savedList = useMemo(() => listings.filter((l) => savedSet.has(l.id) && !removedSet.has(l.id)), [listings, savedSet, removedSet]);
  const contactedList = useMemo(() => listings.filter((l) => contactedSet.has(l.id) && !removedSet.has(l.id)), [listings, contactedSet, removedSet]);
  const removedList = useMemo(() => listings.filter((l) => removedSet.has(l.id)), [listings, removedSet]);
  // area counts reflect other active filters AND match the visible list (hide removed + contacted).
  const areaCounts = useMemo(
    () => countByArea(applyFilters(listings, { ...filters, areas: [] }).filter((l) => !removedSet.has(l.id) && !contactedSet.has(l.id))),
    [listings, filters, removedSet, contactedSet]);
  const toggleArea = (slug: string) =>
    setFilters({ ...filters, areas: filters.areas.includes(slug) ? filters.areas.filter((a) => a !== slug) : [...filters.areas, slug] });

  // Edit price: write to Supabase first, then sync the in-memory listing (throws on
  // failure so the card can surface it and keep editing).
  const handleEditPrice = useCallback(async (id: string, price: number | null) => {
    await updatePrice(id, price);
    updateLocal(id, { price_thb: price });
  }, [updateLocal]);

  // Edit-modal save: write the full field patch to Supabase, then sync local state.
  const handleEditListing = useCallback(async (id: string, patch: Record<string, unknown>) => {
    await updateListing(id, patch);
    updateLocal(id, patch as Partial<Listing>);
  }, [updateLocal]);

  // Mobile filter drawer: close on Escape and move focus into it when opened.
  useEffect(() => {
    if (!drawer) return;
    closeBtnRef.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setDrawer(false); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [drawer]);

  // Cmd/Ctrl+Z undoes the last removal (ignored while typing in a field).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undoRemove(); }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [undoRemove]);

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
              {([
                { v: "list", label: "List" },
                { v: "map", label: "Map" },
                { v: "saved", label: `★ Saved (${savedList.length})` },
                { v: "contacted", label: `✓ Contacted (${contactedList.length})` },
                { v: "removed", label: `Removed (${removedList.length})` },
              ] as const).map(({ v, label }) => (
                <button key={v} role="tab" aria-selected={view === v} onClick={() => setView(v)}
                  className="px-3 py-1 text-sm"
                  style={view === v ? { background: "var(--accent)", color: "var(--accent-ink)" } : { background: "var(--surface)" }}>{label}</button>
              ))}
            </div>
            {view === "list"
              ? <SortBar count={visible.length} sort={filters.sort} onSort={(s) => setFilters({ ...filters, sort: s })} />
              : <span className="text-sm" style={{ color: "var(--muted)" }}>
                  {(view === "saved" ? savedList.length : view === "contacted" ? contactedList.length : view === "removed" ? removedList.length : visible.length).toLocaleString()} listings
                </span>}
          </div>

          {(view === "list" || view === "map") && <div className="mt-3"><ActiveChips filters={filters} setFilters={setFilters} /></div>}

          {view === "map" ? (
            <AreaMap counts={areaCounts} selected={filters.areas} onToggle={toggleArea} />
          ) : view === "saved" ? (
            savedList.length === 0
              ? <p className="rounded-2xl border p-10 text-center text-sm" style={{ borderColor: "var(--line)", color: "var(--muted)" }}>No saved listings yet — tap ☆ Save on a card.</p>
              : <ListingGrid listings={savedList} savedSet={savedSet} contactedSet={contactedSet} onSave={toggleSave} onContacted={toggleContacted} onRemove={remove} onEditPrice={handleEditPrice} onOpen={setEditing} />
          ) : view === "contacted" ? (
            contactedList.length === 0
              ? <p className="rounded-2xl border p-10 text-center text-sm" style={{ borderColor: "var(--line)", color: "var(--muted)" }}>No contacted listings yet — tap ✆ Contacted on a card.</p>
              : <ListingGrid listings={contactedList} savedSet={savedSet} contactedSet={contactedSet} onSave={toggleSave} onContacted={toggleContacted} onRemove={remove} onEditPrice={handleEditPrice} onOpen={setEditing} />
          ) : view === "removed" ? (
            removedList.length === 0
              ? <p className="rounded-2xl border p-10 text-center text-sm" style={{ borderColor: "var(--line)", color: "var(--muted)" }}>Nothing removed. Use ✕ on a card to hide listings you’re not interested in.</p>
              : <ListingGrid listings={removedList} onRestore={restore} onOpen={setEditing} />
          ) : loading ? (
            <Skeleton />
          ) : visible.length === 0 ? (
            <EmptyState onClear={reset} />
          ) : (
            <ListingGrid listings={visible} savedSet={savedSet} contactedSet={contactedSet} onSave={toggleSave} onContacted={toggleContacted} onRemove={remove} onEditPrice={handleEditPrice} onOpen={setEditing} />
          )}
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

      {editing && <EditModal listing={editing} onClose={() => setEditing(null)} onSave={handleEditListing} />}
    </div>
  );
}
