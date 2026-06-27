"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { TrackerRow } from "@/lib/types";
import { fetchTracker, saveTracker } from "@/lib/supabase";

function errMsg(e: unknown): string {
  if (e instanceof Error) return e.message;
  if (e && typeof e === "object" && "message" in e) return String((e as { message: unknown }).message);
  return String(e);
}

const blank = (): TrackerRow => ({
  id: crypto.randomUUID(), listing_url: "", person_name: "", price: null, location_url: "",
  visit_date: "", contact: "", notify_before: "", notes: "", crossed_off: false, sort_order: null,
});

export function useTracker() {
  const [rows, setRows] = useState<TrackerRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);
  const deleted = useRef<Set<string>>(new Set());

  useEffect(() => {
    let on = true;
    fetchTracker()
      .then((r) => on && setRows(r))
      .catch((e) => on && setError(errMsg(e)))
      .finally(() => on && setLoading(false));
    return () => { on = false; };
  }, []);

  const addRow = useCallback(() => { setRows((r) => [...r, blank()]); setDirty(true); }, []);
  const updateRow = useCallback((id: string, patch: Partial<TrackerRow>) => {
    setRows((r) => r.map((x) => (x.id === id ? { ...x, ...patch } : x)));
    setDirty(true);
  }, []);
  const removeRow = useCallback((id: string) => {
    setRows((r) => r.filter((x) => x.id !== id));
    deleted.current.add(id);
    setDirty(true);
  }, []);
  const toggleCrossed = useCallback((id: string) => {
    setRows((r) => r.map((x) => (x.id === id ? { ...x, crossed_off: !x.crossed_off } : x)));
    setDirty(true);
  }, []);
  // move the dragged row to the dropped row's position
  const moveRow = useCallback((dragId: string, dropId: string) => {
    if (dragId === dropId) return;
    setRows((r) => {
      const from = r.findIndex((x) => x.id === dragId);
      const to = r.findIndex((x) => x.id === dropId);
      if (from < 0 || to < 0) return r;
      const next = [...r];
      const [moved] = next.splice(from, 1);
      next.splice(to, 0, moved);
      return next;
    });
    setDirty(true);
  }, []);
  const save = useCallback(async () => {
    setSaving(true); setError(null);
    try {
      const payload = rows.map((r, i) => ({ ...r, sort_order: i })); // persist current order
      await saveTracker(payload, [...deleted.current]);
      setRows(payload);
      deleted.current.clear();
      setDirty(false);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setSaving(false);
    }
  }, [rows]);

  return { rows, loading, error, saving, dirty, addRow, updateRow, removeRow, toggleCrossed, moveRow, save };
}
