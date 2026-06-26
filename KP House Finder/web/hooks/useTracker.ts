"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { TrackerRow } from "@/lib/types";
import { fetchTracker, saveTracker } from "@/lib/supabase";

const blank = (): TrackerRow => ({
  id: crypto.randomUUID(), listing_url: "", person_name: "", price: null, location_url: "",
  visit_date: "", contact: "", notify_before: "", notes: "",
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
      .catch((e) => on && setError(e instanceof Error ? e.message : String(e)))
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
  const save = useCallback(async () => {
    setSaving(true); setError(null);
    try {
      await saveTracker(rows, [...deleted.current]);
      deleted.current.clear();
      setDirty(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }, [rows]);

  return { rows, loading, error, saving, dirty, addRow, updateRow, removeRow, save };
}
