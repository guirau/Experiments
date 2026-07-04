"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { Prospect } from "@/lib/types";
import {
  fetchProspects, fetchTrackedProspectIds, queueForEnrichment, createTrackerFromProspect,
} from "@/lib/supabase";

function errMsg(e: unknown): string {
  if (e instanceof Error) return e.message;
  if (e && typeof e === "object" && "message" in e) return String((e as { message: unknown }).message);
  return String(e);
}

export function useProspects() {
  const [prospects, setProspects] = useState<Prospect[]>([]);
  const [trackedIds, setTrackedIds] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [queueing, setQueueing] = useState(false);

  useEffect(() => {
    let on = true;
    Promise.all([fetchProspects(), fetchTrackedProspectIds()])
      .then(([rows, tracked]) => {
        if (!on) return;
        setProspects(rows);
        setTrackedIds(tracked);
      })
      .catch((e) => on && setError(errMsg(e)))
      .finally(() => on && setLoading(false));
    return () => { on = false; };
  }, []);

  // Re-pull from Supabase after a discover/enrich run so new + newly-priced rows show up.
  const refresh = useCallback(() => {
    Promise.all([fetchProspects(), fetchTrackedProspectIds()])
      .then(([rows, tracked]) => { setProspects(rows); setTrackedIds(tracked); })
      .catch((e) => setError(errMsg(e)));
  }, []);

  const toggleSelect = useCallback((placeId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(placeId)) next.delete(placeId); else next.add(placeId);
      return next;
    });
  }, []);
  const clearSelection = useCallback(() => setSelected(new Set()), []);

  // Queue the selected places, then optimistically flip their status locally.
  const queueSelected = useCallback(async () => {
    const ids = [...selected];
    if (!ids.length) return;
    setQueueing(true); setError(null);
    try {
      await queueForEnrichment(ids);
      const idSet = new Set(ids);
      setProspects((rows) => rows.map((p) => (idSet.has(p.place_id) ? { ...p, enrich_status: "queued" } : p)));
      setSelected(new Set());
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setQueueing(false);
    }
  }, [selected]);

  const addToTracker = useCallback(async (p: Prospect) => {
    setError(null);
    try {
      await createTrackerFromProspect(p);
      setTrackedIds((prev) => new Set(prev).add(p.place_id));
    } catch (e) {
      setError(errMsg(e));
      throw e;
    }
  }, []);

  // prospects that carry real coordinates can be shown on the map.
  const mappable = useMemo(
    () => prospects.filter((p) => p.lat != null && p.lng != null),
    [prospects]);
  // how many are queued for the paid enrich run (drives the button label + cost confirm).
  const queuedCount = useMemo(
    () => prospects.filter((p) => p.enrich_status === "queued").length,
    [prospects]);

  return {
    prospects, mappable, trackedIds, selected, loading, error, queueing, queuedCount,
    toggleSelect, clearSelection, queueSelected, addToTracker, refresh,
  };
}
