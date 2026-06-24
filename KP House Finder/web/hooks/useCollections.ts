"use client";
import { useCallback, useEffect, useRef, useState } from "react";

const REMOVED_KEY = "kp:removed";
const SAVED_KEY = "kp:saved";
const CONTACTED_KEY = "kp:contacted";

function read(key: string): string[] {
  try {
    const v = JSON.parse(localStorage.getItem(key) ?? "[]");
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

// Per-browser collections persisted in localStorage (listings_parsed is never mutated).
// `removed` listings are hidden from the main list; `saved` are the favourites shortlist.
export function useCollections() {
  const [removed, setRemoved] = useState<string[]>([]);
  const [saved, setSaved] = useState<string[]>([]);
  const [contacted, setContacted] = useState<string[]>([]);
  const [hydrated, setHydrated] = useState(false);
  const undoStack = useRef<string[]>([]);

  // hydrate once on the client (SSR-safe: server + first render use []). Intentional
  // setState-in-effect — localStorage isn't available until after mount.
  useEffect(() => {
    /* eslint-disable react-hooks/set-state-in-effect */
    setRemoved(read(REMOVED_KEY));
    setSaved(read(SAVED_KEY));
    setContacted(read(CONTACTED_KEY));
    setHydrated(true);
    /* eslint-enable react-hooks/set-state-in-effect */
  }, []);
  // persist only after hydration so the initial empty render can't clobber storage
  useEffect(() => { if (hydrated) localStorage.setItem(REMOVED_KEY, JSON.stringify(removed)); }, [removed, hydrated]);
  useEffect(() => { if (hydrated) localStorage.setItem(SAVED_KEY, JSON.stringify(saved)); }, [saved, hydrated]);
  useEffect(() => { if (hydrated) localStorage.setItem(CONTACTED_KEY, JSON.stringify(contacted)); }, [contacted, hydrated]);

  const remove = useCallback((id: string) => {
    setRemoved((prev) => (prev.includes(id) ? prev : [...prev, id]));
    undoStack.current.push(id);
  }, []);

  const restore = useCallback((id: string) => {
    setRemoved((prev) => prev.filter((x) => x !== id));
    undoStack.current = undoStack.current.filter((x) => x !== id);
  }, []);

  const undoRemove = useCallback(() => {
    const id = undoStack.current.pop();
    if (id) setRemoved((prev) => prev.filter((x) => x !== id));
  }, []);

  const toggleSave = useCallback((id: string) => {
    setSaved((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }, []);

  const toggleContacted = useCallback((id: string) => {
    setContacted((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }, []);

  return { removed, saved, contacted, remove, restore, undoRemove, toggleSave, toggleContacted };
}
