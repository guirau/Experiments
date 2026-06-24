"use client";
import { useEffect, useState } from "react";
import { fetchListings } from "@/lib/supabase";
import type { Listing } from "@/lib/types";

export function useListings() {
  const [listings, setListings] = useState<Listing[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let on = true;
    fetchListings()
      .then((rows) => on && setListings(rows))
      .catch((e) => on && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => on && setLoading(false));
    return () => { on = false; };
  }, []);
  return { listings, loading, error };
}
