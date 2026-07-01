"use client";
import { useCallback, useEffect, useState } from "react";
import { defaultFilters, filtersToParams, paramsToFilters } from "@/lib/filters";
import type { FilterState } from "@/lib/types";

export function useFilters() {
  const [filters, setFilters] = useState<FilterState>(defaultFilters);
  // Hydrate from the URL after mount. Intentional setState-in-effect: the server and
  // first client render must both use defaults to avoid a hydration mismatch, then we
  // sync to the URL once on the client. Runs once (empty deps).
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { setFilters(paramsToFilters(new URLSearchParams(window.location.search))); }, []);
  useEffect(() => {
    const qs = filtersToParams(filters).toString();
    const next = qs ? `?${qs}` : window.location.pathname;
    window.history.replaceState(null, "", next);
  }, [filters]);
  const reset = useCallback(() => setFilters(defaultFilters()), []);
  return { filters, setFilters, reset };
}
