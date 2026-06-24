"use client";
import { useCallback, useEffect, useState } from "react";
import { defaultFilters, filtersToParams, paramsToFilters } from "@/lib/filters";
import type { FilterState } from "@/lib/types";

export function useFilters() {
  const [filters, setFilters] = useState<FilterState>(defaultFilters);
  useEffect(() => { setFilters(paramsToFilters(new URLSearchParams(window.location.search))); }, []);
  useEffect(() => {
    const qs = filtersToParams(filters).toString();
    const next = qs ? `?${qs}` : window.location.pathname;
    window.history.replaceState(null, "", next);
  }, [filters]);
  const reset = useCallback(() => setFilters(defaultFilters()), []);
  return { filters, setFilters, reset };
}
