"use client";

import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { useStore } from "@/lib/store";

const DEBOUNCE_MS = 250;
const MAX_SUGGESTIONS = 10;

interface Suggestion {
  type: "brand" | "product" | "star";
  label: string;
  value: string;
  id?: number;
}

export default function SearchBar() {
  const searchReady = useStore((s) => s.searchReady);
  const meta = useStore((s) => s.meta);
  const data = useStore((s) => s.data);
  const setSearchQuery = useStore((s) => s.setSearchQuery);
  const setSearchInput = useStore((s) => s.setSearchInput);
  const setVisibleIds = useStore((s) => s.setVisibleIds);
  const setSelectedBrand = useStore((s) => s.setSelectedBrand);
  const setSelectedId = useStore((s) => s.setSelectedId);
  const setSelectedStarIndex = useStore((s) => s.setSelectedStarIndex);

  const input = useStore((s) => s.searchInput);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Parse query for prefix
  const parseQuery = useCallback((query: string): { prefix: string | null; term: string } => {
    const trimmed = query.trim();
    const brandMatch = trimmed.match(/^brand:\s*(.+)/i);
    if (brandMatch) return { prefix: "brand", term: brandMatch[1].trim() };
    const titleMatch = trimmed.match(/^title:\s*(.+)/i);
    if (titleMatch) return { prefix: "title", term: titleMatch[1].trim() };
    const categoryMatch = trimmed.match(/^category:\s*(.+)/i);
    if (categoryMatch) return { prefix: "category", term: categoryMatch[1].trim() };
    return { prefix: null, term: trimmed };
  }, []);

  // Generate suggestions
  const suggestions = useMemo((): Suggestion[] => {
    if (!meta || !data || !input.trim()) return [];
    const { prefix, term } = parseQuery(input);
    const termLower = term.toLowerCase();
    const results: Suggestion[] = [];

    if (prefix === "brand" || prefix === null) {
      // Brand suggestions: count products per brand, sort by count descending
      const brandCounts = new Map<string, number>();
      for (const m of meta) {
        if (!m.brand || !m.brand.toLowerCase().includes(termLower)) continue;
        brandCounts.set(m.brand, (brandCounts.get(m.brand) ?? 0) + 1);
      }
      const sortedBrands = Array.from(brandCounts.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, MAX_SUGGESTIONS)
        .map(([brand]) => brand);
      for (const brand of sortedBrands) {
        results.push({ type: "brand", label: brand, value: brand });
      }
    }

    if (prefix === "title" || prefix === null) {
      // Product suggestions
      for (const m of meta) {
        if (m.title && m.title.toLowerCase().includes(termLower)) {
          results.push({
            type: "product",
            label: m.title,
            value: m.title,
            id: m.id,
          });
          if (results.length >= MAX_SUGGESTIONS) break;
        }
      }
    }

    if (prefix === "category" || prefix === null) {
      // Category suggestions (coarse categories)
      const categories = new Set<string>();
      for (const m of meta) {
        if (m.gt_coarse && m.gt_coarse.toLowerCase().includes(termLower)) {
          categories.add(m.gt_coarse);
        }
      }
      for (const cat of Array.from(categories).slice(0, MAX_SUGGESTIONS)) {
        results.push({ type: "star", label: cat, value: cat });
      }
    }

    return results.slice(0, MAX_SUGGESTIONS);
  }, [input, meta, data, parseQuery]);

  // Only update store query; do not filter the 3D view while typing.
  // The visual changes only when the user selects a suggestion (handleSelectSuggestion).
  const runSearch = useCallback(
    (query: string) => {
      setSearchQuery(query);
      const { term } = parseQuery(query);
      const q = term.toLowerCase();
      if (!q) {
        setVisibleIds(null);
        return;
      }
      // Do not set visibleIds here — keeps 3D view unchanged until a suggestion is selected.
    },
    [setSearchQuery, setVisibleIds, parseQuery]
  );

  useEffect(() => {
    const t = setTimeout(() => runSearch(input), DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [input, runSearch]);

  const handleSelectSuggestion = useCallback((suggestion: Suggestion) => {
    if (suggestion.type === "brand") {
      setSelectedBrand(suggestion.value);
      setSelectedId(null);
      setSelectedStarIndex(null);
      setSearchInput(`brand: ${suggestion.value}`);
    } else if (suggestion.type === "product" && suggestion.id != null) {
      setSelectedId(suggestion.id);
      setSelectedBrand(null);
      setSelectedStarIndex(null);
      setSearchInput(`title: ${suggestion.value}`);
    } else if (suggestion.type === "star") {
      // Find star index for this category
      if (data) {
        for (let i = 0; i < data.starIds.length; i++) {
          const coarseLabel = data.coarseCategories[data.starIds[i]] ?? "";
          if (coarseLabel === suggestion.value) {
            setSelectedStarIndex(i);
            setSelectedBrand(null);
            setSelectedId(null);
            setSearchInput(`category: ${suggestion.value}`);
            break;
          }
        }
      }
    }
    setShowSuggestions(false);
  }, [data, setSelectedBrand, setSelectedId, setSelectedStarIndex, setSearchInput]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (!showSuggestions || suggestions.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedSuggestionIndex((i) => (i + 1) % suggestions.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedSuggestionIndex((i) => (i - 1 + suggestions.length) % suggestions.length);
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (suggestions[selectedSuggestionIndex]) {
        handleSelectSuggestion(suggestions[selectedSuggestionIndex]);
      }
    } else if (e.key === "Escape") {
      setShowSuggestions(false);
    }
  }, [showSuggestions, suggestions, selectedSuggestionIndex, handleSelectSuggestion]);

  return (
    <div className="relative flex flex-col gap-1">
      <input
        ref={inputRef}
        type="text"
        placeholder={searchReady ? "Search: brand: X, title: X, category: X" : "Search indexing…"}
        value={input}
        onChange={(e) => {
          setSearchInput(e.target.value);
          setShowSuggestions(true);
          setSelectedSuggestionIndex(0);
        }}
        onFocus={() => setShowSuggestions(true)}
        onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
        onKeyDown={handleKeyDown}
        disabled={!searchReady}
        className="min-w-[15rem] rounded border border-space-700 bg-space-800 px-2 py-1.5 text-slate-200 placeholder:text-slate-500 disabled:opacity-50"
      />
      {showSuggestions && suggestions.length > 0 && (
        <div className="absolute top-full z-50 mt-1 max-h-64 w-full overflow-y-auto rounded border border-space-700 bg-space-900 shadow-lg">
          {suggestions.map((s, i) => (
            <button
              key={`${s.type}-${s.value}-${i}`}
              type="button"
              onClick={() => handleSelectSuggestion(s)}
              className={`w-full px-3 py-2 text-left text-sm hover:bg-space-800 ${
                i === selectedSuggestionIndex ? "bg-space-800" : ""
              }`}
            >
              <div className="text-slate-300">{s.label}</div>
              <div className="text-xs text-slate-500">{s.type}</div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
