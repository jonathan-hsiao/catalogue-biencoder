"use client";

import { useState } from "react";
import { useStore } from "@/lib/store";
import type { SplitMode } from "@/lib/store";
import SearchBar from "./SearchBar";

export default function ControlPanel() {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const data = useStore((s) => s.data);
  const splitMode = useStore((s) => s.splitMode);
  const setSplitMode = useStore((s) => s.setSplitMode);
  const ambiguousOnly = useStore((s) => s.ambiguousOnly);
  const setAmbiguousOnly = useStore((s) => s.setAmbiguousOnly);
  const interactiveStars = useStore((s) => s.interactiveStars);
  const interactivePoints = useStore((s) => s.interactivePoints);
  const interactiveDimmed = useStore((s) => s.interactiveDimmed);
  const setInteractiveStars = useStore((s) => s.setInteractiveStars);
  const setInteractivePoints = useStore((s) => s.setInteractivePoints);
  const setInteractiveDimmed = useStore((s) => s.setInteractiveDimmed);
  const resetView = useStore((s) => s.resetView);

  if (!data) return null;

  return (
    <div data-ui="control-panel" className="absolute left-4 top-16 z-40 flex flex-col gap-3 rounded-lg bg-space-900/90 p-3 text-sm">
      <SearchBar />
      <button
        onClick={resetView}
        className="rounded bg-space-800 px-3 py-1 text-slate-400 hover:bg-space-700"
      >
        Reset view
      </button>
      <button
        type="button"
        onClick={() => setAdvancedOpen((o) => !o)}
        className="flex items-center gap-1 text-slate-400 hover:text-slate-300"
      >
        <span className="text-slate-500 w-4">{advancedOpen ? "-" : "+"}</span>
        Advanced
      </button>
      {advancedOpen && (
        <div className="flex flex-col gap-3 pl-4 border-l border-space-700">
          <div className="flex flex-col gap-2">
            <div className="text-slate-300">Set Interactive</div>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setInteractiveStars(!interactiveStars)}
                className={`rounded px-3 py-1 ${
                  interactiveStars
                    ? "bg-blue-600 text-white"
                    : "bg-space-800 text-slate-400 hover:bg-space-700"
                }`}
              >
                Stars
              </button>
              <button
                onClick={() => setInteractivePoints(!interactivePoints)}
                className={`rounded px-3 py-1 ${
                  interactivePoints
                    ? "bg-blue-600 text-white"
                    : "bg-space-800 text-slate-400 hover:bg-space-700"
                }`}
              >
                Points
              </button>
              <button
                onClick={() => setInteractiveDimmed(!interactiveDimmed)}
                className={`rounded px-3 py-1 ${
                  interactiveDimmed
                    ? "bg-blue-600 text-white"
                    : "bg-space-800 text-slate-400 hover:bg-space-700"
                }`}
              >
                Dimmed
              </button>
            </div>
          </div>
          {/* Filter Dataset Split - commented out, can be restored by uncommenting
          <div className="text-slate-300">Filter Dataset Split</div>
          <div className="flex gap-2">
            {(
              [
                [0, "All"],
                [1, "Train"],
                [2, "Test"],
              ] as [SplitMode, string][]
            ).map(([mode, label]) => (
              <button
                key={mode}
                onClick={() => setSplitMode(mode)}
                className={`rounded px-3 py-1 ${
                  splitMode === mode
                    ? "bg-blue-600 text-white"
                    : "bg-space-800 text-slate-400 hover:bg-space-700"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          */}
          {/* Show misclassified only - commented out, can be restored by uncommenting
          <label className="flex items-center gap-2 text-slate-400">
            <input
              type="checkbox"
              checked={ambiguousOnly}
              onChange={(e) => setAmbiguousOnly(e.target.checked)}
              className="rounded"
            />
            Show misclassified only
          </label>
          */}
        </div>
      )}
    </div>
  );
}
