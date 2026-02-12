"use client";

import { useStore } from "@/lib/store";
import { getAssetUrl } from "@/lib/utils";
import { useMemo, useState, useEffect } from "react";

export default function Sidebar() {
  const data = useStore((s) => s.data);
  const meta = useStore((s) => s.meta);
  const selectedId = useStore((s) => s.selectedId);
  const selectedStarIndex = useStore((s) => s.selectedStarIndex);
  const selectedBrand = useStore((s) => s.selectedBrand);
  const setFlyToPoint = useStore((s) => s.setFlyToPoint);
  const setSelectedId = useStore((s) => s.setSelectedId);
  const setSelectedStarIndex = useStore((s) => s.setSelectedStarIndex);
  const setSelectedBrand = useStore((s) => s.setSelectedBrand);
  const splitMode = useStore((s) => s.splitMode);
  const ambiguousOnly = useStore((s) => s.ambiguousOnly);
  const visibleIds = useStore((s) => s.visibleIds);

  const open = selectedId != null || selectedStarIndex != null || selectedBrand != null;

  const [descExpanded, setDescExpanded] = useState(false);

  useEffect(() => {
    setDescExpanded(false);
  }, [selectedId]);

  const selectedProduct = useMemo(() => {
    if (selectedId == null || !meta) return null;
    return meta[selectedId] ?? null;
  }, [meta, selectedId]);

  const neighborIds = useMemo(() => {
    if (!data || selectedId == null) return [];
    const ids: number[] = [];
    let count = 0;
    for (let k = 0; k < data.K_GLOBAL && count < data.K_UI; k++) {
      const j: number = data.neighFused[selectedId * data.K_GLOBAL + k];
      if (j === selectedId) continue;
      if (splitMode !== 0 && data.splitId[j] !== splitMode - 1) continue;
      if (ambiguousOnly && data.ambiguous[j] === 0) continue;
      if (visibleIds && !visibleIds.has(j)) continue;
      ids.push(j);
      count++;
    }
    return ids;
  }, [data, selectedId, splitMode, ambiguousOnly, visibleIds]);

  const starData = useMemo(() => {
    if (!data || selectedStarIndex == null) return null;
    const coarseId = data.starIds[selectedStarIndex];
    const reps = data.coarseIdToRepresentatives.get(coarseId);
    const label = data.coarseCategories[coarseId] ?? "";
    const count_all = data.starCountAll[selectedStarIndex];
    const count_train = data.starCountTrain[selectedStarIndex];
    const count_test = data.starCountTest[selectedStarIndex];
    return {
      coarseId,
      label,
      count_all,
      count_train,
      count_test,
      representativeIds: reps ?? [],
    };
  }, [data, selectedStarIndex]);

  /** Brand products grouped by coarse category (star level), in star order. */
  const brandData = useMemo(() => {
    if (!data || !meta || !selectedBrand) return null;
    const brandLower = selectedBrand.toLowerCase();
    const idsByCoarse = new Map<number, number[]>();
    for (let i = 0; i < meta.length && i < data.N; i++) {
      if (!meta[i].brand || meta[i].brand.toLowerCase() !== brandLower) continue;
      const coarseId = data.gtCoarseId[i];
      if (!idsByCoarse.has(coarseId)) idsByCoarse.set(coarseId, []);
      idsByCoarse.get(coarseId)!.push(i);
    }
    const sections: { label: string; coarseId: number; ids: number[] }[] = [];
    for (let si = 0; si < data.starIds.length; si++) {
      const coarseId = data.starIds[si];
      const ids = idsByCoarse.get(coarseId);
      if (!ids || ids.length === 0) continue;
      const label = data.coarseCategories[coarseId] ?? "";
      sections.push({ label, coarseId, ids });
    }
    const productCount = sections.reduce((sum, s) => sum + s.ids.length, 0);
    return { brandName: selectedBrand, sections, productCount };
  }, [data, meta, selectedBrand]);

  const flyToProduct = (globalId: number) => {
    if (!data) return;
    const i = globalId * 3;
    setFlyToPoint([
      data.posFused[i],
      data.posFused[i + 1],
      data.posFused[i + 2],
    ]);
  };

  const flyToStar = () => {
    if (!data || selectedStarIndex == null) return;
    const i = selectedStarIndex * 3;
    setFlyToPoint([
      data.starPos[i],
      data.starPos[i + 1],
      data.starPos[i + 2],
    ]);
  };

  return (
    <div
      data-ui="sidebar"
      className={`absolute right-0 top-0 z-40 h-full w-80 overflow-y-auto rounded-l-lg bg-space-900/90 px-5 pt-7 pb-4 transition-transform duration-200 ease-out ${
        open ? "translate-x-0" : "translate-x-full"
      } ${!open ? "pointer-events-none" : ""}`}
    >
      <div className="space-y-4">
        {selectedProduct && (
          <>
            <div className="text-sm font-medium text-slate-200">Selected Product</div>
            <div className="rounded-lg border border-space-700 p-3">
              {selectedProduct.thumb && (
                <img
                  src={getAssetUrl(selectedProduct.thumb)}
                  alt=""
                  className="mb-2 h-24 w-24 rounded object-cover"
                />
              )}
              <div className="font-medium">{selectedProduct.title || "—"}</div>
              <div className="text-slate-400 text-sm">{selectedProduct.brand}</div>
              <div className="text-slate-500 text-xs mt-1">{selectedProduct.gt_category}</div>
              {selectedProduct.desc && (
                <div className="mt-2">
                  <div
                    className={`text-slate-500 text-xs ${descExpanded ? "" : "line-clamp-4"}`}
                  >
                    {selectedProduct.desc}
                  </div>
                  <button
                    type="button"
                    onClick={() => setDescExpanded((e) => !e)}
                    className="mt-1 text-xs text-slate-500 hover:text-slate-400"
                  >
                    {descExpanded ? "Hide" : "Expand"}
                  </button>
                </div>
              )}
            </div>
            <button
              onClick={() => selectedId != null && flyToProduct(selectedId)}
              className="rounded bg-space-800 px-3 py-1 text-slate-300 hover:bg-space-700"
            >
              Fly to point
            </button>
            <div className="text-sm font-medium text-slate-200">Neighbors (embedding space)</div>
            <div className="space-y-2">
              {neighborIds.map((id, rank) => {
                const m = meta?.[id];
                return (
                  <div
                    key={id}
                    className="flex gap-2 rounded border border-space-700 p-2"
                  >
                    {m?.thumb && (
                      <img
                        src={getAssetUrl(m.thumb)}
                        alt=""
                        className="h-12 w-12 rounded object-cover"
                      />
                    )}
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-xs font-medium">{m?.title ?? id}</div>
                      <div className="text-slate-500 text-xs">{m?.brand}</div>
                      <div className="text-slate-600 text-xs">#{rank + 1}</div>
                    </div>
                    <button
                      onClick={() => flyToProduct(id)}
                      className="self-center rounded px-2 py-0.5 text-xs text-slate-400 hover:bg-space-800"
                    >
                      Fly
                    </button>
                  </div>
                );
              })}
            </div>
          </>
        )}

        {starData && !selectedProduct && !selectedBrand && (
          <>
            <div className="text-sm font-medium text-slate-200">Selected Star (Category)</div>
            <div className="rounded-lg border border-space-700 p-3">
              <div className="font-medium">{starData.label || "—"}</div>
              <div className="text-slate-400 text-xs">
                all: {starData.count_all} | train: {starData.count_train} | test: {starData.count_test}
              </div>
            </div>
            <button
              onClick={flyToStar}
              className="rounded bg-space-800 px-3 py-1 text-slate-300 hover:bg-space-700"
            >
              Fly to star
            </button>
            <div className="text-sm font-medium text-slate-200">Representative products</div>
            {starData.representativeIds.length === 0 ? (
              <div className="text-slate-500 text-sm">No products in this category</div>
            ) : (
              <div className="space-y-2">
                {starData.representativeIds.map((id) => {
                  const m = meta?.[id];
                  return (
                    <div
                      key={id}
                      className="flex gap-2 rounded border border-space-700 p-2"
                    >
                      {m?.thumb && (
                        <img
                          src={getAssetUrl(m.thumb)}
                          alt=""
                          className="h-12 w-12 rounded object-cover"
                        />
                      )}
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-xs font-medium">{m?.title ?? id}</div>
                        <div className="text-slate-500 text-xs">{m?.brand}</div>
                      </div>
                      <button
                        onClick={() => flyToProduct(id)}
                        className="self-center rounded px-2 py-0.5 text-xs text-slate-400 hover:bg-space-800"
                      >
                        Fly
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}

        {brandData && !selectedProduct && (
          <>
            <div className="text-sm font-medium text-slate-200">Selected Brand</div>
            <div className="rounded-lg border border-space-700 p-3">
              <div className="font-medium">{brandData.brandName}</div>
              <div className="text-slate-400 text-xs mt-1">
                {brandData.productCount} product{brandData.productCount !== 1 ? "s" : ""}
              </div>
            </div>
            <div className="space-y-4">
              {brandData.sections.map(({ label, ids }) => (
                <div key={label || "uncategorized"}>
                  <div className="text-xs font-medium text-slate-400 mb-2">{label || "Other"}</div>
                  <div className="space-y-2">
                    {ids.map((id) => {
                      const m = meta?.[id];
                      return (
                        <div
                          key={id}
                          className="flex gap-2 rounded border border-space-700 p-2"
                        >
                          {m?.thumb && (
                            <img
                              src={getAssetUrl(m.thumb)}
                              alt=""
                              className="h-12 w-12 rounded object-cover"
                            />
                          )}
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-xs font-medium">{m?.title ?? id}</div>
                            <div className="text-slate-500 text-xs">{m?.brand}</div>
                          </div>
                          <button
                            onClick={() => flyToProduct(id)}
                            className="self-center rounded px-2 py-0.5 text-xs text-slate-400 hover:bg-space-800"
                          >
                            Fly
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
