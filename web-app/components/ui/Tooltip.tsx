"use client";

import { useState, useEffect } from "react";
import { useStore } from "@/lib/store";
import { getAssetUrl } from "@/lib/utils";

export default function Tooltip() {
  const hoveredId = useStore((s) => s.hoveredId);
  const hoveredStarIndex = useStore((s) => s.hoveredStarIndex);
  const meta = useStore((s) => s.meta);
  const data = useStore((s) => s.data);

  const [pos, setPos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const onMove = (e: MouseEvent) => setPos({ x: e.clientX, y: e.clientY });
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, []);

  const show = hoveredId != null || hoveredStarIndex != null;
  if (!show) return null;

  const product = hoveredId != null && meta && hoveredId < meta.length ? meta[hoveredId] : null;
  const star =
    hoveredStarIndex != null &&
    data &&
    hoveredStarIndex < data.numStars
      ? {
          coarseLabel: data.coarseCategories[data.starIds[hoveredStarIndex]] ?? "",
          count_all: data.starCountAll[hoveredStarIndex],
          count_train: data.starCountTrain[hoveredStarIndex],
          count_test: data.starCountTest[hoveredStarIndex],
        }
      : null;

  return (
    <div
      className="pointer-events-none fixed z-50 max-w-xs rounded-lg border border-space-700 bg-space-900/95 p-3 shadow-xl"
      style={{ left: pos.x + 12, top: pos.y + 12 }}
    >
      {product && (
        <div className="space-y-1 text-xs">
          {product.thumb && (
            <img
              src={getAssetUrl(product.thumb)}
              alt=""
              className="h-20 w-20 rounded object-cover"
            />
          )}
          <div className="font-medium text-slate-200">{product.title || "—"}</div>
          <div className="text-slate-400">{product.brand || "—"}</div>
          {product.desc && (
            <div className="line-clamp-2 text-slate-500">{product.desc}</div>
          )}
          <div className="text-slate-500">
            GT: {product.gt_category || "—"}
          </div>
        </div>
      )}
      {star && !product && (
        <div className="text-xs">
          <div className="font-medium text-slate-200">{star.coarseLabel || "—"}</div>
          <div className="text-slate-400">
            all: {star.count_all} | train: {star.count_train} | test: {star.count_test}
          </div>
        </div>
      )}
    </div>
  );
}
