"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { useStore } from "@/lib/store";
import { loadManifest } from "@/lib/data/ManifestLoader";
import { loadAllBinaries } from "@/lib/data/BinaryLoader";
import { ShardManager } from "@/lib/data/ShardManager";

const Scene3D = dynamic(() => import("@/components/scene/Scene3D"), {
  ssr: false,
});

export default function Home() {
  const [mounted, setMounted] = useState(false);
  const setData = useStore((s) => s.setData);
  const setMeta = useStore((s) => s.setMeta);
  const setSearchReady = useStore((s) => s.setSearchReady);
  const setShardManager = useStore((s) => s.setShardManager);
  const setLoadStatus = useStore((s) => s.setLoadStatus);
  const setMetaLoadProgress = useStore((s) => s.setMetaLoadProgress);
  const loadStatus = useStore((s) => s.loadStatus);
  const metaLoadProgress = useStore((s) => s.metaLoadProgress);
  const data = useStore((s) => s.data);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;

    let cancelled = false;
    const run = async () => {
      try {
        setLoadStatus("Loading manifest…");
        const manifest = await loadManifest();
        if (cancelled) return;

        setLoadStatus("Loading embeddings…");
        const loaded = await loadAllBinaries(manifest);
        if (cancelled) return;

        setData(loaded);
        setLoadStatus("Rendering…");

        const shardManager = new ShardManager(manifest);
        setShardManager(shardManager);
        setLoadStatus("Loading neighbors…");
        setMetaLoadProgress({ loaded: 0, total: 1 });

        shardManager
          .preloadAllMeta((loaded, total) => {
            if (!cancelled) setMetaLoadProgress({ loaded, total });
          })
          .then((meta) => {
            if (cancelled) return;
            setMeta(meta);
            setSearchReady(true);
            setLoadStatus("");
            setMetaLoadProgress(null);
          })
          .catch((e) => {
            if (!cancelled) {
              setLoadStatus("Search index failed: " + (e as Error).message);
            }
          });
      } catch (e) {
        if (!cancelled) {
          setLoadStatus("Failed: " + (e as Error).message);
        }
      }
    };

    run();
    return () => {
      cancelled = true;
    };
  }, [mounted, setData, setMeta, setSearchReady, setShardManager, setLoadStatus, setMetaLoadProgress]);

  return (
    <main className="relative h-screen w-screen">
      {/* Status strip */}
      {(loadStatus || metaLoadProgress) && (
        <div className="absolute left-0 right-0 top-0 z-50 flex items-center justify-center gap-4 bg-space-900/90 px-4 py-2 text-sm text-slate-300">
          {loadStatus && <span>{loadStatus}</span>}
          {metaLoadProgress && (
            <span>
              Search indexing… ({metaLoadProgress.loaded}/{metaLoadProgress.total})
            </span>
          )}
        </div>
      )}

      {data ? (
        <Scene3D />
      ) : (
        <div className="flex h-full items-center justify-center">
          <p className="text-slate-400">
            {loadStatus || "Initializing…"}
          </p>
        </div>
      )}
    </main>
  );
}
