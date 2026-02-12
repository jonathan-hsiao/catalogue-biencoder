"use client";

import { forwardRef, useMemo } from "react";
import * as THREE from "three";
import { useStore } from "@/lib/store";
import { topLevelFromCoarse, categoryPalette } from "@/lib/utils";
import { DIM_BRIGHTNESS_MULT, DIM_ALPHA_MULT } from "@/lib/constants";

/** Creates a round point texture so points render as circles, not squares. Fully transparent outside the gradient to avoid black box. */
function createRoundPointTexture(): THREE.CanvasTexture {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, size, size);
  const gradient = ctx.createRadialGradient(
    size / 2, size / 2, 0,
    size / 2, size / 2, size / 2
  );
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.35, "rgba(255,255,255,0.95)");
  gradient.addColorStop(0.55, "rgba(255,255,255,0.4)");
  gradient.addColorStop(0.8, "rgba(255,255,255,0.04)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

const roundPointTexture = createRoundPointTexture();

/** Shared PointsMaterial config - parameterized so we can tweak opacity for dimmed layer. */
function makeStarMaterial(opacity: number): THREE.PointsMaterial {
  return new THREE.PointsMaterial({
    size: 0.12,
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    opacity,
    map: roundPointTexture,
    alphaTest: 0.02,
    depthWrite: false,
    depthTest: true,
    blending: THREE.AdditiveBlending,
  });
}

const Stars = forwardRef<THREE.Points>(function Stars(_, ref) {
  const data = useStore((s) => s.data);
  const selectedStarIndex = useStore((s) => s.selectedStarIndex);
  const selectedId = useStore((s) => s.selectedId);
  const selectedBrand = useStore((s) => s.selectedBrand);
  const meta = useStore((s) => s.meta);

  const hasSelection = selectedId != null || selectedStarIndex != null || selectedBrand != null;

  const { geometry, geometryDimmed, material, materialDimmed } = useMemo(() => {
    if (!data) return { geometry: null, geometryDimmed: null, material: null, materialDimmed: null };
    const { starIds, starPos, coarseCategories, topLevels } = data;
    const numStars = starIds.length;
    const numTop = topLevels.length;
    const palette = categoryPalette(numTop + 1, 0.7, 0.9);

    const topLevelToIndex = new Map<string, number>();
    topLevels.forEach((t, i) => topLevelToIndex.set(t, i));

    const colors = new Float32Array(numStars * 3);
    for (let i = 0; i < numStars; i++) {
      const coarseLabel = coarseCategories[starIds[i]] ?? "";
      const top = topLevelFromCoarse(coarseLabel);
      const tid = topLevelToIndex.get(top) ?? 0;
      const j = (tid + 1) * 3;
      colors[i * 3] = palette[j];
      colors[i * 3 + 1] = palette[j + 1];
      colors[i * 3 + 2] = palette[j + 2];
    }

    const highlightedStarIndices = new Set<number>();
    if (hasSelection) {
      if (selectedStarIndex != null) {
        highlightedStarIndices.add(selectedStarIndex);
      } else if (selectedId != null) {
        const coarseId = data.gtCoarseId[selectedId];
        for (let i = 0; i < numStars; i++) {
          if (data.starIds[i] === coarseId) {
            highlightedStarIndices.add(i);
            break;
          }
        }
      } else if (selectedBrand != null && meta) {
        const brandProductIds = new Set<number>();
        for (let i = 0; i < meta.length && i < data.N; i++) {
          if (meta[i].brand && meta[i].brand.toLowerCase() === selectedBrand.toLowerCase()) {
            brandProductIds.add(i);
          }
        }
        for (let i = 0; i < numStars; i++) {
          const starId = data.starIds[i];
          const members = data.coarseIdToMembers.get(starId);
          if (members) {
            for (const gid of members) {
              if (brandProductIds.has(gid)) {
                highlightedStarIndices.add(i);
                break;
              }
            }
          }
        }
      }
    }

    const matFull = makeStarMaterial(1.0);
    const matDimmed = makeStarMaterial(DIM_ALPHA_MULT);

    if (!hasSelection || highlightedStarIndices.size === 0) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(starPos, 3));
      geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
      return { geometry: geo, geometryDimmed: null, material: matFull, materialDimmed: null };
    }

    const posHighlight: number[] = [];
    const colHighlight: number[] = [];
    const posDimmed: number[] = [];
    const colDimmed: number[] = [];
    for (let i = 0; i < numStars; i++) {
      const i3 = i * 3;
      if (highlightedStarIndices.has(i)) {
        posHighlight.push(starPos[i3], starPos[i3 + 1], starPos[i3 + 2]);
        colHighlight.push(colors[i3], colors[i3 + 1], colors[i3 + 2]);
      } else {
        posDimmed.push(starPos[i3], starPos[i3 + 1], starPos[i3 + 2]);
        colDimmed.push(
          colors[i3] * DIM_BRIGHTNESS_MULT,
          colors[i3 + 1] * DIM_BRIGHTNESS_MULT,
          colors[i3 + 2] * DIM_BRIGHTNESS_MULT
        );
      }
    }

    const geoHighlight = new THREE.BufferGeometry();
    geoHighlight.setAttribute("position", new THREE.Float32BufferAttribute(posHighlight, 3));
    geoHighlight.setAttribute("color", new THREE.Float32BufferAttribute(colHighlight, 3));

    const geoDim = new THREE.BufferGeometry();
    geoDim.setAttribute("position", new THREE.Float32BufferAttribute(posDimmed, 3));
    geoDim.setAttribute("color", new THREE.Float32BufferAttribute(colDimmed, 3));

    return {
      geometry: geoHighlight,
      geometryDimmed: geoDim,
      material: matFull,
      materialDimmed: matDimmed,
    };
  }, [data, hasSelection, selectedStarIndex, selectedId, selectedBrand, meta]);

  const setRef = (el: THREE.Points | null) => {
    if (typeof ref === "function") ref(el);
    else if (ref) (ref as React.MutableRefObject<THREE.Points | null>).current = el;
  };

  if (!data || !geometry || !material) return null;

  if (geometryDimmed != null && materialDimmed != null) {
    return (
      <>
        <points ref={setRef} geometry={geometry} material={material} frustumCulled={false} renderOrder={2} />
        <points geometry={geometryDimmed} material={materialDimmed} frustumCulled={false} renderOrder={2} />
      </>
    );
  }

  return (
    <points ref={setRef} geometry={geometry} material={material} frustumCulled={false} renderOrder={2} />
  );
});

export default Stars;
