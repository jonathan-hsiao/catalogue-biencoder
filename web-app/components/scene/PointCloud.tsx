"use client";

import { forwardRef, useRef, useMemo, useEffect } from "react";
import * as THREE from "three";
import { useStore } from "@/lib/store";
import {
  POINT_STATE_NORMAL,
  POINT_STATE_NEIGHBOR,
  POINT_STATE_SELECTED,
  HIGHLIGHT_BRIGHTNESS_MULT,
  HIGHLIGHT_ALPHA_MULT,
  HIGHLIGHT_GLOW_STRENGTH,
  DIM_BRIGHTNESS_MULT,
  DIM_ALPHA_MULT,
  DIM_GLOW_STRENGTH,
  SELECTED_COLOR_BASE,
  SELECTED_COLOR_HIGHLIGHT,
  NEIGHBOR_COLOR_BASE,
  NEIGHBOR_COLOR_HIGHLIGHT,
  SELECTED_POINT_SIZE_MULT,
  SELECTED_POINT_SIZE_MAX,
} from "@/lib/constants";
import { categoryPalette } from "@/lib/utils";

const vertexShader = `
  attribute vec3 color;
  attribute float aVisible;
  attribute float aState;
  attribute float aSplitId;
  uniform float uSplitMode;
  uniform float uSelectedPointSizeMult;
  uniform float uSelectedPointSizeMax;
  varying vec3 vColor;
  varying float vState;
  varying float vVisible;
  varying float vDepth;
  void main() {
    vColor = color;
    vState = aState;
    float showBySplit = (uSplitMode < 0.5) ? 1.0 : (abs(aSplitId - (uSplitMode - 1.0)) < 0.5 ? 1.0 : 0.0);
    vVisible = aVisible * showBySplit;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vDepth = -mvPosition.z;
    float size = (vDepth > 0.01) ? clamp(12.0 / vDepth, 2.0, 48.0) : 2.0;
    if (aState > 1.5) {
      size = min(size * uSelectedPointSizeMult, uSelectedPointSizeMax);
    }
    gl_PointSize = size;
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = `
  uniform float uHighlightBrightness;
  uniform float uHighlightAlpha;
  uniform float uHighlightGlow;
  uniform float uDimBrightness;
  uniform float uDimAlpha;
  uniform float uDimGlow;
  uniform vec3 uSelectedColorBase;
  uniform vec3 uSelectedColorHighlight;
  uniform vec3 uNeighborColorBase;
  uniform vec3 uNeighborColorHighlight;
  varying vec3 vColor;
  varying float vState;
  varying float vVisible;
  varying float vDepth;
  void main() {
    if (vVisible < 0.5) discard;
    float d = length(gl_PointCoord - 0.5) * 2.0;
    if (d > 1.0) discard;
    float alpha = 1.0 - smoothstep(0.28, 0.62, d);
    float glow = 1.0 - smoothstep(0.0, 0.45, d);
    float core = 1.0 - smoothstep(0.0, 0.12, d);
    float depthFade = 1.0 - 0.4 * clamp(vDepth / 5.0, 0.0, 1.0);
    
    bool isHighlighted = vState > 0.5;
    bool isDimmed = vState < 0.5 && (uDimBrightness < 1.0 || uDimAlpha < 1.0);
    
    if (vState > 1.5) {
      vec3 c = mix(uSelectedColorBase, uSelectedColorHighlight, core);
      c *= uHighlightBrightness;
      gl_FragColor = vec4(c, alpha * uHighlightAlpha);
    } else if (vState > 0.5) {
      vec3 c = mix(uNeighborColorBase, uNeighborColorHighlight, core);
      c *= uHighlightBrightness;
      gl_FragColor = vec4(c, alpha * uHighlightAlpha);
    } else {
      // Normal or dimmed
      vec3 base = vColor * depthFade;
      float glowStrength = isDimmed ? uDimGlow : 0.15;
      float brightnessMult = isDimmed ? uDimBrightness : 1.0;
      float alphaMult = isDimmed ? uDimAlpha : 0.78;
      
      vec3 withCore = mix(base, base + vec3(0.12, 0.12, 0.15), core * 0.5);
      vec3 final = mix(withCore, withCore + vec3(0.04), glow * glowStrength);
      final *= brightnessMult;
      gl_FragColor = vec4(final, alpha * alphaMult);
    }
  }
`;

const PointCloud = forwardRef<THREE.Points>(function PointCloud(_, ref) {
  const pointsRef = useRef<THREE.Points>(null);
  const data = useStore((s) => s.data);
  const splitMode = useStore((s) => s.splitMode);
  const visibleIds = useStore((s) => s.visibleIds);
  const ambiguousOnly = useStore((s) => s.ambiguousOnly);
  const selectedId = useStore((s) => s.selectedId);
  const selectedStarIndex = useStore((s) => s.selectedStarIndex);
  const selectedBrand = useStore((s) => s.selectedBrand);
  const setSelectedStarIndex = useStore((s) => s.setSelectedStarIndex);
  const shardManager = useStore((s) => s.shardManager);
  const meta = useStore((s) => s.meta);

  const { geometry, material, palette } = useMemo(() => {
    if (!data) return { geometry: null, material: null, palette: null };
    const N = data.N;
    const pos = data.posFused;
    const splitId = data.splitId;
    const gtTopId = data.gtTopId;
    const numTop = data.topLevels.length;

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    const colors = new Float32Array(N * 3);
    const pal = categoryPalette(numTop + 1);
    for (let i = 0; i < N; i++) {
      const tid = Math.max(0, gtTopId[i] + 1);
      const j = Math.min(tid, numTop) * 3;
      colors[i * 3] = pal[j];
      colors[i * 3 + 1] = pal[j + 1];
      colors[i * 3 + 2] = pal[j + 2];
    }
    geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    geo.setAttribute(
      "aSplitId",
      new THREE.Float32BufferAttribute(
        new Float32Array(splitId),
        1
      )
    );
    const aVisible = new Float32Array(N);
    for (let i = 0; i < N; i++) aVisible[i] = 1;
    geo.setAttribute("aVisible", new THREE.Float32BufferAttribute(aVisible, 1));
    const aState = new Float32Array(N);
    geo.setAttribute("aState", new THREE.Float32BufferAttribute(aState, 1));

    const mat = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        uSplitMode: { value: splitMode },
        uSelectedPointSizeMult: { value: SELECTED_POINT_SIZE_MULT },
        uSelectedPointSizeMax: { value: SELECTED_POINT_SIZE_MAX },
        uHighlightBrightness: { value: HIGHLIGHT_BRIGHTNESS_MULT },
        uHighlightAlpha: { value: HIGHLIGHT_ALPHA_MULT },
        uHighlightGlow: { value: HIGHLIGHT_GLOW_STRENGTH },
        uDimBrightness: { value: 1.0 }, // Updated based on selection
        uDimAlpha: { value: 1.0 }, // Updated based on selection
        uDimGlow: { value: DIM_GLOW_STRENGTH },
        uSelectedColorBase: { value: new THREE.Vector3(...SELECTED_COLOR_BASE) },
        uSelectedColorHighlight: { value: new THREE.Vector3(...SELECTED_COLOR_HIGHLIGHT) },
        uNeighborColorBase: { value: new THREE.Vector3(...NEIGHBOR_COLOR_BASE) },
        uNeighborColorHighlight: { value: new THREE.Vector3(...NEIGHBOR_COLOR_HIGHLIGHT) },
      },
      transparent: true,
      depthWrite: false,
      depthTest: true,
      blending: THREE.AdditiveBlending,
    });

    return { geometry: geo, material: mat, palette: pal };
  }, [data, splitMode]);

  // Update aVisible from visibleIds and ambiguousOnly
  useEffect(() => {
    if (!data || !geometry) return;
    const N = data.N;
    const aVisible = geometry.getAttribute("aVisible") as THREE.BufferAttribute;
    if (!aVisible) return;
    const arr = aVisible.array as Float32Array;
    for (let i = 0; i < N; i++) {
      let vis = 1;
      if (ambiguousOnly && data.ambiguous[i] === 0) vis = 0;
      if (visibleIds && !visibleIds.has(i)) vis = 0;
      arr[i] = vis;
    }
    aVisible.needsUpdate = true;
  }, [data, visibleIds, ambiguousOnly, geometry]);

  // Update aState from selectedId, neighbors, and brand selection
  useEffect(() => {
    if (!data || !geometry) return;
    const aState = geometry.getAttribute("aState") as THREE.BufferAttribute;
    if (!aState) return;
    const arr = aState.array as Float32Array;
    const N = data.N;
    const K_UI = data.K_UI;

    // Compute highlighted IDs (selected product + neighbors, or brand products)
    const highlightedIds = new Set<number>();

    if (selectedId != null && selectedId >= 0 && selectedId < N) {
      // Product selected: highlight selected + neighbors
      highlightedIds.add(selectedId);
      const splitId = data.splitId;
      const neigh = data.neighFused;
      let count = 0;
      for (let k = 0; k < data.K_GLOBAL && count < K_UI; k++) {
        const j: number = neigh[selectedId * data.K_GLOBAL + k];
        if (j === selectedId) continue;
        if (splitMode !== 0 && splitId[j] !== splitMode - 1) continue;
        if (ambiguousOnly && data.ambiguous[j] === 0) continue;
        if (visibleIds && !visibleIds.has(j)) continue;
        highlightedIds.add(j);
        count++;
      }
    } else if (selectedBrand != null && meta) {
      // Brand selected: highlight all products for that brand
      for (let i = 0; i < meta.length && i < N; i++) {
        if (meta[i].brand && meta[i].brand.toLowerCase() === selectedBrand.toLowerCase()) {
          highlightedIds.add(i);
        }
      }
    } else if (selectedStarIndex != null) {
      // Star selected: highlight member points
      const starId = data.starIds[selectedStarIndex];
      const members = data.coarseIdToMembers.get(starId);
      if (members) {
        for (const gid of members) {
          if (gid >= 0 && gid < N) highlightedIds.add(gid);
        }
      }
    }

    // Set states: selected=2, neighbor/highlighted=1, normal=0
    for (let i = 0; i < N; i++) {
      if (selectedId === i) {
        arr[i] = POINT_STATE_SELECTED;
      } else if (highlightedIds.has(i)) {
        arr[i] = POINT_STATE_NEIGHBOR; // Use neighbor state for any highlighted point
      } else {
        arr[i] = POINT_STATE_NORMAL;
      }
    }
    aState.needsUpdate = true;
  }, [data, selectedId, selectedStarIndex, selectedBrand, meta, splitMode, ambiguousOnly, visibleIds, geometry]);

  // Update uniforms
  useEffect(() => {
    if (!material || !(material as THREE.ShaderMaterial).uniforms) return;
    const uniforms = (material as THREE.ShaderMaterial).uniforms;
    uniforms.uSplitMode.value = splitMode;
    
    // Apply dimming when something is selected (but not this point)
    const hasSelection = selectedId != null || selectedStarIndex != null || selectedBrand != null;
    uniforms.uDimBrightness.value = hasSelection ? DIM_BRIGHTNESS_MULT : 1.0;
    uniforms.uDimAlpha.value = hasSelection ? DIM_ALPHA_MULT : 1.0;
  }, [splitMode, selectedId, selectedStarIndex, selectedBrand, material]);

  const setRef = (el: THREE.Points | null) => {
    (pointsRef as React.MutableRefObject<THREE.Points | null>).current = el;
    if (typeof ref === "function") ref(el);
    else if (ref) (ref as React.MutableRefObject<THREE.Points | null>).current = el;
  };

  if (!data || !geometry || !material) return null;

  return (
    <points
      ref={setRef}
      geometry={geometry}
      material={material as THREE.Material}
      renderOrder={1}
    />
  );
});

export default PointCloud;
