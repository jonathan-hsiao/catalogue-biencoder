"use client";

import { useEffect, useRef, useMemo } from "react";
import * as THREE from "three";
import { useThree, useFrame } from "@react-three/fiber";
import { useStore } from "@/lib/store";
import { POINT_STATE_NORMAL, POINT_STATE_NEIGHBOR, POINT_STATE_SELECTED } from "@/lib/constants";

/**
 * GPU-based picking: render point indices to a texture, read pixel under cursor.
 * Pixel-perfect and no raycast/threshold mismatch.
 */

const pickVertexShader = `
  attribute float aVisible;
  attribute float aSplitId;
  attribute float aIndex;
  uniform float uSplitMode;
  uniform float uProductCount;
  varying float vIndex;
  varying float vVisible;
  void main() {
    vIndex = aIndex;
    // Stars (index >= productCount) are always visible regardless of split mode
    bool isStar = aIndex >= uProductCount;
    float showBySplit = isStar ? 1.0 : ((uSplitMode < 0.5) ? 1.0 : (abs(aSplitId - (uSplitMode - 1.0)) < 0.5 ? 1.0 : 0.0));
    vVisible = aVisible * showBySplit;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    float depth = -mvPosition.z;
    float size = (depth > 0.01) ? clamp(12.0 / depth, 2.0, 48.0) : 2.0;
    gl_PointSize = size;
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const pickFragmentShader = `
  varying float vIndex;
  varying float vVisible;
  void main() {
    if (vVisible < 0.5) discard;
    float d = length(gl_PointCoord - 0.5) * 2.0;
    if (d > 1.0) discard;
    float id = vIndex + 1.0;
    float r = mod(id, 256.0) / 255.0;
    float g = mod(floor(id / 256.0), 256.0) / 255.0;
    float b = mod(floor(id / 65536.0), 256.0) / 255.0;
    gl_FragColor = vec4(r, g, b, 1.0);
  }
`;

function decodeIndex(r: number, g: number, b: number): number {
  const raw = Math.round(r * 255 + g * 255 * 256 + b * 255 * 65536);
  return raw - 1;
}

interface PickingLayerProps {
  pointCloudRef: React.RefObject<THREE.Points | null>;
  starsRef: React.RefObject<THREE.Points | null>;
}

export default function PickingLayer({ pointCloudRef, starsRef }: PickingLayerProps) {
  const { gl, camera, size } = useThree();
  const pickTargetRef = useRef<THREE.WebGLRenderTarget | null>(null);
  const pickSceneRef = useRef<THREE.Scene | null>(null);
  const pickCameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const pointsPickRef = useRef<THREE.Points | null>(null);
  const starsPickRef = useRef<THREE.Points | null>(null);
  const pixelBuffer = useRef(new Uint8Array(4));
  const pendingDeselectRef = useRef(false);
  const downTimeRef = useRef(0);
  const didDragRef = useRef(false);
  const pointerDownRef = useRef(false);
  const downClientXRef = useRef(0);
  const downClientYRef = useRef(0);
  const lastEmptyClickTimeRef = useRef(0);

  const CLICK_MAX_MS = 300;
  const DRAG_THRESHOLD_PX = 5;
  const DBL_CLICK_MS = 400;

  const data = useStore((s) => s.data);
  const splitMode = useStore((s) => s.splitMode);
  const visibleIds = useStore((s) => s.visibleIds);
  const ambiguousOnly = useStore((s) => s.ambiguousOnly);
  const interactiveStars = useStore((s) => s.interactiveStars);
  const interactivePoints = useStore((s) => s.interactivePoints);
  const interactiveDimmed = useStore((s) => s.interactiveDimmed);
  const selectedId = useStore((s) => s.selectedId);
  const selectedStarIndex = useStore((s) => s.selectedStarIndex);
  const selectedBrand = useStore((s) => s.selectedBrand);
  const meta = useStore((s) => s.meta);

  const N = data?.N ?? 0;
  const numStars = data?.starIds?.length ?? 0;

  const pickScene = useMemo(() => {
    const scene = new THREE.Scene();
    pickSceneRef.current = scene;
    return scene;
  }, []);

  const pickCamera = useMemo(() => {
    const cam = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
    pickCameraRef.current = cam;
    return cam;
  }, []);

  const pointsPickGeo = useMemo(() => {
    if (!data) return null;
    const N = data.N;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(data.posFused, 3));
    geo.setAttribute("aSplitId", new THREE.Float32BufferAttribute(new Float32Array(data.splitId), 1));
    const aVisible = new Float32Array(N);
    for (let i = 0; i < N; i++) aVisible[i] = 1;
    geo.setAttribute("aVisible", new THREE.Float32BufferAttribute(aVisible, 1));
    const aIndex = new Float32Array(N);
    for (let i = 0; i < N; i++) aIndex[i] = i;
    geo.setAttribute("aIndex", new THREE.Float32BufferAttribute(aIndex, 1));
    // aState will be updated in useEffect based on selection
    const aState = new Float32Array(N).fill(0);
    geo.setAttribute("aState", new THREE.Float32BufferAttribute(aState, 1));
    return geo;
  }, [data]);

  const starsPickGeo = useMemo(() => {
    if (!data) return null;
    const { starPos, starIds } = data;
    const nStars = starIds.length;
    const productCount = data.N;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(starPos, 3));
    const aIndex = new Float32Array(nStars);
    for (let i = 0; i < nStars; i++) aIndex[i] = productCount + i;
    geo.setAttribute("aIndex", new THREE.Float32BufferAttribute(aIndex, 1));
    const aVisible = new Float32Array(nStars).fill(1);
    geo.setAttribute("aVisible", new THREE.Float32BufferAttribute(aVisible, 1));
    const aSplitId = new Float32Array(nStars).fill(0);
    geo.setAttribute("aSplitId", new THREE.Float32BufferAttribute(aSplitId, 1));
    return geo;
  }, [data]);

  const pointsPickMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: pickVertexShader,
        fragmentShader: pickFragmentShader,
        uniforms: { 
          uSplitMode: { value: 0 },
          uProductCount: { value: data?.N ?? 0 },
        },
        depthWrite: true,
        depthTest: true,
      }),
    [data]
  );

  const starsPickMat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: pickVertexShader,
        fragmentShader: pickFragmentShader,
        uniforms: { 
          uSplitMode: { value: 0 },
          uProductCount: { value: data?.N ?? 0 },
        },
        depthWrite: true,
        depthTest: true,
      }),
    [data]
  );

  useEffect(() => {
    if (!pointsPickGeo || !pointsPickMat) return;
    // Only add points to pick scene if interactivePoints is enabled
    if (!interactivePoints) {
      if (pointsPickRef.current) {
        pickScene.remove(pointsPickRef.current);
        pointsPickRef.current = null;
      }
      return;
    }
    const points = new THREE.Points(pointsPickGeo, pointsPickMat);
    if (pointsPickRef.current) pickScene.remove(pointsPickRef.current);
    pointsPickRef.current = points;
    pickScene.add(points);
    return () => {
      pickScene.remove(points);
      pointsPickRef.current = null;
    };
  }, [pickScene, pointsPickGeo, pointsPickMat, interactivePoints]);

  useEffect(() => {
    if (!starsPickGeo || !starsPickMat) return;
    // Only add stars to pick scene if interactiveStars is enabled
    if (!interactiveStars) {
      if (starsPickRef.current) {
        pickScene.remove(starsPickRef.current);
        starsPickRef.current = null;
      }
      return;
    }
    const stars = new THREE.Points(starsPickGeo, starsPickMat);
    if (starsPickRef.current) pickScene.remove(starsPickRef.current);
    starsPickRef.current = stars;
    pickScene.add(stars);
    return () => {
      pickScene.remove(stars);
      starsPickRef.current = null;
    };
  }, [pickScene, starsPickGeo, starsPickMat, interactiveStars]);

  useEffect(() => {
    if (!data || !pointsPickRef.current) return;
    const geo = pointsPickRef.current.geometry;
    const aVisible = geo.getAttribute("aVisible") as THREE.BufferAttribute;
    if (!aVisible) return;
    const arr = aVisible.array as Float32Array;
    for (let i = 0; i < data.N; i++) {
      let vis = 1;
      if (ambiguousOnly && data.ambiguous[i] === 0) vis = 0;
      if (visibleIds && !visibleIds.has(i)) vis = 0;
      arr[i] = vis;
    }
    aVisible.needsUpdate = true;
  }, [data, visibleIds, ambiguousOnly]);

  // Update aState attribute for points (same logic as PointCloud)
  useEffect(() => {
    if (!data || !pointsPickRef.current) return;
    const geo = pointsPickRef.current.geometry;
    const aState = geo.getAttribute("aState") as THREE.BufferAttribute;
    if (!aState) return;
    const arr = aState.array as Float32Array;
    const N = data.N;
    const K_UI = data.K_UI;

    // Compute highlighted IDs (selected product + neighbors, or brand products, or star members)
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
        arr[i] = POINT_STATE_NEIGHBOR;
      } else {
        arr[i] = POINT_STATE_NORMAL;
      }
    }
    aState.needsUpdate = true;
  }, [data, selectedId, selectedStarIndex, selectedBrand, meta, splitMode, ambiguousOnly, visibleIds]);

  useEffect(() => {
    if (!data) return;
    (pointsPickMat as THREE.ShaderMaterial).uniforms.uSplitMode.value = splitMode;
    (starsPickMat as THREE.ShaderMaterial).uniforms.uSplitMode.value = splitMode;
    (pointsPickMat as THREE.ShaderMaterial).uniforms.uProductCount.value = data.N;
    (starsPickMat as THREE.ShaderMaterial).uniforms.uProductCount.value = data.N;
  }, [splitMode, pointsPickMat, starsPickMat, data]);

  useFrame(() => {
    if (!data || !pickSceneRef.current || !pickCameraRef.current) return;
    const rt = pickTargetRef.current;
    if (!rt || rt.width !== size.width || rt.height !== size.height) {
      if (rt) rt.dispose();
      pickTargetRef.current = new THREE.WebGLRenderTarget(size.width, size.height, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        format: THREE.RGBAFormat,
        type: THREE.UnsignedByteType,
      });
    }
    const target = pickTargetRef.current!;
    const cam = camera as THREE.PerspectiveCamera;
    pickCameraRef.current!.copy(cam);
    pickCameraRef.current!.aspect = size.width / size.height;
    pickCameraRef.current!.updateProjectionMatrix();
    gl.setRenderTarget(target);
    const oldClear = gl.getClearColor(new THREE.Color());
    const oldAlpha = gl.getClearAlpha();
    gl.setClearColor(0x000000, 0);
    gl.clear(true, true, false);
    gl.setClearColor(oldClear, oldAlpha);
    gl.render(pickSceneRef.current!, pickCameraRef.current!);
    gl.setRenderTarget(null);
  });

  useEffect(() => {
    const canvas = gl.domElement;

    const readPixel = (ndcX: number, ndcY: number): number | null => {
      const rt = pickTargetRef.current;
      if (!rt) return null;
      const x = Math.floor((ndcX * 0.5 + 0.5) * rt.width);
      const yClip = Math.floor((-ndcY * 0.5 + 0.5) * rt.height);
      const y = rt.height - 1 - yClip;
      if (x < 0 || x >= rt.width || y < 0 || y >= rt.height) return null;
      const ctx = gl.getContext() as WebGL2RenderingContext;
      gl.setRenderTarget(rt);
      ctx.readPixels(x, y, 1, 1, ctx.RGBA, ctx.UNSIGNED_BYTE, pixelBuffer.current);
      gl.setRenderTarget(null);
      const r = pixelBuffer.current[0] / 255;
      const g = pixelBuffer.current[1] / 255;
      const b = pixelBuffer.current[2] / 255;
      const a = pixelBuffer.current[3];
      if (a === 0) return null;
      return decodeIndex(r, g, b);
    };

    // Helper: compute highlighted point IDs (same logic as PointCloud)
    const computeHighlightedPointIds = (): Set<number> => {
      const highlightedIds = new Set<number>();
      const storeState = useStore.getState();
      if (!data) return highlightedIds;
      const N = data.N;
      const K_UI = data.K_UI;

      if (storeState.selectedId != null && storeState.selectedId >= 0 && storeState.selectedId < N) {
        highlightedIds.add(storeState.selectedId);
        const splitId = data.splitId;
        const neigh = data.neighFused;
        let count = 0;
        for (let k = 0; k < data.K_GLOBAL && count < K_UI; k++) {
          const j: number = neigh[storeState.selectedId * data.K_GLOBAL + k];
          if (j === storeState.selectedId) continue;
          if (splitMode !== 0 && splitId[j] !== splitMode - 1) continue;
          if (ambiguousOnly && data.ambiguous[j] === 0) continue;
          if (visibleIds && !visibleIds.has(j)) continue;
          highlightedIds.add(j);
          count++;
        }
      } else if (storeState.selectedBrand != null && meta) {
        for (let i = 0; i < meta.length && i < N; i++) {
          if (meta[i].brand && meta[i].brand.toLowerCase() === storeState.selectedBrand.toLowerCase()) {
            highlightedIds.add(i);
          }
        }
      } else if (storeState.selectedStarIndex != null) {
        const starId = data.starIds[storeState.selectedStarIndex];
        const members = data.coarseIdToMembers.get(starId);
        if (members) {
          for (const gid of members) {
            if (gid >= 0 && gid < N) highlightedIds.add(gid);
          }
        }
      }
      return highlightedIds;
    };

    // Helper: compute highlighted star indices (same logic as Stars)
    const computeHighlightedStarIndices = (): Set<number> => {
      const highlightedStarIndices = new Set<number>();
      const storeState = useStore.getState();
      if (!data) return highlightedStarIndices;
      const hasSelection = storeState.selectedId != null || storeState.selectedStarIndex != null || storeState.selectedBrand != null;
      if (!hasSelection) return highlightedStarIndices;

      if (storeState.selectedStarIndex != null) {
        highlightedStarIndices.add(storeState.selectedStarIndex);
      } else if (storeState.selectedId != null) {
        const coarseId = data.gtCoarseId[storeState.selectedId];
        for (let i = 0; i < numStars; i++) {
          if (data.starIds[i] === coarseId) {
            highlightedStarIndices.add(i);
            break;
          }
        }
      } else if (storeState.selectedBrand != null && meta) {
        const brandProductIds = new Set<number>();
        for (let i = 0; i < meta.length && i < data.N; i++) {
          if (meta[i].brand && meta[i].brand.toLowerCase() === storeState.selectedBrand.toLowerCase()) {
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
      return highlightedStarIndices;
    };

    // Helper: check if a point is dimmed
    const isPointDimmed = (pointIndex: number, highlightedPointIds: Set<number>): boolean => {
      const storeState = useStore.getState();
      const hasSelection = storeState.selectedId != null || storeState.selectedStarIndex != null || storeState.selectedBrand != null;
      if (!hasSelection) return false; // Nothing is dimmed when nothing is selected
      return !highlightedPointIds.has(pointIndex) && storeState.selectedId !== pointIndex;
    };

    // Helper: check if a star is dimmed
    const isStarDimmed = (starIndex: number, highlightedStarIndices: Set<number>): boolean => {
      const storeState = useStore.getState();
      const hasSelection = storeState.selectedId != null || storeState.selectedStarIndex != null || storeState.selectedBrand != null;
      if (!hasSelection) return false; // Nothing is dimmed when nothing is selected
      return !highlightedStarIndices.has(starIndex);
    };

    const onPointerMove = (e: PointerEvent) => {
      if (pointerDownRef.current) {
        const dx = Math.abs(e.clientX - downClientXRef.current);
        const dy = Math.abs(e.clientY - downClientYRef.current);
        if (dx > DRAG_THRESHOLD_PX || dy > DRAG_THRESHOLD_PX) didDragRef.current = true;
      }
      if (useStore.getState().cameraMoving) return;
      const storeState = useStore.getState();
      const rect = canvas.getBoundingClientRect();
      const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      const index = readPixel(ndcX, ndcY);
      if (index == null || index < 0) {
        storeState.setHoveredId(null);
        storeState.setHoveredStarIndex(null);
        return;
      }
      
      // Check if dimmed elements should be filtered
      if (!storeState.interactiveDimmed) {
        const highlightedPointIds = computeHighlightedPointIds();
        const highlightedStarIndices = computeHighlightedStarIndices();
        if (index < N) {
          if (isPointDimmed(index, highlightedPointIds)) {
            storeState.setHoveredId(null);
            storeState.setHoveredStarIndex(null);
            return;
          }
        } else if (index < N + numStars) {
          const starIdx = index - N;
          if (isStarDimmed(starIdx, highlightedStarIndices)) {
            storeState.setHoveredId(null);
            storeState.setHoveredStarIndex(null);
            return;
          }
        }
      }

      if (index < N) {
        storeState.setHoveredId(index);
        storeState.setHoveredStarIndex(null);
      } else if (index < N + numStars) {
        storeState.setHoveredId(null);
        storeState.setHoveredStarIndex(index - N);
      } else {
        storeState.setHoveredId(null);
        storeState.setHoveredStarIndex(null);
      }
    };

    const selectCurrentHover = () => {
      const { hoveredId, hoveredStarIndex, data: storeData, shardManager, splitMode: sm, ambiguousOnly: amb, visibleIds: vis } =
        useStore.getState();
      if (hoveredId == null && hoveredStarIndex == null) return;
      useStore.getState().setSelectedStarIndex(hoveredStarIndex ?? null);
      useStore.getState().setSelectedId(hoveredId ?? null);
      useStore.getState().setSelectedBrand(null); // Clear brand when selecting product/star
      if (hoveredId != null && storeData && shardManager) {
        const ids: number[] = [hoveredId];
        let count = 0;
        for (let k = 0; k < storeData.K_GLOBAL && count < storeData.K_UI; k++) {
          const j: number = storeData.neighFused[hoveredId * storeData.K_GLOBAL + k];
          if (j === hoveredId) continue;
          if (sm !== 0 && storeData.splitId[j] !== sm - 1) continue;
          if (amb && storeData.ambiguous[j] === 0) continue;
          if (vis && !vis.has(j)) continue;
          ids.push(j);
          count++;
        }
        shardManager.prefetchShardsForIds(ids);
      }
    };

    const onPointerDown = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const isOnCanvas =
        e.clientX >= rect.left && e.clientX <= rect.right &&
        e.clientY >= rect.top && e.clientY <= rect.bottom;
      if (!isOnCanvas) {
        pointerDownRef.current = false;
        pendingDeselectRef.current = false;
        return;
      }
      pointerDownRef.current = true;
      downClientXRef.current = e.clientX;
      downClientYRef.current = e.clientY;
      didDragRef.current = false;
      const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      const pixelIndex = readPixel(ndcX, ndcY);
      const storeState = useStore.getState();
      
      // Check if dimmed elements should be filtered
      let isEmpty = pixelIndex == null || pixelIndex < 0;
      if (!isEmpty && pixelIndex != null && !storeState.interactiveDimmed) {
        const highlightedPointIds = computeHighlightedPointIds();
        const highlightedStarIndices = computeHighlightedStarIndices();
        if (pixelIndex < N) {
          if (isPointDimmed(pixelIndex, highlightedPointIds)) {
            isEmpty = true;
          }
        } else if (pixelIndex < N + numStars) {
          const starIdx = pixelIndex - N;
          if (isStarDimmed(starIdx, highlightedStarIndices)) {
            isEmpty = true;
          }
        }
      }
      
      const { hoveredId, hoveredStarIndex } = storeState;
      isEmpty = isEmpty || (hoveredId == null && hoveredStarIndex == null);
      if (isEmpty) {
        pendingDeselectRef.current = true;
        downTimeRef.current = Date.now();
      } else {
        pendingDeselectRef.current = false;
      }
      selectCurrentHover();
    };

    const onPointerUpGlobal = (e: PointerEvent) => {
      const target = e?.target as Node | null;
      const isSidebarOrUI = target && typeof (target as Element).closest === "function" &&
        ((target as Element).closest("[data-ui=\"sidebar\"]") ?? (target as Element).closest("[data-ui=\"control-panel\"]"));
      if (isSidebarOrUI) {
        pendingDeselectRef.current = false;
        pointerDownRef.current = false;
        return;
      }
      if (
        pendingDeselectRef.current &&
        !didDragRef.current &&
        Date.now() - downTimeRef.current < CLICK_MAX_MS
      ) {
        const now = Date.now();
        if (now - lastEmptyClickTimeRef.current < DBL_CLICK_MS) {
          useStore.getState().resetView();
          lastEmptyClickTimeRef.current = 0;
        } else {
          lastEmptyClickTimeRef.current = now;
          useStore.getState().setSelectedId(null);
          useStore.getState().setSelectedStarIndex(null);
          useStore.getState().setSelectedBrand(null);
        }
      }
      pendingDeselectRef.current = false;
      pointerDownRef.current = false;
    };

    const onPointerLeave = () => {
      useStore.getState().setHoveredId(null);
      useStore.getState().setHoveredStarIndex(null);
    };

    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerleave", onPointerLeave);
    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointerup", onPointerUpGlobal);
    return () => {
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerleave", onPointerLeave);
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointerup", onPointerUpGlobal);
    };
  }, [gl, N, numStars]);

  return null;
}
