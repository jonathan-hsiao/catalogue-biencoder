"use client";

import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import { useStore } from "@/lib/store";
import {
  MAX_GRAVITY_LINES,
  GRAVITY_USE_DASHES,
  GRAVITY_DASH_SPEED,
  GRAVITY_DASH_SIZE,
  GRAVITY_GAP_SIZE,
  GRAVITY_PULSE_SPEED,
  GRAVITY_LINE_OPACITY_BASE,
  GRAVITY_LINE_OPACITY_PULSE_AMPLITUDE,
} from "@/lib/constants";

const dashedLineVertexShader = `
  attribute float lineDistance;
  uniform float scale;
  varying float vLineDistance;
  void main() {
    vLineDistance = scale * lineDistance;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const gravityLineFragmentShader = `
  uniform float useDashes;
  uniform float dashSize;
  uniform float totalSize;
  uniform float dashOffset;
  uniform float opacity;
  varying float vLineDistance;
  void main() {
    if (useDashes > 0.5 && mod(vLineDistance + dashOffset, totalSize) > dashSize) discard;
    gl_FragColor = vec4(1.0, 1.0, 1.0, opacity);
  }
`;

/** Evenly sample up to max indices from length. Deterministic. */
function sampleIndices(length: number, max: number): number[] {
  if (length <= max) {
    return Array.from({ length }, (_, i) => i);
  }
  const step = length / max;
  const indices: number[] = [];
  for (let i = 0; i < max; i++) {
    indices.push(Math.min(Math.floor(i * step), length - 1));
  }
  return indices;
}

export default function GravityLines() {
  const data = useStore((s) => s.data);
  const meta = useStore((s) => s.meta);
  const selectedStarIndex = useStore((s) => s.selectedStarIndex);
  const selectedId = useStore((s) => s.selectedId);
  const selectedBrand = useStore((s) => s.selectedBrand);

  const line = useMemo(() => {
    if (!data) return null;

    let positions: Float32Array;
    let numSegments: number;

    if (selectedStarIndex != null) {
      // Star selected: lines from star to member points
      const starId = data.starIds[selectedStarIndex];
      const members = data.coarseIdToMembers.get(starId);
      if (!members || members.length === 0) return null;

      const starX = data.starPos[selectedStarIndex * 3];
      const starY = data.starPos[selectedStarIndex * 3 + 1];
      const starZ = data.starPos[selectedStarIndex * 3 + 2];

      const indices = sampleIndices(members.length, MAX_GRAVITY_LINES);
      positions = new Float32Array(indices.length * 3 * 2);
      let i = 0;
      for (const idx of indices) {
        const gid = members[idx];
        positions[i++] = starX;
        positions[i++] = starY;
        positions[i++] = starZ;
        positions[i++] = data.posFused[gid * 3];
        positions[i++] = data.posFused[gid * 3 + 1];
        positions[i++] = data.posFused[gid * 3 + 2];
      }
      numSegments = indices.length;
    } else if (selectedId != null && selectedId >= 0 && selectedId < data.N) {
      // Product selected: line from product to its member star
      const coarseId = data.gtCoarseId[selectedId];
      let starIdx: number | null = null;
      for (let i = 0; i < data.starIds.length; i++) {
        if (data.starIds[i] === coarseId) {
          starIdx = i;
          break;
        }
      }
      if (starIdx == null) return null;

      const pointX = data.posFused[selectedId * 3];
      const pointY = data.posFused[selectedId * 3 + 1];
      const pointZ = data.posFused[selectedId * 3 + 2];
      const starX = data.starPos[starIdx * 3];
      const starY = data.starPos[starIdx * 3 + 1];
      const starZ = data.starPos[starIdx * 3 + 2];

      positions = new Float32Array(6);
      positions[0] = pointX;
      positions[1] = pointY;
      positions[2] = pointZ;
      positions[3] = starX;
      positions[4] = starY;
      positions[5] = starZ;
      numSegments = 1;
    } else if (selectedBrand != null && meta) {
      // Brand selected: lines from each brand product to its member star (sampled)
      const brandLower = selectedBrand.toLowerCase();
      const productStarPairs: { pointX: number; pointY: number; pointZ: number; starX: number; starY: number; starZ: number }[] = [];
      for (let i = 0; i < meta.length && i < data.N; i++) {
        if (!meta[i].brand || meta[i].brand.toLowerCase() !== brandLower) continue;
        const coarseId = data.gtCoarseId[i];
        let starIdx: number | null = null;
        for (let si = 0; si < data.starIds.length; si++) {
          if (data.starIds[si] === coarseId) {
            starIdx = si;
            break;
          }
        }
        if (starIdx == null) continue;
        productStarPairs.push({
          pointX: data.posFused[i * 3],
          pointY: data.posFused[i * 3 + 1],
          pointZ: data.posFused[i * 3 + 2],
          starX: data.starPos[starIdx * 3],
          starY: data.starPos[starIdx * 3 + 1],
          starZ: data.starPos[starIdx * 3 + 2],
        });
      }
      if (productStarPairs.length === 0) return null;
      const indices = sampleIndices(productStarPairs.length, MAX_GRAVITY_LINES);
      positions = new Float32Array(indices.length * 3 * 2);
      let idx = 0;
      for (const i of indices) {
        const p = productStarPairs[i];
        positions[idx++] = p.pointX;
        positions[idx++] = p.pointY;
        positions[idx++] = p.pointZ;
        positions[idx++] = p.starX;
        positions[idx++] = p.starY;
        positions[idx++] = p.starZ;
      }
      numSegments = indices.length;
    } else {
      return null;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geo.setDrawRange(0, numSegments * 2);

    const lineDistances = new Float32Array(numSegments * 2);
    const start = new THREE.Vector3();
    const end = new THREE.Vector3();
    for (let s = 0; s < numSegments; s++) {
      const i = s * 6;
      start.set(positions[i], positions[i + 1], positions[i + 2]);
      end.set(positions[i + 3], positions[i + 4], positions[i + 5]);
      lineDistances[s * 2] = 0;
      lineDistances[s * 2 + 1] = start.distanceTo(end);
    }
    geo.setAttribute("lineDistance", new THREE.Float32BufferAttribute(lineDistances, 1));

    const mat = new THREE.ShaderMaterial({
      vertexShader: dashedLineVertexShader,
      fragmentShader: gravityLineFragmentShader,
      uniforms: {
        scale: { value: 1 },
        useDashes: { value: GRAVITY_USE_DASHES ? 1 : 0 },
        dashSize: { value: GRAVITY_DASH_SIZE },
        totalSize: { value: GRAVITY_DASH_SIZE + GRAVITY_GAP_SIZE },
        dashOffset: { value: 0 },
        opacity: { value: GRAVITY_LINE_OPACITY_BASE },
      },
      transparent: true,
      depthTest: true,
      depthWrite: false,
    });
    return { geometry: geo, material: mat };
  }, [data, meta, selectedStarIndex, selectedId, selectedBrand]);

  const materialRef = useRef<THREE.ShaderMaterial | null>(null);
  if (line?.material) materialRef.current = line.material as THREE.ShaderMaterial;

  useFrame((state, delta) => {
    const mat = materialRef.current;
    if (!mat?.uniforms) return;
    if (GRAVITY_USE_DASHES && mat.uniforms.dashOffset) {
      (mat.uniforms.dashOffset as THREE.IUniform<number>).value -= delta * GRAVITY_DASH_SPEED;
    }
    const t = state.clock.elapsedTime;
    (mat.uniforms.opacity as THREE.IUniform<number>).value =
      GRAVITY_LINE_OPACITY_BASE + GRAVITY_LINE_OPACITY_PULSE_AMPLITUDE * Math.sin(t * GRAVITY_PULSE_SPEED);
  });

  if (!line) return null;
  return (
    <lineSegments
      geometry={line.geometry}
      material={line.material}
      renderOrder={0}
    />
  );
}
