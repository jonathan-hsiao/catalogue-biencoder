"use client";

import { useRef, useMemo } from "react";
import { useThree, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { useStore } from "@/lib/store";
import { FLY_DURATION_MS, FLY_EASE_POWER } from "@/lib/constants";

/** Symmetric ease-in-out: smooth C1 join at t=0.5, slow start and slow end. */
function flyEase(t: number): number {
  if (t <= 0.5) {
    return 0.5 * Math.pow(2 * t, FLY_EASE_POWER);
  }
  return 1 - 0.5 * Math.pow(2 * (1 - t), FLY_EASE_POWER);
}

export default function CameraControls() {
  const controlsRef = useRef<typeof OrbitControls>(null);
  const { camera } = useThree();
  const setCameraMoving = useStore((s) => s.setCameraMoving);
  const consumeFlyTo = useStore((s) => s.consumeFlyTo);

  const anim = useRef<{
    startTime: number;
    startPosition: THREE.Vector3;
    startTarget: THREE.Vector3;
    endPosition: THREE.Vector3;
    endTarget: THREE.Vector3;
  } | null>(null);

  const vecs = useMemo(
    () => ({
      startPosition: new THREE.Vector3(),
      startTarget: new THREE.Vector3(),
      endPosition: new THREE.Vector3(),
      endTarget: new THREE.Vector3(),
      dir: new THREE.Vector3(),
    }),
    []
  );

  useFrame((_, delta) => {
    if (anim.current) {
      const { startTime, startPosition, startTarget, endPosition, endTarget } =
        anim.current;
      const elapsed = (performance.now() - startTime) / 1000;
      const t = Math.min(1, elapsed / (FLY_DURATION_MS / 1000));
      const tEased = flyEase(t);

      const c = camera as THREE.PerspectiveCamera;
      c.position.lerpVectors(startPosition, endPosition, tEased);
      const controls = controlsRef.current as unknown as { target: THREE.Vector3 } | null;
      if (controls?.target) {
        controls.target.lerpVectors(startTarget, endTarget, tEased);
      }

      if (t >= 1) {
        anim.current = null;
        setCameraMoving(false);
      }
      return;
    }

    const pending = useStore.getState().flyToPoint;
    if (!pending) return;
    const fly = consumeFlyTo();
    if (!fly || !controlsRef.current) return;

    const target = new THREE.Vector3(fly.pos[0], fly.pos[1], fly.pos[2]);
    const c = camera as THREE.PerspectiveCamera;
    let endPosition: THREE.Vector3;
    if (fly.angle != null) {
      endPosition = target.clone().add(
        new THREE.Vector3(
          fly.distance * Math.sin(fly.angle),
          0,
          fly.distance * Math.cos(fly.angle)
        )
      );
    } else {
      vecs.dir.copy(c.position).sub(target).normalize();
      endPosition = target.clone().add(vecs.dir.clone().multiplyScalar(fly.distance));
    }
    const controls = controlsRef.current as unknown as { target: THREE.Vector3 };
    const startTarget = controls.target.clone();

    setCameraMoving(true);
    anim.current = {
      startTime: performance.now(),
      startPosition: c.position.clone(),
      startTarget,
      endPosition,
      endTarget: target,
    };
  });

  return (
    <OrbitControls
      ref={controlsRef}
      enableDamping
      dampingFactor={0.05}
      minDistance={0.2}
      maxDistance={20}
      onStart={() => setCameraMoving(true)}
      onEnd={() => setCameraMoving(false)}
    />
  );
}
