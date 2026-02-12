"use client";

import { useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { Suspense } from "react";
import * as THREE from "three";
import PointCloud from "./PointCloud";
import Stars from "./Stars";
import GravityLines from "./GravityLines";
import CameraControls from "./CameraControls";
import PickingLayer from "./PickingLayer";
import ControlPanel from "../ui/ControlPanel";
import Tooltip from "../ui/Tooltip";
import Sidebar from "../ui/Sidebar";
import { useStore } from "@/lib/store";
import { DEFAULT_VIEW_DISTANCE, DEFAULT_VIEW_ANGLE } from "@/lib/constants";

export default function Scene3D() {
  const pointCloudRef = useRef<THREE.Points | null>(null);
  const starsRef = useRef<THREE.Points>(null);

  return (
    <>
      <div className="absolute inset-0">
        <Canvas
          gl={{ antialias: true, alpha: false }}
          camera={{
            position: [
              DEFAULT_VIEW_DISTANCE * Math.sin(DEFAULT_VIEW_ANGLE),
              0,
              DEFAULT_VIEW_DISTANCE * Math.cos(DEFAULT_VIEW_ANGLE),
            ],
            fov: 50,
            near: 0.1,
            far: 1000,
          }}
          onCreated={({ gl }) => {
            gl.setClearColor(0x000008);
          }}
        >
          <Suspense fallback={null}>
            <ambientLight intensity={0.12} />
            <pointLight position={[10, 10, 10]} intensity={0.35} />
            <pointLight position={[-10, -10, 5]} intensity={0.2} />
            <PointCloud ref={pointCloudRef} />
            <Stars ref={starsRef} />
            <GravityLines />
            <PickingLayer pointCloudRef={pointCloudRef} starsRef={starsRef} />
            <CameraControls />
          </Suspense>
        </Canvas>
      </div>
      <ControlPanel />
      <Tooltip />
      <Sidebar />
    </>
  );
}
