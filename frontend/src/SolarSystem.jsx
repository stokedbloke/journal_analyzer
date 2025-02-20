import React from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

// ✅ Ensure correct loading of textures
const textureLoader = new THREE.TextureLoader();
const planetTextures = {
  Mercury: textureLoader.load("/textures/mercury.jpg"),
  Venus: textureLoader.load("/textures/venus.jpg"),
  Earth: textureLoader.load("/textures/earth.jpg"),
  Mars: textureLoader.load("/textures/mars.jpg"),
  Jupiter: textureLoader.load("/textures/jupiter.jpg"),
  Saturn: textureLoader.load("/textures/saturn.jpg"),
  Uranus: textureLoader.load("/textures/uranus.jpg"),
  Neptune: textureLoader.load("/textures/neptune.jpg"),
  Moon: textureLoader.load("/textures/moon.jpg"),
};

// 🟢 Planet component with textures applied
function Planet({ name, position, size }) {
  return (
    <mesh position={position}>
      <sphereGeometry args={[size, 32, 32]} />
      <meshStandardMaterial
        map={planetTextures[name] || null}
        color={!planetTextures[name] ? "gray" : null} // ✅ Fallback color for missing textures
      />
    </mesh>
  );
}

function SolarSystem({ data, zoom }) {
  if (!data || !data.planets) {
    return <p>Loading planetary data...</p>; // ✅ Prevent crashing
  }

  const scale = zoom ? 50 : 15;
  const cameraPos = zoom ? [0, 0, 10] : [0, 0, 70];

  // ✅ Debugging planet textures
  console.log("🔍 Debugging Planet Textures:");
  Object.entries(planetTextures).forEach(([planet, texture]) => {
    console.log(`🔵 ${planet} Texture:`, texture ? "✅ Loaded" : "❌ Missing");
  });

  return (
    <Canvas camera={{ position: cameraPos, fov: 75 }}>
      <ambientLight intensity={1.2} />
      <OrbitControls enableZoom={false} />

      {/* ✅ Draw Orbit Rings with existence check */}
      {data?.planets?.map((planet, index) => (
        planet.orbitRadius && (
          <mesh key={`orbit-${index}`}>
            <ringGeometry args={[planet.orbitRadius * scale, planet.orbitRadius * scale + 0.05, 64]} />
            <meshBasicMaterial color="white" transparent opacity={0.3} side={THREE.DoubleSide} />
          </mesh>
        )
      ))}

      {/* ✅ Draw Planets with existence check */}
      {data?.planets?.map((planet, index) => (
        <Planet
          key={`planet-${index}`}
          name={planet.name}
          position={[planet.position[0] * scale, planet.position[1] * scale, 0]}
          size={planet.name === "Sun" ? 3 : 1} // 🌞 Make Sun bigger
        />
      ))}
    </Canvas>
  );
}

export { SolarSystem };
