// App.jsx

import React, { useState, useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import axios from "axios";
import Slider from "react-slider";
import * as d3 from "d3";
import * as THREE from "three";
import { SolarSystem } from "./SolarSystem";


const textureLoader = new THREE.TextureLoader();
const baseURL = "/textures"; // ✅ Replace process.env.PUBLIC_URL

const planetTextures = {
  Mercury: textureLoader.load(`${baseURL}/mercury.jpg`),
  Venus: textureLoader.load(`${baseURL}/venus.jpg`),
  Earth: textureLoader.load(`${baseURL}/earth.jpg`),
  Mars: textureLoader.load(`${baseURL}/mars.jpg`),
  Jupiter: textureLoader.load(`${baseURL}/jupiter.jpg`),
  Saturn: textureLoader.load(`${baseURL}/saturn.jpg`),
  Uranus: textureLoader.load(`${baseURL}/uranus.jpg`),
  Neptune: textureLoader.load(`${baseURL}/neptune.jpg`),
  Moon: textureLoader.load(`${baseURL}/moon.jpg`),
};


// Function to render orbit rings
function OrbitRing({ radius }) {
  return (
    <mesh>
      <ringGeometry args={[radius - 0.05, radius + 0.05, 128]} />
      <meshBasicMaterial color="white" transparent opacity={0.3} side={THREE.DoubleSide} />
    </mesh>
  );
}

// Function to render a planet or moon
function CelestialBody({ name, position, size }) {
  return (
    <mesh position={position}>
      <sphereGeometry args={[size, 32, 32]} />
      <meshStandardMaterial
        map={planetTextures[name] || null}
        color={!planetTextures[name] ? "gray" : null}
      />
    </mesh>
  );
}


// Backend API URL
const API_URL = "http://127.0.0.1:8000/get_data";

// Color scale for sentiment mapping
const sentimentColor = d3.scaleSequential(d3.interpolateRdBu).domain([-1, 1]);

const planetColors = [
  "gray",     // Mercury
  "orange",   // Venus
  "blue",     // Earth
  "red",      // Mars
  "yellow",   // Jupiter
  "lightblue", // Saturn
  "cyan",     // Uranus
  "darkblue"  // Neptune
];

function Planet({ position, size, texture }) {
  return (
    <mesh position={position}>
      <sphereGeometry args={[size, 32, 32]} />
      <meshStandardMaterial map={texture} />
    </mesh>
  );
}


function App() {
  const [date, setDate] = useState("2025-02-10");
  const [data, setData] = useState(null);
  const [zoom, setZoom] = useState(false);

  useEffect(() => {
    axios.get(`${API_URL}?date=${date}`)
      .then(response => {
        console.log("📥 API Response:", response.data);  // ✅ Debug API response
        setData(response.data);
      })
      .catch(error => console.error("API Error:", error));
  }, [date]);
  console.log("🚀 Debugging Data:", data);
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white">
      <h1 className="text-3xl font-bold mb-4">Interactive Solar System</h1>
      <p className="text-yellow-300 text-sm">(Debug: Current Date = {date})</p>
      
      {/* Date Slider */}
      <div className="w-1/2 mt-4">
      <Slider
        min={0}
        max={(new Date() - new Date(2020, 0, 1)) / (1000 * 60 * 60 * 24)} // Difference in days
        step={1}
        defaultValue={(new Date(2025, 1, 1) - new Date(2020, 0, 1)) / (1000 * 60 * 60 * 24)}
        onChange={(value) => {
          const newDate = new Date(2020, 0, 1);
          newDate.setDate(newDate.getDate() + value);
          setDate(newDate.toISOString().split("T")[0]);
      }}
          className="w-full h-2 bg-gray-400 rounded-full relative"
          thumbClassName="thumb w-6 h-6 bg-blue-500 rounded-full border border-white absolute z-10"
          trackClassName="h-2 bg-blue-500 rounded-full"
        />
      </div>

      {/* Restore Lost CSS for Slider */}
      <style>
        {`
        .thumb {
          width: 16px !important;
          height: 16px !important;
          background-color: blue !important;
          border-radius: 50% !important;
          border: 2px solid white !important;
          cursor: pointer;
        }
        `}
      </style>

      {/* Zoom Button */}
      <button onClick={() => setZoom(!zoom)} className="mt-4 p-2 bg-blue-500 text-white rounded">
        {zoom ? "Reset View" : "Zoom to Inner Planets"}
      </button>

      <p className="mt-2 text-lg">Selected Date: {date}</p>
      
      <div className="w-[90vw] h-[80vh] mt-4">
        {data?.planets?.length > 0 ? <SolarSystem data={data} zoom={zoom} /> : <p>Loading planetary data...</p>}
      </div>


      {data && (
        <div className="mt-4 text-center">
          <h2 className="text-xl font-bold">Emotions:</h2>
          <p className="text-lg font-bold">
            <span className="text-yellow-500">{data.emotion_1}</span>,
            <span className="text-green-500"> {data.emotion_2}</span>,
            <span className="text-blue-500"> {data.emotion_3}</span>
          </p>
          <h2 className="text-lg font-bold mt-2">Keywords:</h2>
          <p className="text-lg">{data?.keywords?.length ? data.keywords.join(", ") : "No keywords available"}</p>
        </div>
      )}
    </div>
  );
}

export default App;
