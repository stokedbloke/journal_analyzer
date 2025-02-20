from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import ephem
import os
import numpy as np
import pandas as pd
import math

import ephem
print(dir(ephem))  # Should list 'Earth', 'Mars', etc.


app = FastAPI()

# Enable CORS (Allows frontend to access API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # ✅ Allow only frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # ✅ Restrict to necessary methods
    allow_headers=["*"],
)

print("✅ CORS Middleware Enabled")

# Database paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(ROOT_DIR, "analyzed_journal_entries.csv")

# Load the CSV file at startup
if os.path.exists(CSV_PATH):
    journal_data = pd.read_csv(CSV_PATH)
    journal_data["date_created"] = pd.to_datetime(journal_data["date_created"]).dt.strftime("%Y-%m-%d")
else:
    journal_data = None

import ephem
import math

import ephem
import math

def get_planet_positions(date):
    """Compute planet positions using PyEphem and normalize to circular orbits."""
    planets = [
        {"name": "Mercury", "obj": ephem.Mercury()},
        {"name": "Venus", "obj": ephem.Venus()},
        {"name": "Mars", "obj": ephem.Mars()},
        {"name": "Jupiter", "obj": ephem.Jupiter()},
        {"name": "Saturn", "obj": ephem.Saturn()},
        {"name": "Uranus", "obj": ephem.Uranus()},
        {"name": "Neptune", "obj": ephem.Neptune()}
    ]

    positions = [{"name": "Sun", "position": [0, 0], "orbitRadius": 0}]

    previous_angle = None  # Track previous day's angle

    for planet in planets:
        planet["obj"].compute(date)  # Compute position for the given date
        ra = float(planet["obj"].ra)  # Right Ascension (angle)
        distance = float(planet["obj"].sun_distance)  # Distance from Sun

        # ✅ Normalize angle to avoid erratic jumps
        angle = (ra * (2 * math.pi / 24)) % (2 * math.pi)

        # ✅ Ensure smooth angle continuity
        if previous_angle is not None:
            while abs(angle - previous_angle) > math.pi:
                angle -= math.copysign(2 * math.pi, angle - previous_angle)

        previous_angle = angle

        x = distance * math.cos(angle)
        y = distance * math.sin(angle)

        positions.append({"name": planet["name"], "position": [x, y], "orbitRadius": distance})

    # ✅ Compute Earth's position separately and normalize
    sun = ephem.Sun()
    sun.compute(date)
    earth_distance = float(sun.earth_distance)  # Distance from Sun
    earth_ra = float(sun.ra)  # Right Ascension

    earth_angle = (earth_ra * (2 * math.pi / 24)) % (2 * math.pi)

    # ✅ Ensure smooth transition for Earth
    if previous_angle is not None:
        while abs(earth_angle - previous_angle) > math.pi:
            earth_angle -= math.copysign(2 * math.pi, earth_angle - previous_angle)

    earth_x = -earth_distance * math.cos(earth_angle)  # Earth opposite Sun
    earth_y = -earth_distance * math.sin(earth_angle)

    positions.append({"name": "Earth", "position": [earth_x, earth_y], "orbitRadius": earth_distance})

    # ✅ Compute Moon's position relative to Earth with smoother transition
    try:
        moon = ephem.Moon()
        moon.compute(date)
        moon_ra = float(moon.ra)
        moon_distance = 0.1  # Scaled down for UI

        moon_angle = (moon_ra * (2 * math.pi / 24)) % (2 * math.pi)

        # ✅ Ensure smooth transition for Moon
        if previous_angle is not None:
            while abs(moon_angle - previous_angle) > math.pi:
                moon_angle -= math.copysign(2 * math.pi, moon_angle - previous_angle)

        moon_x = earth_x + moon_distance * math.cos(moon_angle)
        moon_y = earth_y + moon_distance * math.sin(moon_angle)

        positions.append({"name": "Moon", "position": [moon_x, moon_y], "orbitRadius": 0.1})
        print(f"✅ Moon Calculated: {moon_x}, {moon_y}")
    
    except Exception as e:
        print(f"🚨 Moon Calculation Error: {e}")

    return positions




@app.get("/get_data")
def get_data(date: str):
    try:
        print(f"📅 Fetching data for date: {date}")
        planets = get_planet_positions(date)
        if journal_data is None:
            return {"error": "Journal data file is missing"}

        entry = journal_data[journal_data["date_created"].str.contains(date, na=False)]
        sentiment = entry["dominant_score"].mean() if not entry.empty else 0
        keywords = entry["keywords"].tolist() if not entry.empty else []
        
        emotion_1 = entry["emotion_1"].values[0] if not entry.empty else "N/A"
        emotion_2 = entry["emotion_2"].values[0] if not entry.empty else "N/A"
        emotion_3 = entry["emotion_3"].values[0] if not entry.empty else "N/A"

        return {
            "date": date,
            "planets": planets,
            "sentiment_score": float(sentiment) if sentiment else 0,
            "keywords": keywords,
            "emotion_1": emotion_1,
            "emotion_2": emotion_2,
            "emotion_3": emotion_3,
        }
    
    except Exception as e:
        print(f"❌ ERROR in /get_data: {str(e)}")
        return {"error": str(e)}
