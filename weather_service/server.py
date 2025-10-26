from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Weather Service API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# You should set this environment variable with your WeatherAPI key
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "your_api_key_here")
MOLDOVA_CITY = "Chisinau"  # Capital city of Moldova

@app.get("/")
async def root():
    return {"message": "Weather Service API"}

@app.get("/weather")
async def get_moldova_weather():
    """
    Get the current weather in Moldova (Chisinau)
    """
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHERAPI_KEY}&q={MOLDOVA_CITY}&aqi=no"
        response = requests.get(url)
        response.raise_for_status()
        
        weather_data = response.json()
        
        return {
            "city": MOLDOVA_CITY,
            "temperature": weather_data["current"]["temp_c"],
            "feels_like": weather_data["current"]["feelslike_c"],
            "humidity": weather_data["current"]["humidity"],
            "description": weather_data["current"]["condition"]["text"],
            "wind_speed": weather_data["current"]["wind_kph"] / 3.6  # Convert km/h to m/s
        }
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch weather data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)