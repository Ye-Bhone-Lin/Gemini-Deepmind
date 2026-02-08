import requests
import os
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

WEATHER_API_KEY = os.getenv('OPENWEATHER_MAP_API_KEY')

def get_weather(city: str):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="City not found or API error")
    data = response.json()
    weather_info = {
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "rain": data.get("rain", {}).get("1h", 0),
        "wind_speed": data["wind"]["speed"]
    }
    return weather_info