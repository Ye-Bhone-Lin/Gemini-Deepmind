from fastapi import FastAPI, HTTPException
import requests
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = "AIzaSyBw-tgP-z_eO_Q7OYO5-dBHyKp8P8ZLgWM"
WEATHER_API_KEY = "a0a2c97f9db08aefb16b7fa7fbd9e03d"

app = FastAPI(title="Climate & Disease Risk Predictor with Gemini")

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

def assess_risk_with_gemini(weather: dict, city: str):
    prompt = f"""
    You are an expert in epidemiology and climate. 
    Given the weather in {city}:
    - Temperature: {weather['temp']}°C
    - Humidity: {weather['humidity']}%
    - Rainfall (last 1h): {weather['rain']}mm
    - Wind speed: {weather['wind_speed']} m/s

    Predict the risk level of disease outbreak (Low, Medium, High). 
    Explain your reasoning and give short actionable recommendations.
    """
    client = genai.Client(api_key="AIzaSyBw-tgP-z_eO_Q7OYO5-dBHyKp8P8ZLgWM")

    response = client.models.generate_content(
    model="gemini-3-flash-preview", contents=prompt)
    return response.text

@app.get("/disease-risk/{city}")
def disease_risk(city: str):
    weather = get_weather(city)
    risk_assessment = assess_risk_with_gemini(weather, city)
    return {
        "city": city,
        "weather": weather,
        "risk_assessment": risk_assessment
    }


