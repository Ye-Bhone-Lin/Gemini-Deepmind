from dotenv import load_dotenv 
import os 
from google import genai 

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

def assess_risk_analysis(weather: dict, city: str):
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
    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
    model="gemini-3-flash-preview", contents=prompt)
    return response.text