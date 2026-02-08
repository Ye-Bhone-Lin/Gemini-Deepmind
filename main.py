from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from contextlib import asynccontextmanager
from agents.image_model import ImageAnalysisModel
from agents.weather_model import assess_risk_analysis
from agents.treatment_strategy import TreatmentStrategyModel
from tools.tools import get_weather
from schema.treatment_schema import inputTextRequest

gen_model = {}

@asynccontextmanager
async def startup_lifespan(app: FastAPI):
    image_analysis_model = ImageAnalysisModel(model_name="gemini-3-flash-preview")
    treatment_strategy_model = TreatmentStrategyModel()
    gen_model['image_analysis_model'] = image_analysis_model
    gen_model['treatment_strategy_model'] = treatment_strategy_model

    yield

    gen_model.clear()

app = FastAPI(lifespan=startup_lifespan)

@app.post("/image_analysis")
async def image_analysis(file: UploadFile = File(...)):
    image_bytes = await file.read()

    result = gen_model["image_analysis_model"].one_image_analysis(
        image_bytes,
        file.content_type
    )

    return {"result": result}

@app.get("/disease-risk/{city}")
def disease_risk(city: str):
    weather = get_weather(city)
    risk_assessment = assess_risk_analysis(weather, city)
    return {
        "city": city,
        "weather": weather,
        "risk_assessment": risk_assessment
    }

@app.post("/treatmentstrategy")
def treatment(request: inputTextRequest):
    response = gen_model['treatment_strategy_model'].treatment_reason(request.input_text)
    return response