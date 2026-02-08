import os
from google import genai
from google.genai import types
from dotenv import load_dotenv 

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)

class ImageAnalysisModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def one_image_analysis(self, image_bytes: bytes, mime_type: str):
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )

        response = client.models.generate_content(
            model=self.model_name,
            contents=[
                """"You are an AI-powered crop disease detection assistant. Analyze the uploaded crop image and identify any diseases present. Provide the following information:

                    Crop type (e.g., tomato, rice, wheat).

                    Disease name or 'healthy' if no disease is detected.

                    Severity level (mild, moderate, severe) if applicable.

                    Suggested treatment or preventive measures.
                    Respond concisely in JSON format for integration into real-time applications.
""",
                image_part
            ]
        )

        return response.text
