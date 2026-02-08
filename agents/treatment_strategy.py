import os
from google import genai
from google.genai import types
from dotenv import load_dotenv 

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)

class TreatmentStrategyModel:
    def __init__(self):
        pass 

    def treatment_reason(self, input_text: str):
        response = client.models.generate_content(
            model="gemini-3-flash-preview", contents=["""You are an AI Treatment Strategy engine.

        You will receive a single block of text describing a situation related to
        climate, weather, crop health, disease spread, or environmental stress.
        The text may be informal, incomplete, or unstructured.

        Your task is to generate an explainable treatment and mitigation strategy
        based solely on the information present in the text.

        Instructions:
        1. Identify and summarize key risk signals mentioned or implied in the text.
        2. Infer the most likely disease or stress mechanisms influenced by environmental conditions.
        3. Generate treatment and mitigation recommendations focused on prevention and early response.
        4. Clearly explain the reasoning behind each recommendation.
        5. Prioritize actions by urgency and impact.

        Treatment Guidelines:
        - Focus on practical, low-cost, and scalable actions.
        - Separate immediate actions from short-term strategies.
        - Include both treatment and preventive measures.
        - Assume limited resources unless stated otherwise.

        Output Format:
        - Interpreted Situation Summary
        - Risk Reasoning
        - Recommended Treatment & Mitigation Actions
        • Immediate Actions
        • Short-term Actions
        - Explanation of Recommendations
        - Warning Signs & Monitoring Indicators
        - Confidence Level (Low / Medium / High)

        Safety Constraints:
        - Do NOT diagnose diseases.
        - Do NOT prescribe specific chemicals or medication dosages.
        - Use probabilistic language (e.g., "likely", "may increase risk").
        - Keep recommendations advisory and non-authoritative.""",input_text],
        )

        return response.text