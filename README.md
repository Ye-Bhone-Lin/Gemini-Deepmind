# Gemini Deepmind API

This API provides tools for agricultural analysis, including crop disease detection from images, disease risk assessment based on weather, and treatment strategy generation. It utilizes Google's Gemini models for intelligent analysis.

## Setup

1.  **Environment Variables**: Ensure you have a `.env` file with the following keys:
    *   `GEMINI_API_KEY`: API key for Google Gemini.
    *   `OPENWEATHER_MAP_API_KEY`: API key for OpenWeatherMap.

2.  **Dependencies**: Install the required packages (e.g., `fastapi`, `uvicorn`, `google-genai`, `python-dotenv`, `requests`).

3.  **Run**: Start the server using Uvicorn:
    ```bash
    uvicorn main:app --reload
    ```

## API Endpoints

### 1. Image Analysis
Analyzes an uploaded crop image to identify diseases and suggest treatments.

*   **URL**: `/image_analysis`
*   **Method**: `POST`
*   **Request Body**: `multipart/form-data`
    *   `file`: The image file (UploadFile) to analyze.
*   **Response**: JSON object containing the analysis result from the Gemini model.

**Example Response:**
```json
{
  "result": "{\n  \"Crop type\": \"Tomato\",\n  \"Disease name\": \"Early Blight\",\n  \"Severity level\": \"Moderate\",\n  \"Suggested treatment\": \"Apply copper-based fungicides...\"\n}"
}
```

### 2. Disease Risk Assessment
Predicts the risk of disease outbreak for a specific city based on current weather conditions.

*   **URL**: `/disease-risk/{city}`
*   **Method**: `GET`
*   **Path Parameters**:
    *   `city`: Name of the city (string).
*   **Response**: JSON object with city name, weather data, and risk assessment.

**Example Response:**
```json
{
  "city": "Yangon",
  "weather": {
    "temp": 30.5,
    "humidity": 75,
    "rain": 0.0,
    "wind_speed": 3.2
  },
  "risk_assessment": "Risk Level: Medium\nReasoning: High humidity and warm temperatures create favorable conditions for fungal growth..."
}
```

### 3. Treatment Strategy
Generates an explainable treatment and mitigation strategy based on a text description of a situation.

*   **URL**: `/treatmentstrategy`
*   **Method**: `POST`
*   **Request Body**: JSON
    ```json
    {
      "input_text": "Heavy rainfall expected next week in the rice fields, currently high humidity."
    }
    ```
*   **Response**: String containing the detailed strategy and recommendations.

**Example Response:**
```text
Interpreted Situation Summary: ...
Risk Reasoning: ...
Recommended Treatment & Mitigation Actions: ...
```
