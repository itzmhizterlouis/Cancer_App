import os
import joblib
import logging
import numpy as np
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OncoPredict | Advanced Cancer Diagnostic AI")

# Templates
templates = Jinja2Templates(directory="templates")

# Load the trained model
MODEL_PATH = "cancermodel.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            logger.info(f"Successfully loaded model from {MODEL_PATH}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    else:
        logger.warning(f"Model file {MODEL_PATH} not found.")
        return None

model = load_model()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request, 
    mean_radius: float = Form(...),
    mean_texture: float = Form(...),
    mean_perimeter: float = Form(...),
    mean_area: float = Form(...),
    mean_smoothness: float = Form(...),
    mean_compactness: float = Form(...)
):
    if model is None:
        logger.error("Prediction attempted but model is not loaded.")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction_text": "Error: Diagnostic model is currently unavailable."}
        )

    try:
        # Collect features and log the request
        features = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness]
        logger.info(f"Prediction requested with features: {features}")
        
        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)
        
        # Interpret result (Assuming 0: Malignant, 1: Benign as per original code)
        output = "Malignant (Cancerous)" if prediction[0] == 0 else "Benign (Non-Cancerous)"
        logger.info(f"Prediction result: {output}")

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction_text": f"Prediction: {output}"}
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction_text": f"Error during analysis: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
