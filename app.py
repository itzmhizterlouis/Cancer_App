from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
import numpy as np
import os
import joblib

app = FastAPI(title="Cancer Prediction App")

# Templates (assumes you have templates/index.html)
templates = Jinja2Templates(directory="templates")

# Load the trained model
model_filename = "cancermodel.pkl"

if os.path.exists(model_filename):
    model = joblib.load(model_filename)
else:
    print(f"Error: '{model_filename}' not found. Please run 'python cancermodel.py' first.")
    model = None

@app.get("", include_in_schema=False) 
async def redirect_root(): 
    return RedirectResponse(url="/")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, 
                  mean_radius: float = Form(...),
                  mean_texture: float = Form(...),
                  mean_perimeter: float = Form(...),
                  mean_area: float = Form(...),
                  mean_smoothness: float = Form(...),
                  mean_compactness: float = Form(...)):
    """
    Example: adjust the form parameters to match your actual feature names.
    """
    if model is None:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction_text": "Error: Model not loaded."}
        )

    try:
        # Collect features into numpy array
        features = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness]
        final_features = np.array(features).reshape(1, -1)

        prediction = model.predict(final_features)
        output = "Malignant (Cancerous)" if prediction[0] == 0 else "Benign (Safe)"

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction_text": f"Prediction: {output}"}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction_text": f"Error: {str(e)}"}
        )

# Run with: uvicorn main:app --reload
