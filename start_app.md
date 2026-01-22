# Start App - OncoPredict AI

This guide will help you get the Cancer Prediction application up and running.

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Navigate to the project directory:
   ```bash
   cd "Cancer App"
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the model file (`cancermodel.pkl`) exists. If not, train the model:
   ```bash
   python cancermodel.py
   ```

## Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn app:app --reload
```

The application will be available at `http://127.0.0.1:8000`.

## Features

- **Advanced UI**: Modern, responsive design with medical-themed aesthetics.
- **AI Diagnostics**: Predictive analysis based on mean radius, texture, perimeter, area, smoothness, and compactness.
- **Robust Backend**: Fast and efficient API handling with detailed logging.
