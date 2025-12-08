from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib

app = FastAPI()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # or ["http://localhost:5173"]
    allow_methods=["*"],        # <-- allows OPTIONS
    allow_headers=["*"],
)
# ----------------------------------------------------
# LOAD MODELS AND ARTIFACTS
# ----------------------------------------------------

model7 = joblib.load("xgb_trader_7day.joblib")
model15 = joblib.load("xgb_trader_15day.joblib")
feature_cols = joblib.load("trader_feature_cols.joblib")

# latest feature vector per crop (as generated in your notebook)
df_latest = pd.read_csv("trader_latest_features.csv")

print("Models & feature data loaded successfully.")


# ----------------------------------------------------
# REQUEST BODY FORMAT
# ----------------------------------------------------

class PredictRequest(BaseModel):
    crop: str  # the selected crop name (mustard, soybean…)


# ----------------------------------------------------
# UTILITY: GET LATEST FEATURE ROW FOR THE CROP
# ----------------------------------------------------

def get_feature_vector(crop_name: str):
    crop_name = crop_name.lower().strip()

    matches = df_latest[df_latest["crop_name"] == crop_name]

    if matches.empty:
        raise ValueError(
            f"Crop '{crop_name}' not found. Available: {df_latest['crop_name'].unique().tolist()}"
        )

    row = matches.iloc[0]

    # Prepare the feature row in correct order
    X = row[feature_cols].values.reshape(1, -1)
    return X


# ----------------------------------------------------
# API ENDPOINT FOR 7-DAY & 15-DAY FORECAST
# ----------------------------------------------------

@app.post("/predict_trade_price")
def predict_trade_price(req: PredictRequest):
    try:
        X = get_feature_vector(req.crop)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Predictions
    pred7 = float(model7.predict(X)[0])
    pred15 = float(model15.predict(X)[0])

    return {
        "crop": req.crop,
        "predicted_7_day_price": round(pred7, 2),
        "predicted_15_day_price": round(pred15, 2),
        "unit": "₹ per quintal"
    }


# ----------------------------------------------------
# ROOT URL
# ----------------------------------------------------

@app.get("/")
def home():
    return {"status": "Trade Price Prediction API Running"}