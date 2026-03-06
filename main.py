from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
import os

app = FastAPI()

# -----------------------------------------------------------
# 0. BASE DIRECTORY FIX (ensures correct file loading)
# -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"📂 BASE_DIR = {BASE_DIR}")

# -----------------------------------------------------------
# 1. CORS
# -----------------------------------------------------------
origins = ["http://localhost:5173", "http://localhost:3000", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------
# 2. OILSEED MODEL SETUP
# -----------------------------------------------------------
bst7 = None
bst15 = None
scaler = None
ohe = None
feature_cols_oil = None

try:
    bst7 = xgb.Booster()
    bst7.load_model(os.path.join(BASE_DIR, "xgb_oilseed_7d.model"))

    bst15 = xgb.Booster()
    bst15.load_model(os.path.join(BASE_DIR, "xgb_oilseed_15d.model"))

    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
    ohe = joblib.load(os.path.join(BASE_DIR, "ohe_state_crop.joblib"))
    feature_cols_oil = joblib.load(os.path.join(BASE_DIR, "xgb_feature_cols.joblib"))

    print("✅ Oilseed models loaded.")
except Exception as e:
    print(f"❌ Oilseed model load error: {e}")

class PredictionRequest(BaseModel):
    state: str
    district: str
    mandi: str
    crop: str
    modal_price: float
    min_price: float
    max_price: float
    arrivals: float
    lag_7: float
    lag_14: float
    lag_30: float
    temperature: float
    humidity: float
    rainfall: float
    day_of_year: int
    festival_flag: int
    harvest_season_flag: int

def build_feature_vector(req: PredictionRequest):
    row = pd.DataFrame([req.dict()])

    num_cols = [
        "modal_price", "min_price", "max_price", "arrivals",
        "lag_7", "lag_14", "lag_30",
        "temperature", "humidity", "rainfall", "day_of_year"
    ]

    row[num_cols] = scaler.transform(row[num_cols])

    ohe_cols = ohe.get_feature_names_out(["state", "crop"])
    ohe_vals = ohe.transform(row[["state", "crop"]])

    try:
        ohe_vals = ohe_vals.toarray()
    except:
        pass

    ohe_df = pd.DataFrame(ohe_vals, columns=ohe_cols)
    row = pd.concat([row.reset_index(drop=True), ohe_df], axis=1)
    row = row[feature_cols_oil]

    return xgb.DMatrix(row)

@app.post("/predict")
def predict_oil(req: PredictionRequest):
    try:
        if not bst7:
            raise HTTPException(500, "Oilseed models not loaded.")

        x = build_feature_vector(req)
        pred7 = float(np.expm1(bst7.predict(x)[0]))
        pred15 = float(np.expm1(bst15.predict(x)[0]))

        return {
            "predicted_7_day_price": round(pred7, 2),
            "predicted_15_day_price": round(pred15, 2)
        }
        
    except Exception as e:
        print(f"❌ Oilseed prediction error: {e}")
        raise HTTPException(500, str(e))


# -----------------------------------------------------------
# 3. TRADER MODEL SETUP
# -----------------------------------------------------------

def get_path(filename):
    search_paths = [
        filename,
        os.path.join("artifacts", filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, "artifacts", filename),
    ]

    for p in search_paths:
        if os.path.exists(p):
            return p

    print(f"❌ Missing file: {filename}")
    return None

model_trader_7d = None
model_trader_15d = None
feature_cols_trader = None
df_latest = pd.read_csv("trader_latest_features.csv")
feature_cols = joblib.load("trader_feature_cols.joblib")

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
class PredictRequest(BaseModel):
    crop: str  # the selected crop name (mustard, soybean…)
model7 = joblib.load("xgb_trader_7day.joblib")
model15 = joblib.load("xgb_trader_15day.joblib")
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
        "unit": "₹ per quint"
    }


# ----------------------------------------------------
# ROOT URL
# ----------------------------------------------------


@app.get("/")
def home():
    return {"status": "Trade Price Prediction API Running"}