from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # IMPORT CORS
from pydantic import BaseModel
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd

app = FastAPI()

# ----------------------------
# 1. ENABLE CORS (Crucial for React connection)
# ----------------------------
origins = [
    "http://localhost:5173", # Vite default
    "http://localhost:3000", # CRA default
    "*"                      # Allow all (for development)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# 2. LOAD MODELS
# ----------------------------
# Ensure these files exist in your root directory
try:
    bst7 = xgb.Booster()
    bst7.load_model("xgb_oilseed_7d.model")

    bst15 = xgb.Booster()
    bst15.load_model("xgb_oilseed_15d.model")

    scaler = joblib.load("scaler.joblib")
    ohe = joblib.load("ohe_state_crop.joblib")
    feature_cols = joblib.load("xgb_feature_cols.joblib")
    print("✅ Models and artifacts loaded successfully.")
except Exception as e:
    print(f"❌ Error loading artifacts: {e}")

# ----------------------------
# 3. DATA TYPES
# ----------------------------
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

# ----------------------------
# 4. PREPROCESSING LOGIC
# ----------------------------
def build_feature_vector(req: PredictionRequest):
    # Convert request -> dataframe row
    row = pd.DataFrame([req.dict()])

    # Scale numeric columns
    num_cols = [
        'modal_price','min_price','max_price','arrivals',
        'lag_7','lag_14','lag_30',
        'temperature','humidity','rainfall','day_of_year'
    ]
    
    # Check if scaler expects these columns (basic validation)
    row[num_cols] = scaler.transform(row[num_cols])

    # One-hot encode state & crop
    # Note: Ensure 'state' and 'crop' values match exactly what the OHE was trained on
    ohe_cols = ohe.get_feature_names_out(['state','crop'])
    ohe_vals = ohe.transform(row[['state','crop']])

    ohe_df = pd.DataFrame(ohe_vals.toarray(), columns=ohe_cols)

    row = pd.concat([row.reset_index(drop=True), ohe_df], axis=1)

    # Reorder columns to match XGBoost training order
    row = row[feature_cols]

    return xgb.DMatrix(row)

# ----------------------------
# 5. ENDPOINT
# ----------------------------
@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        x = build_feature_vector(req)

        pred7_log = bst7.predict(x)[0]
        pred15_log = bst15.predict(x)[0]

        # Convert back from log (Inverse Log Transformation)
        pred7 = float(np.expm1(pred7_log))
        pred15 = float(np.expm1(pred15_log))

        return {
            "predicted_7_day_price": round(pred7, 2),
            "predicted_15_day_price": round(pred15, 2)
        }
    except Exception as e:
        return {"error": str(e)}