"""
Assignment 3: Deployment API
Serves the refined Student Dropout Prediction Model via FastAPI
Samsung Innovation Campus — AI Course
"""

import os
import uvicorn
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.joblib")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    # We will raise an informative warning or handle it when the API starts
    print(f"Warning: Model artifact not found at {MODEL_PATH}. Make sure to run assignment3_refinement.py first.")

app = FastAPI(
    title="UPV Student Trajectory Abandonment Prediction API",
    description="API for predicting whether a student will abandon their academic trajectory, using Spanish UPV 2022 dataset features.",
    version="1.0.0"
)

# Define request schema with the expected top 20 features from the model
class StudentFeatures(BaseModel):
    # Top 20 features from the RF importance on the UPV dataset
    cred_pend_sup_tit: float = Field(..., description="Pending credits for title")
    nota14_hash: float = Field(..., description="14-scale admission grade (5-14)")
    rendimiento_cuat_b: float = Field(..., description="Second semester performance percentage (0-100)")
    nota10_hash: float = Field(..., description="10-scale admission grade (5-10)")
    cred_mat_normal: float = Field(..., description="Regular credits enrolled")
    cred_sup_tit: float = Field(..., description="Credits completed for title")
    cred_sup_sem_b: float = Field(..., description="Semester B completed credits")
    nota_asig_hash: float = Field(..., description="Subject grade average (0-10)")
    estudios_p_hash: int = Field(..., description="Father's education level code")
    estudios_m_hash: int = Field(..., description="Mother's education level code")
    anyo_ingreso: int = Field(..., description="University admission year")
    cred_mat_sem_b: float = Field(..., description="Semester B enrolled credits")
    cred_sup_normal: float = Field(..., description="Regular completed credits")
    cred_sup_total: float = Field(..., description="Total completed credits")
    cred_sup: float = Field(..., description="Completed credits indicator")
    asig1: float = Field(..., description="First year courses registered count")
    cred_mat_sem_a: float = Field(..., description="Semester A enrolled credits")
    tipo_ingreso: int = Field(..., description="Admission type code")
    cred_sup_1o: float = Field(..., description="First year completed credits")
    cred_mat_total: float = Field(..., description="Total enrolled credits")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cred_pend_sup_tit": 12.0,
                "nota14_hash": 10.5,
                "rendimiento_cuat_b": 80.0,
                "nota10_hash": 7.5,
                "cred_mat_normal": 60.0,
                "cred_sup_tit": 228.0,
                "cred_sup_sem_b": 30.0,
                "nota_asig_hash": 7.2,
                "estudios_p_hash": 2,
                "estudios_m_hash": 3,
                "anyo_ingreso": 2021,
                "cred_mat_sem_b": 30.0,
                "cred_sup_normal": 60.0,
                "cred_sup_total": 60.0,
                "cred_sup": 60.0,
                "asig1": 10.0,
                "cred_mat_sem_a": 30.0,
                "tipo_ingreso": 1,
                "cred_sup_1o": 54.0,
                "cred_mat_total": 60.0
            }
        }
    )

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class: A (abandoned) or B (continuing)")
    confidence: float = Field(..., description="Probability of predicted class")
    probabilities: dict[str, float] = Field(..., description="Probabilities for each class")
    model_used: str = Field(..., description="Name of the model used for prediction")

# Global variables to store loaded models
artifact = None
model = None
selected_features = None
class_names = None
label_encoder = None

def load_model():
    global artifact, model, selected_features, class_names, label_encoder
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Please train the model first.")
    
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    selected_features = artifact["selected_features"]
    class_names = artifact["class_names"]
    label_encoder = artifact["label_encoder"]
    print(f"Model loaded: {artifact['model_name']}")

@app.get("/health", summary="Health Check")
def health_check():
    """Returns the API health status and the loaded model name."""
    try:
        if model is None:
            load_model()
        return {
            "status": "healthy",
            "model_loaded": artifact["model_name"]
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }

@app.post("/predict", response_model=PredictionResponse, summary="Predict Student Trajectory Outcome")
def predict(student: StudentFeatures):
    """
    Accepts student data and predicts their academic outcome (A: Abandoned, B: Continuing).
    Feds the top 20 features into the pre-processing and model pipeline.
    """
    try:
        if model is None:
            load_model()

        # Convert Pydantic model to dict
        raw_dict = student.model_dump()
        
        # Verify and align all 20 selected features
        input_data = {}
        for feat in selected_features:
            if feat not in raw_dict:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing feature required by model: {feat}"
                )
            input_data[feat] = raw_dict[feat]
            
        # Convert to pandas DataFrame with correct feature order
        df_input = pd.DataFrame([input_data])[selected_features]
        
        # Predict class (pipeline handles scaling and prediction)
        pred_enc = model.predict(df_input)[0]
        prediction_label = label_encoder.inverse_transform([pred_enc])[0]
        
        # Get class probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_input)[0]
            prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
            confidence = float(probs[pred_enc])
        else:
            prob_dict = {prediction_label: 1.0}
            confidence = 1.0
            
        return PredictionResponse(
            prediction=prediction_label,
            confidence=confidence,
            probabilities=prob_dict,
            model_used=artifact["model_name"]
        )
        
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=503, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    # Start the Uvicorn server on localhost:8000
    uvicorn.run("assignment3_deployment:app", host="127.0.0.1", port=8000, reload=True)
